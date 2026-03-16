import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from others.runtime import resolve_vit_attention_backend, sdpa_context


TASK_NAMES = ("dt-st", "dt-lt", "nt-st", "nt-lt", "ad-st", "ad-lt")
ATOM_NAMES = ("vm", "im", "cm", "sc", "cc")
ATOM_TO_INDEX = {name: idx for idx, name in enumerate(ATOM_NAMES)}
TASK_TO_ATOMS = {
    "dt-st": ("vm", "sc"),
    "dt-lt": ("vm", "cc"),
    "nt-st": ("im", "sc"),
    "nt-lt": ("im", "cc"),
    "ad-st": ("cm", "sc"),
    "ad-lt": ("cm", "cc"),
}


def _build_task_expert_masks():
    expert_pairs = [None]
    for atom_in in ATOM_NAMES:
        for atom_out in ATOM_NAMES:
            expert_pairs.append((atom_in, atom_out))

    masks = []
    indices = []
    for task_name in TASK_NAMES:
        active_atoms = set(TASK_TO_ATOMS[task_name])
        task_mask = [0]
        for pair in expert_pairs[1:]:
            atom_in, atom_out = pair
            task_mask.append(int(atom_in != atom_out and atom_in in active_atoms and atom_out in active_atoms))
        masks.append(task_mask)
        indices.append([idx for idx, flag in enumerate(task_mask) if flag == 1])
    return expert_pairs, masks, indices


EXPERT_PAIRS, TASK_EXPERT_MASKS, TASK_EXPERT_INDICES = _build_task_expert_masks()


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, attention_backend="auto"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attention_backend = str(attention_backend).strip().lower()
        self.attention_info = resolve_vit_attention_backend(
            self.attention_backend,
            use_sdpa_available=bool(hasattr(F, "scaled_dot_product_attention")),
        )

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        use_sdpa = (
            self.attention_info["uses_sdpa"]
            and x.is_cuda
            and hasattr(F, "scaled_dot_product_attention")
        )
        if use_sdpa:
            with sdpa_context(self.attention_backend):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        return x


class Mlp_moe(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, ncls=6, moae=False, moae_router_noise=0.01):
        super().__init__()
        self.ncls = ncls
        self.moae = moae
        self.moae_router_noise = max(0.0, float(moae_router_noise))

        self.patch_fc1 = nn.Linear(in_dim, hidden_dim)
        self.patch_act = nn.GELU()
        self.patch_fc2 = nn.Linear(hidden_dim, out_dim)

        if self.moae:
            self.num_atoms = len(ATOM_NAMES)
            task_atom_pairs = [TASK_TO_ATOMS[TASK_NAMES[i]] for i in range(self.ncls)]
            left_atom_indices = torch.tensor([ATOM_TO_INDEX[left] for left, _ in task_atom_pairs], dtype=torch.long)
            right_atom_indices = torch.tensor([ATOM_TO_INDEX[right] for _, right in task_atom_pairs], dtype=torch.long)
            self.register_buffer("task_left_route_keys", left_atom_indices * self.num_atoms + right_atom_indices, persistent=False)
            self.register_buffer("task_right_route_keys", right_atom_indices * self.num_atoms + left_atom_indices, persistent=False)
            route_keys = sorted({int(key.item()) for key in torch.cat((self.task_left_route_keys, self.task_right_route_keys), dim=0)})
            self.route_specs = tuple((route_key, *divmod(route_key, self.num_atoms)) for route_key in route_keys)

            self.gate_delta = nn.Parameter(torch.empty(self.ncls, in_dim))
            trunc_normal_(self.gate_delta, std=0.02)

            self.atom_in_layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in ATOM_NAMES])
            self.atom_out_layers = nn.ModuleList([nn.Linear(hidden_dim, in_dim) for _ in ATOM_NAMES])
        else:
            self.num_atoms = len(ATOM_NAMES)
            self.route_specs = ()
            self.gate_delta = None
            self.atom_in_layers = None
            self.atom_out_layers = None

        self.drop = nn.Dropout(0.03)

    def rescale_output_layers(self, layer_id):
        scale = math.sqrt(2.0 * layer_id)
        self.patch_fc2.weight.data.div_(scale)
        if not self.moae or self.atom_out_layers is None:
            return
        for layer in self.atom_out_layers:
            layer.weight.data.div_(scale)

    def load_pretrained_fc1(self, weight=None, bias=None):
        self.patch_fc1.weight.data.copy_(weight)
        if bias is not None and self.patch_fc1.bias is not None:
            self.patch_fc1.bias.data.copy_(bias)
        if not self.moae or self.atom_in_layers is None:
            return
        for layer in self.atom_in_layers:
            layer.weight.data.copy_(weight)
            if bias is not None and layer.bias is not None:
                layer.bias.data.copy_(bias)

    def load_pretrained_fc2(self, weight=None, bias=None):
        self.patch_fc2.weight.data.copy_(weight)
        if bias is not None and self.patch_fc2.bias is not None:
            self.patch_fc2.bias.data.copy_(bias)
        if not self.moae or self.atom_out_layers is None:
            return
        for layer in self.atom_out_layers:
            layer.weight.data.copy_(weight)
            if bias is not None and layer.bias is not None:
                layer.bias.data.copy_(bias)

    def _apply_patch_mlp(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.patch_fc2(self.patch_act(self.patch_fc1(tokens)))

    def _compute_task_gate_logits(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        gate_logits = torch.einsum("bnd,nd->bn", cls_tokens.float(), self.gate_delta.float())
        if self.training and self.moae_router_noise > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.moae_router_noise
        return gate_logits

    def _forward_weighted_top1_moae(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        gate_delta = self._compute_task_gate_logits(cls_tokens)
        choose_left = gate_delta >= 0
        p_left = gate_delta.sigmoid()
        chosen_weights = torch.where(choose_left, p_left, 1.0 - p_left).to(dtype=cls_tokens.dtype)
        left_route_keys = self.task_left_route_keys.unsqueeze(0).expand(cls_tokens.shape[0], -1)
        right_route_keys = self.task_right_route_keys.unsqueeze(0).expand(cls_tokens.shape[0], -1)
        chosen_route_keys = torch.where(choose_left, left_route_keys, right_route_keys)

        flat_tokens = cls_tokens.reshape(-1, cls_tokens.shape[-1])
        flat_route_keys = chosen_route_keys.reshape(-1)
        flat_weights = chosen_weights.reshape(-1, 1)
        mixed_flat = torch.empty_like(flat_tokens)
        for route_key, src_idx, dst_idx in self.route_specs:
            route_positions = torch.nonzero(flat_route_keys == route_key, as_tuple=False).flatten()
            if route_positions.numel() == 0:
                continue
            route_tokens = flat_tokens.index_select(0, route_positions)
            route_hidden = F.gelu(self.atom_in_layers[src_idx](route_tokens))
            route_outputs = self.atom_out_layers[dst_idx](route_hidden) * flat_weights.index_select(0, route_positions)
            mixed_flat.index_copy_(0, route_positions, route_outputs)
        return mixed_flat.reshape_as(cls_tokens)

    def forward(self, x):
        cls_tokens = x[:, :self.ncls]
        patch_tokens = self._apply_patch_mlp(x[:, self.ncls:])

        if not self.moae or self.atom_in_layers is None or self.atom_out_layers is None:
            cls_tokens = self._apply_patch_mlp(cls_tokens)
        else:
            cls_tokens = self._forward_weighted_top1_moae(cls_tokens)
        x = torch.cat((cls_tokens, patch_tokens), dim=1)
        return self.drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        dpr=0.0,
        ncls=6,
        moae=False,
        moae_router_noise=0.01,
        attention_backend="auto",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attention_backend=attention_backend)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp_moe(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            ncls=ncls,
            moae=moae,
            moae_router_noise=moae_router_noise,
        )
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed_overlap(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        print(f'using stride: {stride_size}, and patch number is num_y{self.num_y} * num_x{self.num_x}')
        self.num_patches = self.num_x * self.num_y
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        ncls=6,
        moae=False,
        moae_router_noise=0.01,
        attention_backend="auto",
    ):
        super().__init__()
        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size,
            patch_size=patch_size,
            stride_size=stride_size,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, ncls, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + ncls, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dpr=dpr[i],
                    ncls=ncls,
                    moae=moae,
                    moae_router_noise=moae_router_noise,
                    attention_backend=attention_backend,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ncls = ncls
        self.moae = moae
        self.attention_backend = str(attention_backend).strip().lower()
        first_attention = self.blocks[0].attn.attention_info if self.blocks else {"requested": "eager", "active": "eager", "uses_sdpa": False}
        self.attention_info = dict(first_attention)
        self.attention_info["num_blocks"] = len(self.blocks)

        self.apply(self._init_weights)
        self.fix_init_weight()
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    def fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            scale = math.sqrt(2.0 * (layer_id + 1))
            layer.attn.proj.weight.data.div_(scale)
            layer.mlp.rescale_output_layers(layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for key, value in param_dict.items():
            if 'head' in key:
                continue
            if key == 'pos_embed' and value.shape != self.pos_embed.shape:
                value = interpolate_pos_embed(self, value)
            if key == 'cls_token':
                value = value.expand(-1, self.ncls, -1)
            try:
                if "blocks" in key and ".mlp.fc1." in key:
                    block_idx = int(key.split('.')[1])
                    if key.endswith(".weight"):
                        self.blocks[block_idx].mlp.load_pretrained_fc1(weight=value)
                    else:
                        self.blocks[block_idx].mlp.load_pretrained_fc1(
                            weight=self.blocks[block_idx].mlp.patch_fc1.weight.data,
                            bias=value,
                        )
                elif "blocks" in key and ".mlp.fc2." in key:
                    block_idx = int(key.split('.')[1])
                    if key.endswith(".weight"):
                        self.blocks[block_idx].mlp.load_pretrained_fc2(weight=value)
                    else:
                        self.blocks[block_idx].mlp.load_pretrained_fc2(
                            weight=self.blocks[block_idx].mlp.patch_fc2.weight.data,
                            bias=value,
                        )
                elif "blocks" in key:
                    self.state_dict()[key].copy_(value)
                else:
                    self.state_dict()[key].copy_(value)
            except Exception:
                print('===========================ERROR=========================')
                print(
                    'shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(
                        key, value.shape, self.state_dict()[key].shape
                    )
                )


def interpolate_pos_embed(self, pos_embed_checkpoint):
    orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
    new_size = (self.patch_embed.num_y, self.patch_embed.num_x)
    extra_tokens = pos_embed_checkpoint[:, :1]
    extra_tokens = extra_tokens.expand(-1, self.ncls, -1)
    pos_tokens = pos_embed_checkpoint[0, 1:]
    pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=new_size, mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat([extra_tokens, pos_tokens], dim=1)
    print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size[0] * new_size[1]))
    return new_pos_embed


def vit_base_patch16_224_ReID_moe(
    img_size=(256, 128),
    stride_size=16,
    drop_path_rate=0.1,
    ncls=1,
    moae=False,
    moae_router_noise=0.01,
    attention_backend="auto",
):
    return ViT(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
        ncls=ncls,
        moae=moae,
        moae_router_noise=moae_router_noise,
        attention_backend=attention_backend,
    )
