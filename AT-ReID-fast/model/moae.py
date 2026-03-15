import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from others.runtime import resolve_vit_attention_backend, sdpa_context


TASK_NAMES = ("dt-st", "dt-lt", "nt-st", "nt-lt", "ad-st", "ad-lt")
ATOM_NAMES = ("vm", "im", "cm", "sc", "cc")
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
    def __init__(self, in_dim, hidden_dim, out_dim, ncls=6, moae=False, k=1):
        super().__init__()
        self.ncls = ncls
        self.moae = moae
        self.k = int(k)

        if self.moae:
            self.nexp = len(EXPERT_PAIRS)
            self.gt = [list(mask) for mask in TASK_EXPERT_MASKS[: self.ncls]]
        else:
            self.nexp = 1
            self.gt = [[1] for _ in range(self.ncls)]
        self.task_expert_indices = [[idx for idx, flag in enumerate(mask) if flag == 1] for mask in self.gt]
        self.active_cls_expert_indices = sorted({idx for indices in self.task_expert_indices for idx in indices})
        self.active_atom_inputs = sorted(
            {
                EXPERT_PAIRS[expert_idx][0]
                for expert_idx in self.active_cls_expert_indices
                if expert_idx > 0 and EXPERT_PAIRS[expert_idx] is not None
            }
        )
        self.expert_pairs = {
            expert_idx: EXPERT_PAIRS[expert_idx]
            for expert_idx in self.active_cls_expert_indices
            if expert_idx > 0 and EXPERT_PAIRS[expert_idx] is not None
        }

        self.gate = nn.ModuleList([nn.Linear(in_dim, len(self.task_expert_indices[i]), bias=False) for i in range(self.ncls)])

        hidden_dim2 = hidden_dim
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        ])

        if self.moae:
            atom_in_layers = nn.ModuleDict({name: nn.Linear(in_dim, hidden_dim2) for name in ATOM_NAMES})
            atom_out_layers = nn.ModuleDict({name: nn.Linear(hidden_dim2, in_dim) for name in ATOM_NAMES})
            self.atom_in_layers = atom_in_layers
            self.atom_out_layers = atom_out_layers
            for atom_in in ATOM_NAMES:
                for atom_out in ATOM_NAMES:
                    self.ffn.append(
                        nn.Sequential(
                            self.atom_in_layers[atom_in],
                            nn.GELU(),
                            self.atom_out_layers[atom_out],
                        )
                    )
        else:
            self.atom_in_layers = None
            self.atom_out_layers = None

        self.drop = nn.Dropout(0.03)

    def rescale_output_layers(self, layer_id):
        scale = math.sqrt(2.0 * layer_id)
        self.ffn[0][2].weight.data.div_(scale)
        if not self.moae or self.atom_out_layers is None:
            return
        seen = set()
        for layer in self.atom_out_layers.values():
            if id(layer) in seen:
                continue
            layer.weight.data.div_(scale)
            seen.add(id(layer))

    def load_pretrained_fc1(self, weight=None, bias=None):
        self.ffn[0][0].weight.data.copy_(weight)
        if bias is not None and self.ffn[0][0].bias is not None:
            self.ffn[0][0].bias.data.copy_(bias)
        if not self.moae or self.atom_in_layers is None:
            return
        for layer in self.atom_in_layers.values():
            layer.weight.data.copy_(weight)
            if bias is not None and layer.bias is not None:
                layer.bias.data.copy_(bias)

    def load_pretrained_fc2(self, weight=None, bias=None):
        self.ffn[0][2].weight.data.copy_(weight)
        if bias is not None and self.ffn[0][2].bias is not None:
            self.ffn[0][2].bias.data.copy_(bias)
        if not self.moae or self.atom_out_layers is None:
            return
        for layer in self.atom_out_layers.values():
            layer.weight.data.copy_(weight)
            if bias is not None and layer.bias is not None:
                layer.bias.data.copy_(bias)

    def _sparsify_gate(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-1] <= 1 or self.k <= 0 or self.k >= gate.shape[-1]:
            return gate
        topk_indices = gate.topk(k=self.k, dim=-1).indices
        sparse_gate = torch.zeros_like(gate)
        sparse_gate.scatter_(1, topk_indices, gate.gather(1, topk_indices))
        sparse_gate = sparse_gate / sparse_gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return sparse_gate

    def forward(self, x):
        cls_tokens = x[:, :self.ncls]
        patch_tokens = x[:, self.ncls:]
        patch_tokens = self.ffn[0](patch_tokens)

        flat_cls_tokens = cls_tokens.reshape(-1, cls_tokens.shape[-1])
        expert_outputs = {}
        if self.moae and self.atom_in_layers is not None and self.atom_out_layers is not None:
            hidden_cache = {
                atom_name: F.gelu(self.atom_in_layers[atom_name](flat_cls_tokens))
                for atom_name in self.active_atom_inputs
            }
            for expert_idx, (atom_in, atom_out) in self.expert_pairs.items():
                expert_outputs[expert_idx] = self.atom_out_layers[atom_out](hidden_cache[atom_in]).reshape(
                    cls_tokens.shape[0],
                    self.ncls,
                    -1,
                )
        else:
            expert_outputs[0] = self.ffn[0](flat_cls_tokens).reshape(cls_tokens.shape[0], self.ncls, -1)

        mixed_cls_tokens = []
        for task_idx in range(self.ncls):
            token = cls_tokens[:, task_idx, :]
            expert_indices = self.task_expert_indices[task_idx]
            if len(expert_indices) == 1:
                mixed_cls_tokens.append(expert_outputs[expert_indices[0]][:, task_idx : task_idx + 1, :])
                continue
            gate = self.gate[task_idx](token).softmax(dim=-1)
            gate = self._sparsify_gate(gate)
            selected_outputs = torch.stack(
                [expert_outputs[expert_idx][:, task_idx, :] for expert_idx in expert_indices],
                dim=1,
            )
            mixed = (selected_outputs * gate.unsqueeze(-1)).sum(dim=1, keepdim=True)
            mixed_cls_tokens.append(mixed)

        cls_tokens = torch.cat(mixed_cls_tokens, dim=1)
        x = torch.cat((cls_tokens, patch_tokens), dim=1)
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dpr=0.0, ncls=6, moae=False, attention_backend="auto"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attention_backend=attention_backend)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp_moe(in_dim=dim, hidden_dim=int(dim * mlp_ratio), out_dim=dim, ncls=ncls, moae=moae)
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
                            weight=self.blocks[block_idx].mlp.ffn[0][0].weight.data,
                            bias=value,
                        )
                elif "blocks" in key and ".mlp.fc2." in key:
                    block_idx = int(key.split('.')[1])
                    if key.endswith(".weight"):
                        self.blocks[block_idx].mlp.load_pretrained_fc2(weight=value)
                    else:
                        self.blocks[block_idx].mlp.load_pretrained_fc2(
                            weight=self.blocks[block_idx].mlp.ffn[0][2].weight.data,
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
        attention_backend=attention_backend,
    )
