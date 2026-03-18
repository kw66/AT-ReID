from __future__ import annotations

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
TASK_ROUTE_NAMES = {
    task_name: (f"{atom_left}->{atom_right}", f"{atom_right}->{atom_left}")
    for task_name, (atom_left, atom_right) in TASK_TO_ATOMS.items()
}
MOAE_BASE_BALANCE_EMA = 0.9
MOAE_BASE_BALANCE_STEP = 0.01
MOAE_BASE_BALANCE_CLAMP = 0.5
MOAE_BALANCE_UPDATE = "direct-rms"
MOAE_BALANCE_STEP_DECAY = "linear"
MOAE_BALANCE_FREEZE_EPOCH = 40


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
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        ncls=6,
        moae=False,
    ):
        super().__init__()
        self.ncls = ncls
        self.moae = bool(moae)
        self.moae_balance_ema = MOAE_BASE_BALANCE_EMA
        self.moae_balance_step = MOAE_BASE_BALANCE_STEP
        self.moae_balance_clamp = MOAE_BASE_BALANCE_CLAMP
        self.moae_balance_freeze_epoch = MOAE_BALANCE_FREEZE_EPOCH
        self.moae_balance_update = MOAE_BALANCE_UPDATE
        self.moae_balance_step_decay = MOAE_BALANCE_STEP_DECAY

        self.patch_fc1 = nn.Linear(in_dim, hidden_dim)
        self.patch_act = nn.GELU()
        self.patch_fc2 = nn.Linear(hidden_dim, out_dim)

        if self.moae:
            self.num_atoms = len(ATOM_NAMES)
            src_indices = []
            dst_indices = []
            for task_name in TASK_NAMES[: self.ncls]:
                left_atom, right_atom = TASK_TO_ATOMS[task_name]
                src_indices.append((ATOM_TO_INDEX[left_atom], ATOM_TO_INDEX[right_atom]))
                dst_indices.append((ATOM_TO_INDEX[right_atom], ATOM_TO_INDEX[left_atom]))
            self.register_buffer(
                "task_pair_src_indices",
                torch.tensor(src_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "task_pair_dst_indices",
                torch.tensor(dst_indices, dtype=torch.long),
                persistent=False,
            )
            self.gate_pair = nn.Parameter(torch.empty(self.ncls, 2, in_dim))
            trunc_normal_(self.gate_pair, std=0.02)
            self.atom_in_layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(self.num_atoms)])
            self.atom_out_layers = nn.ModuleList([nn.Linear(hidden_dim, in_dim) for _ in range(self.num_atoms)])
            self.register_buffer("balance_bias", torch.zeros(self.ncls, 2, dtype=torch.float32), persistent=False)
            self.register_buffer("balance_usage_ema", torch.full((self.ncls, 2), 0.5, dtype=torch.float32), persistent=False)
        else:
            self.num_atoms = len(ATOM_NAMES)
            self.gate_pair = None
            self.atom_in_layers = None
            self.atom_out_layers = None
            self.register_buffer("task_pair_src_indices", torch.zeros((0, 2), dtype=torch.long), persistent=False)
            self.register_buffer("task_pair_dst_indices", torch.zeros((0, 2), dtype=torch.long), persistent=False)
            self.register_buffer("balance_bias", torch.zeros((0, 2), dtype=torch.float32), persistent=False)
            self.register_buffer("balance_usage_ema", torch.zeros((0, 2), dtype=torch.float32), persistent=False)

        self.drop = nn.Dropout(0.03)
        self.last_route_stats = None
        self.last_balance_fractions = None
        self.moae_epoch = 0

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

    def _prepare_gate_inputs(self, cls_tokens: torch.Tensor):
        return F.normalize(cls_tokens.float(), dim=-1)

    def _compute_pair_gate_logits(self, gate_inputs: torch.Tensor) -> torch.Tensor:
        gate_weights = F.normalize(self.gate_pair.float(), dim=-1)
        gate_logits = torch.einsum("bnd,nrd->bnr", gate_inputs, gate_weights)
        if self.balance_bias.numel() > 0:
            gate_logits = gate_logits + self.balance_bias.float().unsqueeze(0)
        return gate_logits

    def _compute_route_probabilities(self, cls_tokens: torch.Tensor):
        gate_inputs = self._prepare_gate_inputs(cls_tokens)
        gate_logits = self._compute_pair_gate_logits(gate_inputs)
        route_probs = gate_logits.softmax(dim=-1)
        chosen_route = route_probs.argmax(dim=-1)
        return route_probs, chosen_route

    @staticmethod
    def _stack_linear_params(layers, dtype, device):
        weights = torch.stack([layer.weight for layer in layers], dim=0).to(dtype=dtype, device=device)
        if layers[0].bias is None:
            return weights, None
        biases = torch.stack([layer.bias for layer in layers], dim=0).to(dtype=dtype, device=device)
        return weights, biases

    def _compute_route_candidates(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        atom_in_weight, atom_in_bias = self._stack_linear_params(self.atom_in_layers, cls_tokens.dtype, cls_tokens.device)
        atom_out_weight, atom_out_bias = self._stack_linear_params(self.atom_out_layers, cls_tokens.dtype, cls_tokens.device)

        source_hidden_all = torch.einsum("bnd,ahd->bnah", cls_tokens, atom_in_weight)
        if atom_in_bias is not None:
            source_hidden_all = source_hidden_all + atom_in_bias.view(1, 1, self.num_atoms, -1)
        source_hidden_all = F.gelu(source_hidden_all)

        batch_size = cls_tokens.shape[0]
        hidden_dim = source_hidden_all.shape[-1]
        source_indices = self.task_pair_src_indices.view(1, self.ncls, 2, 1).expand(batch_size, -1, -1, hidden_dim)
        source_hidden_pair = source_hidden_all.gather(2, source_indices)

        dst_indices = self.task_pair_dst_indices.reshape(-1)
        route_out_weight = atom_out_weight.index_select(0, dst_indices).view(self.ncls, 2, cls_tokens.shape[-1], hidden_dim)
        route_candidates = torch.einsum("bnrh,nrdh->bnrd", source_hidden_pair, route_out_weight)
        if atom_out_bias is not None:
            route_out_bias = atom_out_bias.index_select(0, dst_indices).view(self.ncls, 2, cls_tokens.shape[-1])
            route_candidates = route_candidates + route_out_bias.unsqueeze(0)
        return route_candidates

    def _record_route_stats(self, route_probs: torch.Tensor, chosen_route: torch.Tensor, chosen_weights: torch.Tensor) -> None:
        if not self.moae:
            self.last_route_stats = None
            self.last_balance_fractions = None
            return
        route_probs = route_probs.detach().float()
        left_prob = route_probs[:, :, 0]
        right_prob = route_probs[:, :, 1]
        left_select = chosen_route.detach().eq(0).float()
        self.last_route_stats = {
            "token_counts": torch.full(
                (self.ncls,),
                float(chosen_route.shape[0]),
                device=chosen_route.device,
                dtype=torch.float32,
            ),
            "left_select_counts": left_select.sum(dim=0),
            "left_prob_sums": left_prob.sum(dim=0),
            "right_prob_sums": right_prob.sum(dim=0),
            "chosen_weight_sums": chosen_weights.detach().float().sum(dim=0),
        }
        route_onehot = F.one_hot(chosen_route.detach(), num_classes=2).float()
        self.last_balance_fractions = route_onehot.mean(dim=0)

    def consume_route_stats(self):
        stats = self.last_route_stats
        self.last_route_stats = None
        return stats

    def _current_balance_step(self) -> float:
        if self.moae_balance_step <= 0:
            return 0.0
        epoch = max(int(self.moae_epoch), 1)
        if epoch > self.moae_balance_freeze_epoch:
            return 0.0
        progress = (epoch - 1) / max(self.moae_balance_freeze_epoch - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        scale = 1.0 - progress
        return float(self.moae_balance_step) * max(scale, 0.0)

    def update_balance_state(self):
        if (
            not self.moae
            or self.last_balance_fractions is None
        ):
            return
        effective_step = self._current_balance_step()
        if effective_step <= 0 or self.balance_bias.numel() == 0:
            self.last_balance_fractions = None
            return
        with torch.no_grad():
            fractions = self.last_balance_fractions.to(device=self.balance_usage_ema.device, dtype=self.balance_usage_ema.dtype)
            self.balance_usage_ema.mul_(self.moae_balance_ema).add_(fractions * (1.0 - self.moae_balance_ema))
            target = self.balance_usage_ema.new_full(self.balance_usage_ema.shape, 0.5)
            delta = target - fractions
            denom = delta.square().mean().sqrt().clamp_min(1e-6)
            delta = delta / denom
            self.balance_bias.add_(delta * effective_step)
            self.balance_bias.sub_(self.balance_bias.mean(dim=-1, keepdim=True))
            self.balance_bias.clamp_(-self.moae_balance_clamp, self.moae_balance_clamp)
        self.last_balance_fractions = None

    def _forward_weighted_top1_moae(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        route_probs, chosen_route = self._compute_route_probabilities(cls_tokens)
        chosen_weights = route_probs.gather(2, chosen_route.unsqueeze(-1)).squeeze(-1)
        self._record_route_stats(route_probs, chosen_route, chosen_weights)
        chosen_weights = chosen_weights.to(dtype=cls_tokens.dtype)
        route_candidates = self._compute_route_candidates(cls_tokens)
        chosen_outputs = route_candidates.gather(
            2,
            chosen_route.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, cls_tokens.shape[-1]),
        ).squeeze(2)
        return chosen_outputs * chosen_weights.unsqueeze(-1)

    def forward(self, x):
        cls_tokens = x[:, :self.ncls]
        patch_tokens = self._apply_patch_mlp(x[:, self.ncls:])

        if not self.moae or self.atom_in_layers is None or self.atom_out_layers is None:
            self.last_route_stats = None
            self.last_balance_fractions = None
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
        self.moae = bool(moae)
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

    def consume_moae_route_stats(self):
        if not self.moae:
            return None
        merged = None
        for block in self.blocks:
            block_stats = block.mlp.consume_route_stats()
            if block_stats is None:
                continue
            if merged is None:
                merged = {
                    key: value.detach().clone()
                    for key, value in block_stats.items()
                }
            else:
                for key, value in block_stats.items():
                    merged[key] = merged[key] + value.detach()
        return merged

    def set_moae_epoch(self, epoch: int):
        for block in self.blocks:
            if hasattr(block.mlp, "moae_epoch"):
                block.mlp.moae_epoch = int(epoch)

    def update_moae_balance(self):
        for block in self.blocks:
            if hasattr(block.mlp, "update_balance_state"):
                block.mlp.update_balance_state()

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
