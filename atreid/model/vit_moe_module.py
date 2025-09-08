import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import numpy as np

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mids):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 3 B H N c
        q, k, v = qkv.unbind(0)  # B H N c
        attn = ((q * self.scale) @ k.transpose(-2, -1))  # B H N N
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlpmoe(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k=1):
        super().__init__()
        self.k = k
        self.ncls = 6
        self.nexp = 26# vm im cm sc cc
        gt0 = [[0, 1, 0, 0, 1, 0],  # vmsc
               [0, 1, 0, 0, 0, 1],  # vmcc
               [0, 0, 1, 0, 1, 0],  # imsc
               [0, 0, 1, 0, 0, 1],  # imcc
               [0, 0, 0, 1, 1, 0],  # cmsc
               [0, 0, 0, 1, 0, 1]]  # cmcc
        self.gt = [[0] + [(j != k) * gt0[i][j] * gt0[i][k] for j in range(1, 6)
                          for k in range(1, 6)] for i in range(6)]
        self.gate = nn.ModuleList([
            nn.Linear(in_dim, sum(self.gt[i]), bias=False) for i in range(self.ncls)])
        hidden_dim2 = hidden_dim
        vm1 = nn.Linear(in_dim, hidden_dim2)
        im1 = nn.Linear(in_dim, hidden_dim2)
        cm1 = nn.Linear(in_dim, hidden_dim2)
        sc1 = nn.Linear(in_dim, hidden_dim2)
        cc1 = nn.Linear(in_dim, hidden_dim2)
        vm2 = nn.Linear(hidden_dim2, in_dim)
        im2 = nn.Linear(hidden_dim2, in_dim)
        cm2 = nn.Linear(hidden_dim2, in_dim)
        sc2 = nn.Linear(hidden_dim2, in_dim)
        cc2 = nn.Linear(hidden_dim2, in_dim)
        atom1 = [vm1, im1, cm1, sc1, cc1]
        atom2 = [vm2, im2, cm2, sc2, cc2]
        self.moe = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim), ), ])
        for i in range(5):
            for j in range(5):
                self.moe.append(nn.Sequential(atom1[i], nn.GELU(), atom2[j]))

    def forward(self, x, mids):
        cls_tokens, patch_tokens = x[:, :self.ncls], x[:, self.ncls:]
        patch_tokens = self.moe[0](patch_tokens)
        x = []
        tokens = list(cls_tokens.chunk(self.ncls, 1))
        for i in range(len(tokens)):
            token = tokens[i]
            gt = self.gt[i]
            out = torch.cat([self.moe[j](token).unsqueeze(3)
                             for j in range(len(self.moe)) if gt[j] == 1], 3)
            gate = self.gate[i](token).unsqueeze(2)
            gate = gate.softmax(dim=-1)
            if self.k < sum(gt):
                gate_topk = gate.topk(k=self.k + 1, dim=-1)[0][:, :, :, -1]
                gate_topk = ((gate - gate_topk.unsqueeze(3)) > 0).float()
                gate = gate * gate_topk
                gate = gate / (gate.sum(dim=-1, keepdim=True).clamp(1e-6))
            x.append((out * gate).sum(3))
        x = torch.cat(x, 1)
        x = torch.cat((x, patch_tokens), 1)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dpr=0., layer=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlpmoe(in_dim=dim, hidden_dim=int(dim * mlp_ratio), out_dim=dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.layer = layer

    def forward(self, x, mids):
        x = x + self.drop_path(self.attn(self.norm1(x), mids))
        x = x + self.drop_path(self.mlp(self.norm2(x), mids))
        return x


class PatchEmbed_overlap(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(
            stride_size, self.num_y, self.num_x))
        self.num_patches = self.num_x * self.num_y
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., drop_path_rate=0., ncls=6):
        super().__init__()
        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size,
                                              stride_size=stride_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, ncls, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + ncls, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dpr=dpr[i], layer=i) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ncls = ncls

        self.apply(self._init_weights)
        self.fix_init_weight()
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if layer_id not in self.layers:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
            else:
                nexp = layer.mlp.nexp
                for i in range(nexp):
                    rescale(layer.mlp.moe[i][2].weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mids):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed

        for blk in self.blocks:
            x = blk(x, mids)
        x = self.norm(x)
        return x[:, :self.ncls]

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k:  # 分类头
                continue
            if k == 'pos_embed' and v.shape != self.pos_embed.shape:
                v = interpolate_pos_embed(self, v)
            if k == 'cls_token':
                v = v.expand(-1, self.ncls, -1)
            try:
                if "blocks" in k:
                    nexp = layer.mlp.nexp
                    if 'fc1' in k:
                        for i in range(0, nexp):
                            self.state_dict()[k.replace('fc1', f'moe.{i}.0')].copy_(v)
                    elif 'fc2' in k:
                        for i in range(0, nexp):
                            self.state_dict()[k.replace('fc2', f'moe.{i}.2')].copy_(v)
                    else:
                        self.state_dict()[k].copy_(v)
                else:
                    self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(
                    k, v.shape, self.state_dict()[k].shape))


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


def vit_base_patch16_224_ReID_moe(img_size=(256, 128), stride_size=16, drop_path_rate=0.2, ncls=6):
    model = ViT(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4, drop_path_rate=drop_path_rate, ncls=ncls)
    return model

