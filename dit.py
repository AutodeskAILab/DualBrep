import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from timm.models.vision_transformer import Attention

import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from itertools import repeat
from collections.abc import Iterable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from timm.models.layers import DropPath


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)

        self.mlp = MLP(width=width)
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)

    def forward(self, x, y, t, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        t = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(t, t, t, need_weights=False)[0].reshape(B, N, C)
        x = x + self.cross_attn(x, y, y, need_weights=False)[0]
        x = x + gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class DiTBlock_xtrans(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        from x_transformers import Attention
        self.attn = Attention(dim=width, flash=True, heads=heads, qk_norm=True, dropout=0.0)
        self.cross_attn = Attention(dim=width, flash=True, heads=heads, qk_norm=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)

        self.mlp = MLP(width=width)
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)

    def forward(self, x, y, t, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        t = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(t)
        x = x + self.cross_attn(x, context=y)
        x = x + gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PlainDiT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_size,
                 depth,
                 heads,
                 xtrans=False
                 ):
        super().__init__()

        # x embedding
        self.x_embed = nn.Linear(in_channels, hidden_size, bias=True)
        DiT_class = DiTBlock_xtrans if xtrans else DiTBlock
        self.blocks = nn.ModuleList([
            DiT_class(
                width=hidden_size,
                heads=heads,
            )
            for i in range(depth)
        ])
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_channels)
        )

    def forward(self, x, cond, t):
        x = self.x_embed(x)
        for block in self.blocks:
            x = block(x, cond, t)
        x = self.final_layer(x)
        return x