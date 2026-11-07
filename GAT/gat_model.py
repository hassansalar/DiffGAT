"""
Graph Attention Network (mini-batch version) with residual connections,
LayerNorm, and explicit attention return for design-time analysis.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, n_heads: int = 8, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x.size(0)
        h = self.W(x).view(N, self.n_heads, self.head_dim)
        src, dst = edge_index[0], edge_index[1]

        e_src = (h * self.a_src).sum(dim=-1)
        e_dst = (h * self.a_dst).sum(dim=-1)
        e = self.leaky_relu(e_src[src] + e_dst[dst])

        attn = self.dropout(self._segment_softmax(e, dst, N))

        out = h.new_zeros(N, self.n_heads, self.head_dim)
        out.index_add_(0, dst, h[src] * attn.unsqueeze(-1))
        return out.view(N, self.out_dim), attn

    @staticmethod
    def _segment_softmax(
        e: torch.Tensor, index: torch.Tensor, N: int
    ) -> torch.Tensor:
        idx = index.unsqueeze(1).expand_as(e)
        max_e = e.new_zeros(N, e.size(1)).scatter_reduce(
            0, idx, e, reduce="amax", include_self=False
        )
        exp_e = torch.exp(e - max_e[index])
        denom = e.new_zeros(N, e.size(1)).index_add_(0, index, exp_e)
        return exp_e / (denom[index] + 1e-8)


class GATModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.layers = nn.ModuleList(
            GATLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        h = self.input_proj(x)
        all_attn: Optional[List[torch.Tensor]] = [] if return_attn else None
        for layer, norm in zip(self.layers, self.norms):
            h_new, attn = layer(h, edge_index)
            h = norm(h + self.dropout(F.gelu(h_new)))   # residual + GELU
            if return_attn and all_attn is not None:
                all_attn.append(attn)
        logits = self.classifier(h).squeeze(-1)
        return logits, all_attn
