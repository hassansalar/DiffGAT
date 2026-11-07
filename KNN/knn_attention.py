"""
KNN-Attention module (optional middle layer; not used by the main pipeline).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class KNNAttentionModule(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, temperature: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.temperature = temperature
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, neighbor_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query(x).unsqueeze(1)
        k = self.key(neighbor_feats)
        v = self.value(neighbor_feats)
        attn = F.softmax(
            torch.bmm(q, k.transpose(1, 2)) / self.temperature, dim=-1
        )
        out = self.out(torch.bmm(attn, v).squeeze(1))
        return out, attn.squeeze(1)


class KNNEmbedder:
    def __init__(self, n_neighbors: int = 16) -> None:
        self.n_neighbors = n_neighbors
        self.index: Optional[NearestNeighbors] = None
        self.reference_embeddings: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> "KNNEmbedder":
        logger.info("Building KNN index with %s samples", f"{embeddings.shape[0]:,}")
        self.index = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric="euclidean"
        )
        self.index.fit(embeddings)
        self.reference_embeddings = embeddings
        return self

    def get_neighbors(
        self, embeddings: np.ndarray, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.index is not None
        return self.index.kneighbors(embeddings, n_neighbors=k or self.n_neighbors)

    def aggregate(
        self,
        embeddings: np.ndarray,
        k: Optional[int] = None,
        chunk: int = 2048,
    ) -> np.ndarray:
        assert self.index is not None and self.reference_embeddings is not None
        distances, indices = self.index.kneighbors(
            embeddings, n_neighbors=k or self.n_neighbors
        )
        w = 1.0 / (distances + 1e-8)
        w /= w.sum(axis=1, keepdims=True)
        out = np.empty_like(embeddings)
        for s in range(0, len(embeddings), chunk):
            nb = self.reference_embeddings[indices[s : s + chunk]]
            out[s : s + chunk] = np.einsum("ck,ckd->cd", w[s : s + chunk], nb)
        return out
