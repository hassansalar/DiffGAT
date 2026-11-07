"""
kNN graph builders. Mini-batch graphs are constructed entirely on the GPU.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def build_batch_graph(X: torch.Tensor, k: int = 10) -> torch.Tensor:
    n = X.size(0)
    k = min(k, n - 1)
    if k < 1:
        return torch.zeros((2, 1), dtype=torch.long, device=X.device)

    dist = torch.cdist(X, X)
    dist.fill_diagonal_(float("inf"))
    nbr = dist.topk(k, dim=1, largest=False).indices

    src = torch.arange(n, device=X.device).unsqueeze(1).expand(n, k).reshape(-1)
    dst = nbr.reshape(-1)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def build_knn_graph(X: np.ndarray, k: int = 10) -> torch.Tensor:
    from sklearn.neighbors import NearestNeighbors

    logger.info("Building KNN graph (k=%d) on %s samples", k, f"{len(X):,}")
    _, indices = (
        NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        .fit(X)
        .kneighbors(X)
    )
    src = np.repeat(np.arange(len(X)), k)
    dst = indices[:, 1:].reshape(-1)
    edge_index = torch.from_numpy(
        np.stack(
            [np.concatenate([src, dst]), np.concatenate([dst, src])]
        ).astype(np.int64)
    )
    logger.info("   Edges: %s", f"{edge_index.shape[1]:,}")
    return edge_index
