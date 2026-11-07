"""
Torch-free k-mer frequency embedding used as DNABERT fallback.
Default k=5 → 1024-d (much richer than 3-mer / 64-d or 4-mer / 256-d).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_LUT = np.full(256, -1, dtype=np.int16)
for _i, _ch in enumerate("ACGT"):
    _LUT[ord(_ch)] = _i


def kmer_counts(sequences: Sequence[str], k: int = 5) -> np.ndarray:
    sequences = list(sequences)
    n = len(sequences)
    n_kmers = 4 ** k
    if n == 0:
        return np.zeros((0, n_kmers), dtype=np.float32)

    L = max(len(s) for s in sequences)
    if L < k:
        return np.zeros((n, n_kmers), dtype=np.float32)

    raw = "".join(s.ljust(L, "\x00") for s in sequences).encode(
        "latin-1", errors="replace"
    )
    codes = _LUT[np.frombuffer(raw, dtype=np.uint8).reshape(n, L)].astype(np.int64)

    n_win = L - k + 1
    idx = np.zeros((n, n_win), dtype=np.int64)
    ok = np.ones((n, n_win), dtype=bool)
    for j in range(k):
        c = codes[:, j : j + n_win]
        ok &= c >= 0
        idx = idx * 4 + np.where(c >= 0, c, 0)

    rows = np.broadcast_to(np.arange(n)[:, None], idx.shape)
    flat = (rows * n_kmers + idx)[ok]
    counts = np.bincount(flat, minlength=n * n_kmers).reshape(n, n_kmers).astype(
        np.float32
    )
    totals = counts.sum(axis=1, keepdims=True)
    np.divide(counts, totals, out=counts, where=totals > 0)
    return counts
