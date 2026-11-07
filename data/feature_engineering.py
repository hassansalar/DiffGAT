"""
Vectorised extraction of the classic 69 CRISPR features + a few high-signal extras.
gc_pos_* and cg_pos_* are identical (kept for interface compatibility).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_W = 23
_PAD_G, _PAD_T = "\x00", "\x01"

_ENERGY = {
    "GC": -3.0, "CG": -2.5, "GG": -1.5, "CC": -1.5,
    "AT": -0.5, "TA": -0.5, "GT": -1.0, "TG": -1.0,
    "AC": -0.8, "CA": -0.8, "GA": -0.7, "AG": -0.7,
    "TC": -0.6, "CT": -0.6,
}


def _build_energy_table() -> np.ndarray:
    tbl = np.full((256, 256), -0.5, dtype=np.float32)
    idx = np.arange(256)
    tbl[idx, idx] = 0.0
    for pair, e in _ENERGY.items():
        tbl[ord(pair[0]), ord(pair[1])] = e
    return tbl


_ENERGY_TABLE = _build_energy_table()
_SEED_W = np.array([1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)

FEATURE_NAMES: tuple[str, ...] = tuple(
    ["gc_total"]
    + [f"gc_pos_{i+1}" for i in range(20)]
    + ["delta_g"]
    + [f"cg_pos_{i+1}" for i in range(20)]
    + [f"mismatch_pos_{i+1}" for i in range(20)]
    + [
        "pam_valid",
        "cg_repeat_count",
        "max_cg_repeat",
        "seed_score",
        "mismatch_count",
        "cas9_energy",
        "chromatin_access",
        # extra high-signal features
        "seed_mismatch_count",
        "distal_mismatch_count",
        "gc_seed",
        "purine_content",
    ]
)
N_FEATURES = len(FEATURE_NAMES)

FEATURE_NAMES_NON_REDUNDANT: tuple[str, ...] = tuple(
    n for n in FEATURE_NAMES if not n.startswith("cg_pos_")
)


def split_guide_target(seq: str) -> tuple[str, str]:
    if len(seq) < 46:
        return seq[:20], seq[20:40]
    return seq[:23], seq[23:46]


def _to_matrix(strings: Sequence[str], pad: str) -> np.ndarray:
    joined = "".join(s[:_W].ljust(_W, pad) for s in strings)
    return np.frombuffer(
        joined.encode("latin-1", errors="replace"), dtype=np.uint8
    ).reshape(len(strings), _W)


def _extract_chunk(seqs: Sequence[str]) -> np.ndarray:
    n = len(seqs)
    guides, targets = zip(*(split_guide_target(s) for s in seqs))
    G = _to_matrix(guides, _PAD_G)
    T = _to_matrix(targets, _PAD_T)
    G20, T20 = G[:, :20], T[:, :20]

    valid = (G20 != 0) & (T20 != 1)
    is_g = G == ord("G")
    is_c = G == ord("C")
    is_a = G == ord("A")
    is_t = G == ord("T")
    gc = is_g | is_c
    gc20 = gc[:, :20].astype(np.float32)
    mism = (valid & (G20 != T20)).astype(np.float32)

    lens = np.fromiter(map(len, seqs), dtype=np.float64, count=n)
    gc_n = np.fromiter(
        (s.count("G") + s.count("C") for s in seqs), dtype=np.float64, count=n
    )
    gc_total = np.divide(gc_n, lens, out=np.zeros(n), where=lens > 0)

    score = np.where(valid, _ENERGY_TABLE[G20, T20], 0.0).sum(axis=1)
    delta_g = -5.0 - 0.3 * score

    pam = np.fromiter(
        (
            1.0 if len(t) >= 3 and t[-2:] == "GG" and t[-3] in "ACGT" else 0.0
            for t in targets
        ),
        dtype=np.float32,
        count=n,
    )

    pair = (is_c[:, :-1] & is_g[:, 1:]) | (is_g[:, :-1] & is_c[:, 1:])
    taken_prev = np.zeros(n, dtype=bool)
    count = np.zeros(n, dtype=np.int32)
    for i in range(pair.shape[1]):
        taken = pair[:, i] & ~taken_prev
        count += taken
        taken_prev = taken
    cg_repeat_count = count / 10.0

    curr = np.zeros(n, dtype=np.int32)
    longest = np.zeros(n, dtype=np.int32)
    for i in range(20):
        curr = np.where(gc[:, i], curr + 1, 0)
        longest = np.maximum(longest, curr)
    max_cg_repeat = longest / 5.0

    v8 = valid[:, :8]
    eq8 = G[:, :8] == T[:, :8]
    w = _SEED_W
    seed_num = np.where(v8, np.where(eq8, w, -0.7 * w), 0.0).sum(axis=1)
    seed_den = (v8 * w).sum(axis=1)
    seed_score = np.maximum(0.0, seed_num / np.maximum(seed_den, 1e-9))

    mismatch_count = mism.sum(axis=1) / 20.0
    cas9_energy = -8.0 - 6.0 * (gc20.sum(axis=1) / 20.0)
    chromatin = 0.3 + 0.5 * gc_total

    # Extra high-signal features
    seed_mism = mism[:, :8].sum(axis=1) / 8.0
    distal_mism = mism[:, 12:].sum(axis=1) / max(8.0, 1.0)
    gc_seed = gc20[:, :8].mean(axis=1)
    purine = ((is_a | is_g)[:, :20].sum(axis=1) / 20.0).astype(np.float32)

    out = np.concatenate(
        [
            gc_total[:, None],
            gc20,
            delta_g[:, None],
            gc20,          # cg_pos_* (duplicate of gc_pos_*)
            mism,
            pam[:, None],
            cg_repeat_count[:, None],
            max_cg_repeat[:, None],
            seed_score[:, None],
            mismatch_count[:, None],
            cas9_energy[:, None],
            chromatin[:, None],
            seed_mism[:, None],
            distal_mism[:, None],
            gc_seed[:, None],
            purine[:, None],
        ],
        axis=1,
    )
    return out.astype(np.float32)


def extract_features(
    sequences: Sequence[str], chunk_size: int = 100_000
) -> np.ndarray:
    """Return an (n, N_FEATURES) float32 feature matrix."""
    sequences = list(sequences)
    out = np.empty((len(sequences), N_FEATURES), dtype=np.float32)
    for s in range(0, len(sequences), chunk_size):
        out[s : s + chunk_size] = _extract_chunk(sequences[s : s + chunk_size])
    return out
