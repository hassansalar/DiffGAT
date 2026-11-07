"""
Filtering, class balancing, index-based stratified splitting and scaling.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()

    def filter_sequences(
        self, sequences: List[str], labels: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Soft filter – only drop clearly invalid sequences.

        Previous low_GC threshold (<1%) was discarding the majority of the
        dataset because many records are mixed-case or contain PAM tails.
        We now only remove very short, N-dominated, or non-ACGT sequences.
        """
        n = len(sequences)
        # normalise to uppercase so GC / ACGT counts are reliable
        sequences = [s.upper() for s in sequences]
        lens = np.fromiter(map(len, sequences), dtype=np.float64, count=n)
        n_cnt = np.fromiter((s.count("N") for s in sequences), dtype=np.float64, count=n)
        acgt = np.fromiter(
            (sum(ch in "ACGT" for ch in s) for s in sequences),
            dtype=np.float64, count=n,
        )
        safe = np.maximum(lens, 1.0)
        short = lens < 15          # guide+target should be ~40–46 nt
        too_many_n = (n_cnt / safe) > 0.5
        low_acgt = (acgt / safe) < 0.6
        keep = ~short & ~too_many_n & ~low_acgt
        idx = np.flatnonzero(keep)
        logger.info(
            "Filter diagnostics: short=%d, too_many_N=%d, low_ACGT=%d → keep %s / %s",
            int(short.sum()), int(too_many_n.sum()), int(low_acgt.sum()),
            f"{len(idx):,}", f"{n:,}",
        )
        return [sequences[i] for i in idx], labels[idx]

    def balance_classes(
        self,
        sequences: List[str],
        labels: np.ndarray,
        target_size: int | None = None,
    ) -> Tuple[List[str], np.ndarray]:
        on_idx = np.flatnonzero(labels == 1)
        off_idx = np.flatnonzero(labels == 0)
        if target_size is None:
            target_size = min(len(on_idx), len(off_idx))
        target_size = min(target_size, len(on_idx), len(off_idx))

        rng = np.random.default_rng(self.random_state)
        sampled = np.concatenate(
            [
                rng.choice(on_idx, target_size, replace=False),
                rng.choice(off_idx, target_size, replace=False),
            ]
        )
        rng.shuffle(sampled)
        logger.info("Balanced to %s samples per class", f"{target_size:,}")
        return [sequences[i] for i in sampled], labels[sampled]

    def split_indices(
        self, y: np.ndarray, test_size: float = 0.15, val_size: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.arange(len(y))
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=test_size + val_size,
            stratify=y,
            random_state=self.random_state,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_size / (test_size + val_size),
            stratify=y[temp_idx],
            random_state=self.random_state,
        )
        logger.info(
            "Split → Train: %s | Val: %s | Test: %s",
            f"{len(train_idx):,}",
            f"{len(val_idx):,}",
            f"{len(test_idx):,}",
        )
        return train_idx, val_idx, test_idx

    def normalize(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        X_train_s = self.scaler.fit_transform(X_train)
        return (
            X_train_s,
            self.scaler.transform(X_val),
            self.scaler.transform(X_test),
            self.scaler,
        )
