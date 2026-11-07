"""
Data loading and a thin torch Dataset.
Feature extraction is performed once, vectorised, after filtering/balancing.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.feature_engineering import FEATURE_NAMES, extract_features

logger = logging.getLogger(__name__)


def load_sequences(data_dir: str) -> tuple[List[str], np.ndarray]:
    files = sorted(
        f
        for f in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)
        if os.path.isfile(f)
    )
    sequences: List[str] = []
    labels: List[str] = []
    for path in tqdm(files, desc="Loading files"):
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 5:
                    sequences.append(parts[4])
                    labels.append(parts[-1])
    return sequences, np.asarray(labels, dtype=np.float32)


class CRISPRDataset(Dataset):
    def __init__(
        self,
        sequences: List[str],
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> None:
        self.raw_sequences = list(sequences)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.features = (
            extract_features(self.raw_sequences) if features is None else features
        )

    @staticmethod
    def get_feature_names() -> List[str]:
        return list(FEATURE_NAMES)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "label": torch.tensor(self.labels[idx]),
            "sequence": self.raw_sequences[idx],
        }
