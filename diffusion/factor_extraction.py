"""
Extract the seven actionable factors from Diffusion-Map loadings.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pearson_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    cov = A.T @ B
    denom = np.outer(
        np.sqrt((A ** 2).sum(axis=0)), np.sqrt((B ** 2).sum(axis=0))
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / denom
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


class FactorExtractor:
    ACTIONABLE_FACTORS = [
        "CG content",
        "Delta G binding energy",
        "CG position",
        "Mismatch position",
        "PAM validation",
        "CG repetition",
        "Seed region score",
    ]

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = list(feature_names)

    def compute_loadings(
        self, X_features: np.ndarray, X_diffusion: np.ndarray
    ) -> pd.DataFrame:
        corr = pearson_matrix(X_features, X_diffusion)
        return pd.DataFrame(
            corr,
            index=self.feature_names,
            columns=[f"DC{j+1}" for j in range(X_diffusion.shape[1])],
        )

    def extract_factors(self, df: pd.DataFrame) -> Dict[str, float]:
        results: Dict[str, float] = {}
        idx = set(df.index)

        def max_abs(cols: List[str]) -> float | None:
            cols = [c for c in cols if c in idx]
            return float(df.loc[cols].abs().max().max()) if cols else None

        groups = {
            "CG content": ["gc_total"] + [f"gc_pos_{i+1}" for i in range(20)],
            "Delta G binding energy": ["delta_g"],
            "CG position": [f"cg_pos_{i+1}" for i in range(20)],
            "Mismatch position": [f"mismatch_pos_{i+1}" for i in range(20)],
            "PAM validation": ["pam_valid"],
            "CG repetition": ["cg_repeat_count", "max_cg_repeat"],
            "Seed region score": ["seed_score"],
        }
        for name, cols in groups.items():
            val = max_abs(cols)
            if val is not None:
                results[name] = val
        return results

    def print_report(self, results: Dict[str, float]) -> Dict[str, float]:
        print("\n" + "=" * 70)
        print("7 Actionable Factors Extracted")
        print("=" * 70)
        for i, (factor, score) in enumerate(results.items(), 1):
            print(f"  {i}. {factor:35s} | Loading: {score:.4f}")
        print("=" * 70)
        return results
