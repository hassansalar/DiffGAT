"""
Generate the 7-factor design checklist from on-target samples.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class ChecklistGenerator:
    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    def compute_ranges(self, X_on: np.ndarray) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}

        def get_range(name: str, p_low: float = 5, p_high: float = 95) -> Dict[str, float]:
            idx = self.feature_names.index(name)
            v = X_on[:, idx]
            return {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "p5": float(np.percentile(v, p_low)),
                "p95": float(np.percentile(v, p_high)),
            }

        results["gc"] = get_range("gc_total")
        results["delta_g"] = get_range("delta_g")
        results["cg_repeat"] = get_range("cg_repeat_count")
        results["max_cg_repeat"] = get_range("max_cg_repeat")
        results["seed_score"] = get_range("seed_score")
        return results

    def generate(self, X_on: np.ndarray, output_path: str | Path) -> str:
        r = self.compute_ranges(X_on)

        checklist = f"""
═══════════════════════════════════════════════════════════════════════════════
        CHECKLIST for gRNA Design – 7 Actionable Factors
═══════════════════════════════════════════════════════════════════════════════

【1. CG content】
  ✅ Safe range: {r['gc']['p5']:.3f} – {r['gc']['p95']:.3f} ({r['gc']['p5']*100:.1f}% – {r['gc']['p95']*100:.1f}%)
  📊 On-target mean: {r['gc']['mean']:.3f} ± {r['gc']['std']:.3f}

【2. Delta G binding energy (ΔG)】
  ✅ Safe range: {r['delta_g']['p5']:.3f} to {r['delta_g']['p95']:.3f} kcal/mol
  📊 On-target mean: {r['delta_g']['mean']:.3f} ± {r['delta_g']['std']:.3f}

【3. CG position】
  ✅ Focus on positions with high correlation from factor analysis
  ✅ Avoid CG-rich regions in the seed (positions 1-8)

【4. Mismatch position】
  ✅ Avoid mismatches in the seed region (positions 1-8)
  ✅ Mismatches in positions 15-20 are more tolerable

【5. PAM validation】
  ✅ SpCas9: PAM must be 5'-NGG-3'
  ❌ If no valid PAM, abort design

【6. CG repetition】
  ✅ Max count: ≤ {r['cg_repeat']['p95']:.3f}
  ✅ Max consecutive: ≤ {r['max_cg_repeat']['p95']:.3f}
  ❌ Avoid GGGG and CCCC runs

【7. Seed region score】
  ✅ Minimum score: ≥ {r['seed_score']['p5']:.3f}
  📊 On-target mean: {r['seed_score']['mean']:.3f} ± {r['seed_score']['std']:.3f}

═══════════════════════════════════════════════════════════════════════════════
📊 Source: Analysis of {len(X_on):,} on-target samples
🔬 Method: Diffusion Map + Factor Loading Analysis
═══════════════════════════════════════════════════════════════════════════════
"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(checklist)

        logger.info("Checklist saved: %s", output_path)
        print(checklist)
        return checklist
