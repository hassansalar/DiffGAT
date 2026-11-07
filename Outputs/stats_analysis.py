"""
Statistical analyses + publication-ready figures for the CRISPR Off-Target paper.

Produces:
  - Bootstrap 95% CIs (table + bar chart)
  - Ablation study (table + grouped bar chart)
  - Label-permutation test (p-value + null histogram)
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - Training history curves
  - Paper metrics summary (.txt + .json)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Consistent style for paper figures
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------
def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Non-parametric bootstrap CIs for accuracy, precision, recall, F1, AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        pred = (yp > 0.5).astype(int)
        metrics["accuracy"].append(accuracy_score(yt, pred))
        metrics["precision"].append(precision_score(yt, pred, zero_division=0))
        metrics["recall"].append(recall_score(yt, pred, zero_division=0))
        metrics["f1"].append(f1_score(yt, pred, zero_division=0))
        try:
            metrics["auc"].append(roc_auc_score(yt, yp))
        except ValueError:
            metrics["auc"].append(0.5)

    alpha = (1 - ci) / 2
    result = {}
    for name, vals in metrics.items():
        arr = np.asarray(vals)
        result[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_low": float(np.percentile(arr, 100 * alpha)),
            "ci_high": float(np.percentile(arr, 100 * (1 - alpha))),
            "samples": arr,
        }
    return result


def plot_bootstrap_ci(
    boot: Dict[str, Dict[str, float]],
    save_path: str | Path,
    title: str = "Bootstrap 95% Confidence Intervals",
) -> None:
    names = list(boot.keys())
    means = [boot[n]["mean"] for n in names]
    lows = [boot[n]["ci_low"] for n in names]
    highs = [boot[n]["ci_high"] for n in names]
    yerr = [
        [m - l for m, l in zip(means, lows)],
        [h - m for m, h in zip(means, highs)],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=yerr, capsize=6, color="#2ecc71", edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    for i, (m, lo, hi) in enumerate(zip(means, lows, highs)):
        ax.text(i, min(hi + 0.02, 1.05), f"{m:.3f}\n[{lo:.3f}–{hi:.3f}]",
                ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Bootstrap CI chart → %s", save_path)


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    threshold: float = 0.5,
) -> np.ndarray:
    preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["Off-target (0)", "On-target (1)"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=classes, yticklabels=classes,
        ylabel="True label", xlabel="Predicted label",
        title="Confusion Matrix (Test Set)",
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Confusion matrix → %s", save_path)
    return cm


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------
def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5, label=f"GAT (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.2, label="Chance")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#e74c3c")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("ROC curve → %s", save_path)
    return float(auc)


# ---------------------------------------------------------------------------
# Precision-Recall Curve
# ---------------------------------------------------------------------------
def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(recall, precision, color="#3498db", lw=2.5, label=f"GAT (AP = {ap:.4f})")
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, color="gray", ls="--", lw=1.2, label=f"Baseline ({baseline:.2f})")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve (Test Set)", fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("PR curve → %s", save_path)
    return float(ap)


# ---------------------------------------------------------------------------
# Label Permutation (Randomization) Test
# ---------------------------------------------------------------------------
def permutation_test(
    trainer_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    observed_auc: float,
    n_perm: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train the same architecture on label-permuted data and build a null
    distribution of test AUC. p-value = fraction of null AUCs ≥ observed.
    """
    rng = np.random.default_rng(seed)
    null_aucs = []

    for i in range(n_perm):
        y_perm = y_train.copy()
        rng.shuffle(y_perm)
        model, trainer = trainer_factory()
        orig_epochs, orig_patience = trainer.config.epochs, trainer.config.patience
        trainer.config.epochs = min(30, orig_epochs)
        trainer.config.patience = 8
        trainer.fit(X_train, y_perm, X_val, y_val)
        res = trainer.test(X_test, y_test)
        null_aucs.append(res["auc"])
        trainer.config.epochs = orig_epochs
        trainer.config.patience = orig_patience
        if (i + 1) % 10 == 0 or (i + 1) == n_perm:
            logger.info("   Permutation %d/%d  null AUC=%.4f", i + 1, n_perm, res["auc"])

    null_aucs = np.asarray(null_aucs)
    p_value = float(np.mean(null_aucs >= observed_auc))
    # one-sided exact-style floor for reporting
    p_report = max(p_value, 1.0 / (n_perm + 1))
    return {
        "observed_auc": observed_auc,
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
        "p_value": p_value,
        "p_value_report": p_report,
        "n_perm": n_perm,
        "null_aucs": null_aucs,
    }


def plot_permutation(
    perm_result: Dict[str, Any],
    save_path: str | Path,
) -> None:
    null = perm_result["null_aucs"]
    obs = perm_result["observed_auc"]
    p = perm_result.get("p_value_report", perm_result["p_value"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null, bins=min(30, max(10, len(null) // 3)), color="#3498db",
            edgecolor="k", alpha=0.75, label="Null (permuted labels)")
    ax.axvline(obs, color="#e74c3c", lw=2.5, label=f"Observed AUC = {obs:.4f}")
    ax.set_xlabel("Test AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Label Permutation Test  (p ≤ {p:.4f})", fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Permutation chart → %s", save_path)


# ---------------------------------------------------------------------------
# Ablation Study
# ---------------------------------------------------------------------------
def run_ablation(
    components: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    trainer_factory,
    figure_dir: str | Path,
) -> Dict[str, Dict[str, float]]:
    """components: name → (X_train, X_val, X_test) already scaled."""
    results = {}
    for name, (Xtr, Xva, Xte) in components.items():
        logger.info("Ablation: %s  shape=%s", name, Xtr.shape)
        model, trainer = trainer_factory(in_dim=Xtr.shape[1])
        orig_e, orig_p = trainer.config.epochs, trainer.config.patience
        trainer.config.epochs = min(50, orig_e)
        trainer.config.patience = 12
        trainer.fit(Xtr, y_train, Xva, y_val)
        res = trainer.test(Xte, y_test)
        results[name] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "auc": res["auc"],
        }
        trainer.config.epochs = orig_e
        trainer.config.patience = orig_p
        logger.info("   %s → AUC=%.4f  F1=%.4f", name, res["auc"], res["f1"])

    plot_ablation(results, Path(figure_dir) / "ablation_study.png")
    return results


def plot_ablation(
    results: Dict[str, Dict[str, float]],
    save_path: str | Path,
) -> None:
    names = list(results.keys())
    metrics = ["auc", "f1", "accuracy", "precision", "recall"]
    x = np.arange(len(names))
    width = 0.15
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        ax.bar(x + i * width, vals, width, label=m.upper(),
               color=colors[i], edgecolor="k", alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study – Component Contribution", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=5)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Ablation chart → %s", save_path)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------
def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#3498db", lw=1.8)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#e74c3c", lw=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(history["val_auc"], label="Val AUC", color="#2ecc71", lw=1.8)
    axes[1].plot(history["val_f1"], label="Val F1", color="#f39c12", lw=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Training history → %s", save_path)


# ---------------------------------------------------------------------------
# Paper summary report
# ---------------------------------------------------------------------------
def write_paper_report(
    test_results: Dict[str, Any],
    boot: Dict[str, Dict[str, float]],
    ablation: Dict[str, Dict[str, float]],
    perm: Dict[str, Any],
    cm: np.ndarray,
    ap: float,
    figure_dir: str | Path,
    output_dir: str | Path,
) -> str:
    """Write a plain-text + JSON report suitable for the paper Methods/Results."""
    lines = []
    lines.append("=" * 72)
    lines.append("CRISPR Off-Target Platform – Paper Metrics Report")
    lines.append("=" * 72)

    lines.append("\n## 1. Test-set performance")
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        lines.append(f"  {k.capitalize():12s}: {test_results[k]:.4f}")
    lines.append(f"  {'AP (PR)':12s}: {ap:.4f}")

    lines.append("\n## 2. Bootstrap 95% Confidence Intervals")
    lines.append(f"  {'Metric':12s}  {'Mean':>8s}  {'95% CI':>18s}")
    for m, v in boot.items():
        lines.append(
            f"  {m:12s}  {v['mean']:8.4f}  [{v['ci_low']:.4f} – {v['ci_high']:.4f}]"
        )

    lines.append("\n## 3. Confusion Matrix (rows=true, cols=pred)")
    lines.append(f"                Pred 0      Pred 1")
    lines.append(f"  True 0     {cm[0, 0]:10,}  {cm[0, 1]:10,}")
    lines.append(f"  True 1     {cm[1, 0]:10,}  {cm[1, 1]:10,}")

    lines.append("\n## 4. Ablation Study (AUC / F1)")
    for name, r in sorted(ablation.items(), key=lambda x: -x[1]["auc"]):
        lines.append(f"  {name:25s}  AUC={r['auc']:.4f}  F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}")

    lines.append("\n## 5. Label Permutation (Randomization) Test")
    lines.append(f"  Observed AUC     : {perm['observed_auc']:.4f}")
    lines.append(f"  Null mean ± std  : {perm['null_mean']:.4f} ± {perm['null_std']:.4f}")
    lines.append(f"  p-value          : {perm['p_value']:.4f}")
    lines.append(f"  p-value (report) : ≤ {perm.get('p_value_report', perm['p_value']):.4f}")
    lines.append(f"  n permutations   : {perm.get('n_perm', len(perm['null_aucs']))}")

    lines.append("\n## 6. Figures")
    for name in (
        "training_history.png",
        "bootstrap_ci.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "pr_curve.png",
        "ablation_study.png",
        "permutation_test.png",
    ):
        lines.append(f"  - {figure_dir}/{name}")

    lines.append("\n" + "=" * 72)
    text = "\n".join(lines) + "\n"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / "paper_metrics_report.txt"
    txt_path.write_text(text, encoding="utf-8")

    # JSON (without large arrays)
    j = {
        "test_results": {k: float(test_results[k]) for k in
                         ("accuracy", "precision", "recall", "f1", "auc")},
        "average_precision": ap,
        "bootstrap_ci": {
            m: {k: float(v[k]) for k in ("mean", "std", "ci_low", "ci_high")}
            for m, v in boot.items()
        },
        "confusion_matrix": cm.tolist(),
        "ablation": ablation,
        "permutation": {
            "observed_auc": perm["observed_auc"],
            "null_mean": perm["null_mean"],
            "null_std": perm["null_std"],
            "p_value": perm["p_value"],
            "p_value_report": perm.get("p_value_report", perm["p_value"]),
            "n_perm": perm.get("n_perm", len(perm["null_aucs"])),
        },
    }
    json_path = out / "paper_metrics_report.json"
    json_path.write_text(json.dumps(j, indent=2), encoding="utf-8")

    logger.info("Paper report → %s  &  %s", txt_path, json_path)
    print(text)
    return text
