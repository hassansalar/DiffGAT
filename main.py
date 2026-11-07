"""
Main pipeline – CRISPR Off-Target Prediction Platform (v4 – high performance)
Includes: Ablation study, Label-Permutation test, Bootstrap CIs + charts.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from data.crispr_dataset import load_sequences
from data.feature_engineering import FEATURE_NAMES, extract_features
from data.preprocessing import Preprocessor
from diffusion.diffusion_map import DiffusionMap
from diffusion.factor_extraction import FactorExtractor
from dnabert.encoder import DNABertEncoder
from gat.gat_model import GATModel
from outputs.checklist_generator import ChecklistGenerator
from outputs.stats_analysis import (
    bootstrap_metrics,
    plot_bootstrap_ci,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    permutation_test,
    plot_permutation,
    run_ablation,
    plot_training_history,
    write_paper_report,
)
from trainer.trainer import GATTrainer
from utils.logger import setup_logger
from utils.seed import set_all_seeds

config = get_config()
logger = setup_logger()
set_all_seeds(config.seed)
torch.set_float32_matmul_precision("high")

OUTPUT_DIR = config.OUTPUT_DIR
MODEL_DIR = config.MODEL_DIR
CHECKLIST_DIR = config.CHECKLIST_DIR
CACHE_DIR = config.CACHE_DIR
FIGURE_DIR = config.FIGURE_DIR
for d in (OUTPUT_DIR, MODEL_DIR, CHECKLIST_DIR, CACHE_DIR, FIGURE_DIR):
    os.makedirs(d, exist_ok=True)


def _drop_constant_cols(X_tr, X_va, X_te, tol=1e-8):
    """Remove near-constant columns (e.g. PAM when always 1)."""
    std = X_tr.std(axis=0)
    keep = std > tol
    n_drop = int((~keep).sum())
    if n_drop:
        logger.info("Dropping %d constant feature column(s) before GAT", n_drop)
    return X_tr[:, keep], X_va[:, keep], X_te[:, keep], keep


def _make_trainer_factory(default_in_dim: int | None = None):
    """Return a callable that builds a fresh model + trainer.

    Accepts either a positional int or keyword ``in_dim=`` so it works
    both for the main run and for the ablation helper.
    """
    def factory(in_dim: int | None = None, **_kwargs):
        dim = in_dim if in_dim is not None else default_in_dim
        if dim is None:
            raise ValueError("in_dim must be provided")
        model = GATModel(
            in_dim=dim,
            hidden_dim=config.gat_hidden_dim,
            n_heads=config.gat_n_heads,
            n_layers=config.gat_n_layers,
            dropout=config.gat_dropout,
        ).to(config.device)
        trainer = GATTrainer(model, config, save_dir=MODEL_DIR)
        return model, trainer
    return factory


def main() -> dict:
    print("=" * 70)
    print("CRISPR Off-Target Platform – Full Pipeline (v4 – High Performance)")
    print("=" * 70)

    # ------------------------------------------------------------------ 1. Load
    print("\n[1/9] Loading data...")
    sequences, labels = load_sequences(config.data_path)
    n_on = int(labels.sum())
    print(
        f"   Loaded {len(sequences):,} | label=1: {n_on:,} | "
        f"label=0: {len(sequences) - n_on:,}"
    )
    feature_names = list(FEATURE_NAMES)

    # ------------------------------------------------------------------ 2. Preprocess
    print("\n[2/9] Preprocessing...")
    prep = Preprocessor(random_state=config.random_state)
    if config.filter_invalid:
        sequences, labels = prep.filter_sequences(sequences, labels)
    if config.balance_classes:
        sequences, labels = prep.balance_classes(
            sequences, labels, target_size=config.max_samples_per_class
        )

    print("   Extracting features (vectorised)...")
    X_all = extract_features(sequences)

    # ------------------------------------------------------------------ 3. Split
    print("\n[3/9] Splitting data (index-based)...")
    tr, va, te = prep.split_indices(labels, config.test_size, config.val_size)
    y_train, y_val, y_test = labels[tr], labels[va], labels[te]
    X_on_raw = X_all[tr][y_train == 1]
    X_train, X_val, X_test, _ = prep.normalize(
        X_all[tr], X_all[va], X_all[te]
    )

    # ------------------------------------------------------------------ 4. Diffusion
    print("\n[4/9] Diffusion Map...")
    diffusion = DiffusionMap(
        n_components=config.diffusion_n_components,
        sigma=config.diffusion_sigma,
        nystroem_samples=config.diffusion_nystroem_samples,
        batch_size=config.diffusion_batch_size,
        use_pca=config.diffusion_use_pca,
        pca_components=config.diffusion_pca_components,
        random_state=config.random_state,
    )
    X_train_diff = diffusion.fit_transform(X_train)
    X_val_diff = diffusion.transform(X_val)
    X_test_diff = diffusion.transform(X_test)

    print("\nExtracting 7 actionable factors...")
    extractor = FactorExtractor(feature_names)
    factors = extractor.extract_factors(
        extractor.compute_loadings(X_train, X_train_diff)
    )
    extractor.print_report(factors)

    # ------------------------------------------------------------------ 5. DNABert
    print("\n[5/9] DNABert / k-mer embedding...")
    dnabert = DNABertEncoder(
        model_name=config.dnabert_model_name,
        max_len=config.dnabert_max_len,
        batch_size=config.dnabert_batch_size,
        pooling=config.dnabert_pooling,
        device=config.device,
        use_amp=config.dnabert_use_amp,
        cache_dir=CACHE_DIR,
        kmer_k=config.kmer_k,
    )
    dnabert.load_model()
    E_all = dnabert.encode(sequences)
    E_train, E_val, E_test = E_all[tr], E_all[va], E_all[te]
    del E_all
    print(f"   Embedding shape: {E_train.shape}  (fallback={dnabert.use_fallback})")

    # PCA-reduce high-dim k-mer embeddings (keeps signal, cuts noise/dim)
    if E_train.shape[1] > 128:
        from sklearn.decomposition import PCA as _PCA
        n_comp = min(128, E_train.shape[0] - 1, E_train.shape[1])
        print(f"   PCA-reducing sequence embedding: {E_train.shape[1]} → {n_comp}")
        _pca_e = _PCA(n_components=n_comp, random_state=config.random_state)
        E_train = _pca_e.fit_transform(E_train).astype(np.float32)
        E_val = _pca_e.transform(E_val).astype(np.float32)
        E_test = _pca_e.transform(E_test).astype(np.float32)

    # ------------------------------------------------------------------ 6. New Embedding
    print("\n[6/9] New Embedding (features ⊕ diffusion ⊕ sequence emb)...")
    NewEmbed_train = np.concatenate(
        [X_train, X_train_diff, E_train], axis=1
    ).astype(np.float32)
    NewEmbed_val = np.concatenate(
        [X_val, X_val_diff, E_val], axis=1
    ).astype(np.float32)
    NewEmbed_test = np.concatenate(
        [X_test, X_test_diff, E_test], axis=1
    ).astype(np.float32)

    NewEmbed_train, NewEmbed_val, NewEmbed_test, keep_mask = _drop_constant_cols(
        NewEmbed_train, NewEmbed_val, NewEmbed_test
    )
    emb_scaler = StandardScaler()
    NewEmbed_train = emb_scaler.fit_transform(NewEmbed_train).astype(np.float32)
    NewEmbed_val = emb_scaler.transform(NewEmbed_val).astype(np.float32)
    NewEmbed_test = emb_scaler.transform(NewEmbed_test).astype(np.float32)
    print(f"   Shape after clean+scale: {NewEmbed_train.shape}")

    for name, arr in [
        ("NewEmbed_train", NewEmbed_train),
        ("NewEmbed_val", NewEmbed_val),
        ("NewEmbed_test", NewEmbed_test),
        ("y_train", y_train),
        ("y_val", y_val),
        ("y_test", y_test),
    ]:
        np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr)

    # ------------------------------------------------------------------ 7. GAT Training
    print("\n[7/9] GAT Training (mini-batch mode)...")
    factory = _make_trainer_factory(default_in_dim=NewEmbed_train.shape[1])
    gat, trainer = factory()
    n_params = sum(p.numel() for p in gat.parameters())
    print(f"   GAT parameters: {n_params:,}")

    history = trainer.fit(NewEmbed_train, y_train, NewEmbed_val, y_val)
    test_results = trainer.test(NewEmbed_test, y_test)

    plot_training_history(history, os.path.join(FIGURE_DIR, "training_history.png"))

    print("   Extracting attention weights (test set, first batch)...")
    attns = trainer.extract_attention(NewEmbed_test[: config.eval_batch_size])
    if attns:
        attn_path = os.path.join(OUTPUT_DIR, "attention_weights_layer0.npy")
        np.save(attn_path, attns[0].numpy())
        print(f"   Saved layer-0 attention: {attn_path}")

    # ------------------------------------------------------------------ 8. Statistics + paper figures
    print("\n[8/9] Statistical analyses & paper figures...")

    # --- Confusion matrix, ROC, PR (fast) ---
    print("   → Confusion matrix / ROC / PR curves ...")
    cm = plot_confusion_matrix(
        y_test, test_results["probs"],
        os.path.join(FIGURE_DIR, "confusion_matrix.png"),
    )
    roc_auc = plot_roc_curve(
        y_test, test_results["probs"],
        os.path.join(FIGURE_DIR, "roc_curve.png"),
    )
    ap = plot_pr_curve(
        y_test, test_results["probs"],
        os.path.join(FIGURE_DIR, "pr_curve.png"),
    )
    print(f"      ROC-AUC={roc_auc:.4f}  |  Average Precision={ap:.4f}")
    print(f"      Confusion matrix:\n{cm}")

    # --- Bootstrap CI ---
    print("\n   → Bootstrap CIs ...")
    boot = bootstrap_metrics(
        y_test,
        test_results["probs"],
        n_boot=config.bootstrap_iters,
        ci=config.confidence_level,
        seed=config.seed,
    )
    print("\n   Bootstrap 95% CI:")
    for m, v in boot.items():
        print(f"      {m:10s}: {v['mean']:.4f}  [{v['ci_low']:.4f} – {v['ci_high']:.4f}]")
    plot_bootstrap_ci(boot, os.path.join(FIGURE_DIR, "bootstrap_ci.png"))

    # --- Ablation ---
    print("\n   → Ablation study ...")
    def _scale(tr, va, te):
        sc = StandardScaler()
        return (
            sc.fit_transform(tr).astype(np.float32),
            sc.transform(va).astype(np.float32),
            sc.transform(te).astype(np.float32),
        )

    Xf_tr, Xf_va, Xf_te = _scale(X_train, X_val, X_test)
    Xd_tr, Xd_va, Xd_te = _scale(X_train_diff, X_val_diff, X_test_diff)
    Xe_tr, Xe_va, Xe_te = _scale(E_train, E_val, E_test)

    Xfd_tr = np.concatenate([Xf_tr, Xd_tr], 1)
    Xfd_va = np.concatenate([Xf_va, Xd_va], 1)
    Xfd_te = np.concatenate([Xf_te, Xd_te], 1)
    Xfd_tr, Xfd_va, Xfd_te = _scale(Xfd_tr, Xfd_va, Xfd_te)

    Xfe_tr = np.concatenate([Xf_tr, Xe_tr], 1)
    Xfe_va = np.concatenate([Xf_va, Xe_va], 1)
    Xfe_te = np.concatenate([Xf_te, Xe_te], 1)
    Xfe_tr, Xfe_va, Xfe_te = _scale(Xfe_tr, Xfe_va, Xfe_te)

    Xde_tr = np.concatenate([Xd_tr, Xe_tr], 1)
    Xde_va = np.concatenate([Xd_va, Xe_va], 1)
    Xde_te = np.concatenate([Xd_te, Xe_te], 1)
    Xde_tr, Xde_va, Xde_te = _scale(Xde_tr, Xde_va, Xde_te)

    ablation_components = {
        "Full (Feat+Diff+Emb)": (NewEmbed_train, NewEmbed_val, NewEmbed_test),
        "Feat only": (Xf_tr, Xf_va, Xf_te),
        "Diff only": (Xd_tr, Xd_va, Xd_te),
        "Emb only": (Xe_tr, Xe_va, Xe_te),
        "Feat+Diff": (Xfd_tr, Xfd_va, Xfd_te),
        "Feat+Emb": (Xfe_tr, Xfe_va, Xfe_te),
        "Diff+Emb": (Xde_tr, Xde_va, Xde_te),
    }
    ablation_results = run_ablation(
        ablation_components,
        y_train, y_val, y_test,
        factory,
        FIGURE_DIR,
    )
    print("\n   Ablation summary:")
    for name, r in sorted(ablation_results.items(), key=lambda x: -x[1]["auc"]):
        print(f"      {name:25s}: AUC={r['auc']:.4f}  F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}")

    # --- Label Permutation Test ---
    print("\n   → Label Permutation (Randomization) Test ...")
    def perm_factory():
        return factory(in_dim=NewEmbed_train.shape[1])

    perm_result = permutation_test(
        perm_factory,
        NewEmbed_train, y_train,
        NewEmbed_val, y_val,
        NewEmbed_test, y_test,
        observed_auc=test_results["auc"],
        n_perm=config.n_permutation,
        seed=config.seed,
    )
    print(f"\n   Permutation test: observed AUC={perm_result['observed_auc']:.4f}")
    print(f"      Null mean±std  = {perm_result['null_mean']:.4f} ± {perm_result['null_std']:.4f}")
    print(f"      p-value        = {perm_result['p_value']:.4f}")
    print(f"      p-value (report) ≤ {perm_result.get('p_value_report', perm_result['p_value']):.4f}")
    plot_permutation(perm_result, os.path.join(FIGURE_DIR, "permutation_test.png"))

    # --- Paper report (txt + json) ---
    write_paper_report(
        test_results, boot, ablation_results, perm_result,
        cm, ap, FIGURE_DIR, OUTPUT_DIR,
    )

    # ------------------------------------------------------------------ 9. Checklist
    print("\n[9/9] Generating checklist...")
    ChecklistGenerator(feature_names).generate(
        X_on_raw,
        output_path=os.path.join(CHECKLIST_DIR, "checklist_7_factors.txt"),
    )

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        print(f"   {k.capitalize():<10s}: {test_results[k]:.4f}")
    print(f"   AP (PR)   : {ap:.4f}")
    print(
        f"\nOutputs:   {OUTPUT_DIR}/\n"
        f"Figures:   {FIGURE_DIR}/\n"
        f"  - training_history.png\n"
        f"  - bootstrap_ci.png\n"
        f"  - confusion_matrix.png\n"
        f"  - roc_curve.png\n"
        f"  - pr_curve.png\n"
        f"  - ablation_study.png\n"
        f"  - permutation_test.png\n"
        f"Report:    {OUTPUT_DIR}/paper_metrics_report.txt\n"
        f"Checklist: {CHECKLIST_DIR}/\n"
        f"Model:     {MODEL_DIR}/best_gat.pt"
    )
    return {
        "test_results": test_results,
        "history": history,
        "factors": factors,
        "bootstrap": boot,
        "ablation": ablation_results,
        "permutation": perm_result,
        "confusion_matrix": cm,
        "average_precision": ap,
    }


if __name__ == "__main__":
    results = main()
