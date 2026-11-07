"""
Central configuration – CRISPR Off-Target Platform (v4 – high-performance)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models" / "saved"
FIGURE_DIR = OUTPUT_DIR / "figures"
CHECKLIST_DIR = OUTPUT_DIR / "checklists"
CACHE_DIR = BASE_DIR / "cache"

for d in (OUTPUT_DIR, MODEL_DIR, FIGURE_DIR, CHECKLIST_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    OUTPUT_DIR: str = str(OUTPUT_DIR)
    MODEL_DIR: str = str(MODEL_DIR)
    FIGURE_DIR: str = str(FIGURE_DIR)
    CHECKLIST_DIR: str = str(CHECKLIST_DIR)
    CACHE_DIR: str = str(CACHE_DIR)

    # Data
    data_path: str = "/kaggle/input/datasets/hassansalari588/crisper/paper_data-classification"
    proteomic_path: Optional[str] = None
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    balance_classes: bool = True
    max_samples_per_class: int = 20_000   # keep more data after softer filter
    filter_invalid: bool = True
    n_features: int = 73

    # Diffusion Map
    diffusion_n_components: int = 8
    diffusion_sigma: Union[float, str] = "median"
    diffusion_nystroem_samples: int = 1_500
    diffusion_batch_size: int = 8_192
    diffusion_use_pca: bool = True
    diffusion_pca_components: int = 30

    # DNABert / sequence embedding
    dnabert_model_name: str = "zhihan1996/DNABERT-2-117M"
    dnabert_max_len: int = 36
    dnabert_batch_size: int = 128
    dnabert_pooling: str = "mean"
    dnabert_use_amp: bool = True
    dnabert_use_fallback: bool = True
    kmer_k: int = 5                       # richer fallback (4^5 = 1024-d)

    # GAT architecture (increased capacity)
    gat_hidden_dim: int = 256
    gat_n_heads: int = 8
    gat_n_layers: int = 4
    gat_dropout: float = 0.15
    gat_k_neighbors: int = 12
    use_edge_attr: bool = False

    # Training
    batch_size: int = 256
    eval_batch_size: int = 256
    epochs: int = 150
    learning_rate: float = 8e-4
    weight_decay: float = 3e-4
    patience: int = 30
    gradient_clip: float = 1.0
    label_smoothing: float = 0.05
    use_pos_weight: bool = True           # handle residual imbalance
    mixup_alpha: float = 0.15             # light mixup on features

    # Statistics
    bootstrap_iters: int = 1_000
    confidence_level: float = 0.95
    n_permutation: int = 100              # label permutation test
    ablation_runs: int = 1                # full ablation (set >1 for mean±std)

    # Runtime
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self) -> None:
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print(f"CUDA not available → using {self.device}")


def get_config() -> Config:
    return Config()
