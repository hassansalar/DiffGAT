"""
GAT trainer with label smoothing, mixup, cosine LR, consistent batch graphs,
pos_weight, and PyTorch-2.6-safe checkpoints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gat.graph_builder import build_batch_graph

logger = logging.getLogger(__name__)


class GATTrainer:
    def __init__(self, model: nn.Module, config: Any, save_dir: str | Path) -> None:
        self.model = model
        self.config = config
        self.device = config.device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        ls = float(getattr(config, "label_smoothing", 0.0))
        self.label_smoothing = ls
        self.mixup_alpha = float(getattr(config, "mixup_alpha", 0.0))
        self.use_pos_weight = bool(getattr(config, "use_pos_weight", False))
        self.pos_weight: Optional[torch.Tensor] = None

        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        self.best_val_auc = 0.0
        self.no_improve = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_f1": [],
        }

        self.k_graph = getattr(config, "gat_k_neighbors", 15)
        self.eval_batch_size = getattr(config, "eval_batch_size", config.batch_size)

    def set_pos_weight(self, y_train: np.ndarray) -> None:
        if not self.use_pos_weight:
            return
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        if n_pos > 0:
            w = n_neg / n_pos
            self.pos_weight = torch.tensor([w], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            logger.info("Using pos_weight = %.3f", w)

    def _tensor(self, a: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=self.device)

    def _smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0:
            return y
        eps = self.label_smoothing
        return y * (1.0 - eps) + 0.5 * eps

    def _mixup(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mixup_alpha <= 0 or X.size(0) < 2:
            return X, y
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        idx = torch.randperm(X.size(0), device=X.device)
        X_mix = lam * X + (1 - lam) * X[idx]
        y_mix = lam * y + (1 - lam) * y[idx]
        return X_mix, y_mix

    def _train_step(self, X_b: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        X_b, y_b = self._mixup(X_b, y_b)
        edge_index = build_batch_graph(X_b, k=self.k_graph)
        self.optimizer.zero_grad(set_to_none=True)
        logits, _ = self.model(X_b, edge_index)
        loss = self.criterion(logits, self._smooth_labels(y_b))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        return loss.detach()

    def train_epoch(
        self, X: torch.Tensor, y: torch.Tensor, batch_size: int
    ) -> float:
        self.model.train()
        n = X.size(0)
        perm = torch.randperm(n, device=self.device)
        total = torch.zeros((), device=self.device)
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            total += self._train_step(X[idx], y[idx])
            n_batches += 1
        self.scheduler.step()
        return (total / max(n_batches, 1)).item()

    @torch.no_grad()
    def evaluate(
        self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> Dict[str, Any]:
        self.model.eval()
        X_t = self._tensor(X)
        y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        y_t = self._tensor(y_np)

        probs, losses = [], []
        for i in range(0, X_t.size(0), self.eval_batch_size):
            X_b = X_t[i : i + self.eval_batch_size]
            y_b = y_t[i : i + self.eval_batch_size]
            logits, _ = self.model(X_b, build_batch_graph(X_b, k=self.k_graph))
            losses.append(self.criterion(logits, y_b))
            probs.append(torch.sigmoid(logits))
        probs_arr = torch.cat(probs).cpu().numpy()
        loss = torch.stack(losses).mean().item()

        preds = (probs_arr > 0.5).astype(int)
        return {
            "loss": loss,
            "accuracy": accuracy_score(y_np, preds),
            "precision": precision_score(y_np, preds, zero_division=0),
            "recall": recall_score(y_np, preds, zero_division=0),
            "f1": f1_score(y_np, preds, zero_division=0),
            "auc": roc_auc_score(y_np, probs_arr) if len(np.unique(y_np)) > 1 else 0.5,
            "probs": probs_arr,
        }

    @torch.no_grad()
    def extract_attention(
        self, X: np.ndarray | torch.Tensor, batch_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        self.model.eval()
        X_t = self._tensor(X)
        bs = batch_size or self.eval_batch_size
        all_attns: List[List[torch.Tensor]] = []
        for i in range(0, X_t.size(0), bs):
            X_b = X_t[i : i + bs]
            edge_index = build_batch_graph(X_b, k=self.k_graph)
            _, attns = self.model(X_b, edge_index, return_attn=True)
            if attns is not None:
                all_attns.append([a.cpu() for a in attns])
        if not all_attns:
            return []
        n_layers = len(all_attns[0])
        return [
            torch.cat([batch[l] for batch in all_attns], dim=0)
            for l in range(n_layers)
        ]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        logger.info("Starting GAT training (mini-batch mode)")
        logger.info(
            "   Train: %s | Val: %s | batch %d | k=%d",
            f"{len(X_train):,}",
            f"{len(X_val):,}",
            self.config.batch_size,
            self.k_graph,
        )
        self.set_pos_weight(y_train)

        X_tr = self._tensor(X_train)
        y_tr = self._tensor(y_train)
        X_va = self._tensor(X_val)

        ckpt_path = self.save_dir / "best_gat.pt"
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(X_tr, y_tr, self.config.batch_size)
            val = self.evaluate(X_va, y_val)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val["loss"])
            self.history["val_auc"].append(val["auc"])
            self.history["val_f1"].append(val["f1"])
            logger.info(
                "Epoch %3d | Train Loss: %.4f | Val Loss: %.4f | "
                "Val AUC: %.4f | Val F1: %.4f",
                epoch + 1,
                train_loss,
                val["loss"],
                val["auc"],
                val["f1"],
            )

            if val["auc"] > self.best_val_auc:
                self.best_val_auc = val["auc"]
                self.no_improve = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "best_val_auc": float(self.best_val_auc),
                        "epoch": int(epoch),
                    },
                    ckpt_path,
                )
            else:
                self.no_improve += 1
                if self.no_improve >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Training complete. Best AUC: %.4f", self.best_val_auc)
        return self.history

    def test(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        logger.info("Testing...")
        r = self.evaluate(X_test, y_test)
        for k in ("accuracy", "precision", "recall", "f1", "auc"):
            logger.info("   %-9s: %.4f", k, r[k])
        return r
