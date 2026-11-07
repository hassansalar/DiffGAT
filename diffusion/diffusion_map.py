"""
Diffusion Map with Nystrom out-of-sample extension.
Default bandwidth is the median heuristic (robust for standardised features).
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

logger = logging.getLogger(__name__)


class DiffusionMap:
    def __init__(
        self,
        n_components: int = 8,
        sigma: Union[float, str] = "median",
        nystroem_samples: int = 1_500,
        batch_size: int = 8_192,
        use_pca: bool = True,
        pca_components: int = 30,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.sigma = sigma
        self.nystroem_samples = nystroem_samples
        self.batch_size = batch_size
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state
        self.gamma: Optional[float] = (
            None if sigma == "median" else 1.0 / (2.0 * float(sigma) ** 2)
        )

        self.pca: Optional[PCA] = None
        self.reference_data: Optional[np.ndarray] = None
        self.reference_embedding: Optional[np.ndarray] = None
        self.D_ref_inv: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        logger.info("Diffusion Map on %s samples", f"{X.shape[0]:,}")

        if self.use_pca and X.shape[1] > self.pca_components:
            logger.info("   PCA: %d → %d", X.shape[1], self.pca_components)
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X = self.pca.fit_transform(X).astype(np.float32)

        n_ref = min(self.nystroem_samples, len(X))
        rng = np.random.default_rng(self.random_state)
        X_ref = X[rng.choice(len(X), n_ref, replace=False)].copy()

        d2 = euclidean_distances(X_ref, squared=True)
        med_d2 = (
            float(np.median(d2[np.triu_indices(n_ref, 1)])) if n_ref > 1 else 1.0
        )
        if self.sigma == "median":
            self.gamma = 1.0 / max(med_d2, 1e-12)
            logger.info("   sigma='median' → σ ≈ %.3f", np.sqrt(med_d2 / 2))
        elif self.gamma is not None and np.exp(-self.gamma * med_d2) < 1e-6:
            logger.warning(
                "   sigma=%.3f is tiny relative to data (median d²=%.1f); "
                "typical affinity %.1e → kernel ≈ identity. "
                "Consider sigma='median' (≈%.2f).",
                float(self.sigma),
                med_d2,
                np.exp(-self.gamma * med_d2),
                np.sqrt(med_d2 / 2),
            )
        del d2

        logger.info("   Fitting SpectralEmbedding on %s references", f"{n_ref:,}")
        model = SpectralEmbedding(
            n_components=self.n_components,
            affinity="rbf",
            gamma=self.gamma,
            random_state=self.random_state,
            n_jobs=-1,
        )
        embedding_ref = model.fit_transform(X_ref)

        K_ref = getattr(model, "affinity_matrix_", None)
        if K_ref is None or getattr(K_ref, "shape", None) != (n_ref, n_ref):
            K_ref = rbf_kernel(X_ref, X_ref, gamma=self.gamma)
        D_ref = np.asarray(K_ref.sum(axis=1)).ravel()
        self.D_ref_inv = (1.0 / np.sqrt(D_ref + 1e-10)).astype(np.float32)

        self.reference_data = X_ref
        self.reference_embedding = embedding_ref.astype(np.float32)

        logger.info("   Computing Nystrom extension")
        embedding = self._nystroem_batched(X)
        logger.info("Diffusion Map done: %s", embedding.shape)
        return embedding

    def _nystroem_batched(self, X: np.ndarray) -> np.ndarray:
        assert self.reference_embedding is not None and self.D_ref_inv is not None
        n = len(X)
        out = np.empty((n, self.n_components), dtype=np.float32)
        W = self.reference_embedding * self.D_ref_inv[:, None]
        for s in range(0, n, self.batch_size):
            K_b = rbf_kernel(
                X[s : s + self.batch_size], self.reference_data, gamma=self.gamma
            )
            out[s : s + self.batch_size] = K_b @ W
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)
        return self._nystroem_batched(X)
