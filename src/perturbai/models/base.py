from __future__ import annotations

import numpy as np


class PerturbModel:
    """X: perturbation features (n_perts x d). Y: delta expression (n_perts x n_genes)."""
    name = "base"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "PerturbModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
