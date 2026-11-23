"""Classical baselines. Every deep model must beat these to justify its complexity."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from perturbai.models.base import PerturbModel


class ControlBaseline(PerturbModel):
    """Predicts no change (delta = 0). The minimum bar."""
    name = "control"

    def fit(self, X, Y):
        self.n_genes = Y.shape[1]; return self

    def predict(self, X):
        return np.zeros((len(X), self.n_genes))


class MeanBaseline(PerturbModel):
    """Predicts the average training delta for every perturbation (surprisingly hard to beat)."""
    name = "mean"

    def fit(self, X, Y):
        self.mu = Y.mean(0, keepdims=True); return self

    def predict(self, X):
        return np.repeat(self.mu, len(X), axis=0)


class RidgeModel(PerturbModel):
    name = "ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, X, Y):
        self.sc = StandardScaler().fit(X)
        self.m = Ridge(alpha=self.alpha).fit(self.sc.transform(X), Y); return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(X))


class KNNModel(PerturbModel):
    """Similarity-weighted average of the deltas of the k most similar training perturbations."""
    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k

    @staticmethod
    def _norm(A):
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)

    def fit(self, X, Y):
        self.mu_x = X.mean(0, keepdims=True)
        self.Xt = self._norm(X - self.mu_x); self.Y = Y; return self

    def predict(self, X):
        S = self._norm(X - self.mu_x) @ self.Xt.T
        k = min(self.k, S.shape[1])
        idx = np.argsort(-S, axis=1)[:, :k]
        w = np.clip(np.take_along_axis(S, idx, 1), 0, None) + 1e-6
        w = w / w.sum(1, keepdims=True)
        return np.einsum("nk,nkg->ng", w, self.Y[idx])


class RandomForestModel(PerturbModel):
    """RF predicting PCA-compressed deltas (keeps multi-output training tractable)."""
    name = "rf"

    def __init__(self, n_estimators=200, n_pca_targets=50, min_samples_leaf=2, seed=0, n_jobs=-1):
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                       max_features=0.5, random_state=seed, n_jobs=n_jobs)
        self.n_pca = n_pca_targets

    def fit(self, X, Y):
        k = max(1, min(self.n_pca, Y.shape[0] - 1, Y.shape[1]))
        self.pca = PCA(n_components=k, random_state=0).fit(Y)
        self.rf = RandomForestRegressor(**self.kw).fit(X, self.pca.transform(Y)); return self

    def predict(self, X):
        return self.pca.inverse_transform(self.rf.predict(X))


def predict_additive(single_delta: pd.DataFrame, combos, sep: str = "+") -> pd.DataFrame:
    """Combinatorial baseline: delta(A+B) = delta(A) + delta(B). Rows with missing singles are dropped."""
    rows, names = [], []
    for c in combos:
        parts = c.split(sep)
        if all(p in single_delta.index for p in parts):
            rows.append(single_delta.loc[parts].sum(0).values); names.append(c)
    return pd.DataFrame(rows, index=names, columns=single_delta.columns)
