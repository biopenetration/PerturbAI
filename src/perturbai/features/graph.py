"""Gene graphs and graph-propagated embeddings (lightweight alternative to full GNNs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def knn_graph(emb: np.ndarray, k: int = 15) -> sp.csr_matrix:
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=min(k, S.shape[0] - 1) - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(S.shape[0]), idx.shape[1])
    A = sp.csr_matrix((np.ones(rows.size), (rows, idx.ravel())), shape=S.shape)
    return ((A + A.T) > 0).astype(np.float32)


def string_adjacency(edges: pd.DataFrame, genes, a="protein1", b="protein2",
                     score="combined_score", threshold: float = 700) -> sp.csr_matrix:
    """Build adjacency from a STRING-style edge table already mapped to gene symbols."""
    gi = {g: i for i, g in enumerate(genes)}
    e = edges[(edges[score] >= threshold) & edges[a].isin(gi) & edges[b].isin(gi)]
    r = e[a].map(gi).values; c = e[b].map(gi).values
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(gi), len(gi)))
    return ((A + A.T) > 0).astype(np.float32)


def propagate(A: sp.csr_matrix, E: np.ndarray, steps: int = 2, alpha: float = 0.5) -> np.ndarray:
    """E_t = alpha * E_0 + (1-alpha) * D^-1/2 (A+I) D^-1/2 E_{t-1}."""
    A = A + sp.eye(A.shape[0])
    d = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(1.0 / np.sqrt(d))
    An = Dinv @ A @ Dinv
    Z = E.copy()
    for _ in range(steps):
        Z = alpha * E + (1 - alpha) * (An @ Z)
    return Z
