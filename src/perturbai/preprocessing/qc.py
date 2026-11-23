"""Cell/gene QC, normalisation, perturbation efficacy filtering, subsetting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def row_sums(X):
    return np.asarray(X.sum(axis=1)).ravel()


def normalize_log1p_matrix(X, target_sum: float = 1e4):
    """Library-size normalise to target_sum then log1p. Returns (norm_nonlog, lognorm)."""
    X = X.tocsr().astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    s = row_sums(X); s[s == 0] = 1
    scale = (target_sum / s).astype(np.float32)
    if sp.issparse(X):
        Xn = sp.diags(scale) @ X
        Xl = Xn.copy(); Xl.data = np.log1p(Xl.data)
    else:
        Xn = X * scale[:, None]; Xl = np.log1p(Xn)
    return Xn.astype(np.float32), Xl.astype(np.float32)


def cell_qc_metrics(adata, mito_prefix: str = "MT-") -> None:
    X = adata.X
    adata.obs["n_counts"] = row_sums(X)
    adata.obs["n_genes_expr"] = np.asarray((X > 0).sum(axis=1)).ravel()
    mito = adata.var_names.str.upper().str.startswith(mito_prefix)
    adata.obs["pct_mito"] = (row_sums(X[:, mito]) / np.maximum(adata.obs["n_counts"], 1) * 100) if mito.any() else 0.0


def filter_cells_genes(adata, min_genes=200, min_cells=10, max_pct_mito=20.0):
    keep = (adata.obs["n_genes_expr"] >= min_genes) & (adata.obs["pct_mito"] <= max_pct_mito)
    adata = adata[keep.values].copy()
    gene_cells = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    return adata[:, gene_cells >= min_cells].copy()


def select_genes(Xlog, var_names, n_top: int, force_include=()) -> np.ndarray:
    """Top-variance genes (+ forced genes, e.g. perturbation targets). Returns boolean mask."""
    mean = np.asarray(Xlog.mean(axis=0)).ravel()
    sq = np.asarray(Xlog.multiply(Xlog).mean(axis=0)).ravel() if sp.issparse(Xlog) else (Xlog ** 2).mean(0)
    var = sq - mean ** 2
    mask = np.zeros(len(var), bool)
    mask[np.argsort(-var)[:n_top]] = True
    mask |= pd.Index(var_names).isin(list(force_include))
    return mask


def choose_perturbations(labels, control_label, n_perts, min_cells, seed=0, candidates=None):
    """Pick up to n_perts perturbations with >= min_cells cells, randomly (reproducible)."""
    counts = pd.Series(np.asarray(labels).astype(str)).value_counts()
    counts = counts[(counts.index != control_label) & (counts >= min_cells)]
    if candidates is not None:
        counts = counts[counts.index.isin(candidates)]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(counts))[:n_perts]
    return list(counts.index[idx])


def subsample_cells(labels, control_label, keep_perts, max_ctrl, max_per_pert, seed=0) -> np.ndarray:
    """Boolean mask: controls capped at max_ctrl, each kept perturbation capped at max_per_pert."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    mask = np.zeros(len(labels), bool)
    ci = np.where(labels == control_label)[0]
    mask[rng.choice(ci, min(max_ctrl, len(ci)), replace=False)] = True
    for p in keep_perts:
        pi = np.where(labels == p)[0]
        mask[rng.choice(pi, min(max_per_pert, len(pi)), replace=False)] = True
    return mask
