"""Pseudobulk differential expression (replicate-aware) and BH correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from perturbai.features.effects import group_means


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(ranked, 1.0)
    return out


def _replicates(X, labels, batch, target, n_pseudoreps, min_cells_rep, seed):
    """Return (pseudobulk means [n_reps, genes]) for one label. Uses batches if given, else random splits."""
    idx = np.where(labels == target)[0]
    if batch is not None:
        b = batch[idx]
        keys = b
    else:
        rng = np.random.default_rng(seed)
        keys = rng.integers(0, n_pseudoreps, len(idx)).astype(str)
    M, g, c = group_means(X[idx], keys)
    return M[c >= min_cells_rep], g[c >= min_cells_rep]


def pseudobulk_de(X, labels, var_names, pert: str, control_label="control", batch=None,
                  n_pseudoreps: int = 5, min_cells_rep: int = 5, seed: int = 0) -> pd.DataFrame:
    """Welch t-test on pseudobulk replicates (batches if provided, else random pseudo-replicates).
    X must be log-normalised. 'log_fc' is the difference of mean log-expression."""
    labels = np.asarray(labels).astype(str)
    batch = None if batch is None else np.asarray(batch).astype(str)
    P, _ = _replicates(X, labels, batch, pert, n_pseudoreps, min_cells_rep, seed)
    C, _ = _replicates(X, labels, batch, control_label, n_pseudoreps, min_cells_rep, seed + 1)
    if len(P) < 2 or len(C) < 2:
        raise ValueError(f"Not enough replicates for {pert}: {len(P)} vs {len(C)}")
    t, p = stats.ttest_ind(P, C, axis=0, equal_var=False)
    p = np.nan_to_num(p, nan=1.0)
    df = pd.DataFrame({"log_fc": P.mean(0) - C.mean(0), "t": np.nan_to_num(t), "pval": p,
                       "padj": bh_fdr(p)}, index=pd.Index(var_names, name="gene"))
    df["n_rep_pert"], df["n_rep_ctrl"] = len(P), len(C)
    return df.sort_values("pval")
