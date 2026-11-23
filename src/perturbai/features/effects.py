"""Perturbation effect estimation: delta = mean(perturbed) - mean(control).

All functions work on plain matrices (dense or scipy sparse) + label arrays, so they are
testable without AnnData. Use `effects_from_adata` for the AnnData convenience wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class EffectResult:
    delta: pd.DataFrame      # perturbations x genes
    ctrl_mean: pd.Series     # genes
    n_cells: pd.Series       # perturbations


def group_means(X, keys):
    """Mean expression per group. Returns (means [n_groups, n_genes], group names, counts)."""
    keys = np.asarray(keys).astype(str)
    uniq, inv = np.unique(keys, return_inverse=True)
    n = len(keys)
    G = sp.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(len(uniq), n))
    counts = np.asarray(G.sum(axis=1)).ravel()
    S = G @ X
    S = S.toarray() if sp.issparse(S) else np.asarray(S)
    return S / counts[:, None], uniq, counts


def compute_effects(X, labels, var_names, control_label: str = "control",
                    batch=None, min_cells: int = 20) -> EffectResult:
    labels = np.asarray(labels).astype(str)
    var_names = pd.Index(var_names)
    if not (labels == control_label).any():
        raise ValueError(f"No cells with control label '{control_label}'")
    M0, g0, c0 = group_means(X, labels)
    gi = {g: i for i, g in enumerate(g0)}
    ctrl_mean = pd.Series(M0[gi[control_label]], index=var_names)

    if batch is None:
        perts = [g for g in g0 if g != control_label and c0[gi[g]] >= min_cells]
        rows = [gi[g] for g in perts]
        delta = pd.DataFrame(M0[rows] - ctrl_mean.values, index=perts, columns=var_names)
        n = pd.Series(c0[rows], index=perts)
        return EffectResult(delta, ctrl_mean, n)

    # batch-matched: compare each (pert, batch) mean to the control of the same batch
    batch = np.asarray(batch).astype(str)
    keys = np.char.add(np.char.add(labels, "||"), batch)
    M, groups, cnt = group_means(X, keys)
    parsed = pd.DataFrame([g.rsplit("||", 1) for g in groups], columns=["pert", "batch"])
    parsed["i"] = np.arange(len(groups)); parsed["n"] = cnt
    ctrl = parsed[parsed.pert == control_label].set_index("batch")
    out, ns = {}, {}
    for p, sub in parsed[parsed.pert != control_label].groupby("pert"):
        sub = sub[sub.batch.isin(ctrl.index)]
        if sub.empty or sub.n.sum() < min_cells:
            continue
        d = np.zeros(M.shape[1])
        for _, r in sub.iterrows():
            d += r.n * (M[int(r.i)] - M[int(ctrl.loc[r.batch, "i"])])
        out[p] = d / sub.n.sum(); ns[p] = int(sub.n.sum())
    delta = pd.DataFrame.from_dict(out, orient="index", columns=var_names)
    return EffectResult(delta, ctrl_mean, pd.Series(ns))


def effects_from_adata(adata, pert_col="perturbation", control_label="control",
                       batch_col: str | None = None, min_cells: int = 20, layer: str | None = None):
    X = adata.layers[layer] if layer else adata.X
    batch = adata.obs[batch_col].values if batch_col and batch_col in adata.obs else None
    return compute_effects(X, adata.obs[pert_col].values, adata.var_names, control_label, batch, min_cells)


def split_half_effects(X, labels, var_names, control_label="control", batch=None,
                       min_cells: int = 20, seed: int = 0):
    """Estimate effects from two random halves of each perturbation's cells (noise ceiling)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    half = rng.integers(0, 2, len(labels))
    res = []
    for h in (0, 1):
        m = half == h
        res.append(compute_effects(X[m], labels[m], var_names, control_label,
                                   None if batch is None else np.asarray(batch)[m], min_cells // 2))
    common = res[0].delta.index.intersection(res[1].delta.index)
    return res[0].delta.loc[common], res[1].delta.loc[common]


def knockdown_efficiency(X_norm, labels, var_names, control_label="control",
                         min_target_expr: float = 0.05) -> pd.DataFrame:
    """On-target knockdown on NON-log normalized expression.
    kd = 1 - mean(target gene | perturbed) / mean(target gene | control)."""
    M, groups, cnt = group_means(X_norm, labels)
    gi = {g: i for i, g in enumerate(groups)}
    var_idx = {g: i for i, g in enumerate(pd.Index(var_names))}
    ctrl = M[gi[control_label]]
    rows = []
    for g in groups:
        if g == control_label:
            continue
        j = var_idx.get(g)
        if j is None:
            rows.append((g, np.nan, np.nan, np.nan, int(cnt[gi[g]]), False)); continue
        c, p = ctrl[j], M[gi[g], j]
        rows.append((g, c, p, 1 - p / c if c > 0 else np.nan, int(cnt[gi[g]]), c >= min_target_expr))
    return pd.DataFrame(rows, columns=["perturbation", "ctrl_expr", "pert_expr", "knockdown",
                                       "n_cells", "assessable"]).set_index("perturbation")


def effect_size_summary(delta: pd.DataFrame, thr: float = 0.1) -> pd.DataFrame:
    a = delta.values
    return pd.DataFrame({"l2_norm": np.linalg.norm(a, axis=1),
                         "n_genes_changed": (np.abs(a) > thr).sum(1),
                         "frac_down": ((a < -thr).sum(1) / np.maximum((np.abs(a) > thr).sum(1), 1))},
                        index=delta.index)
