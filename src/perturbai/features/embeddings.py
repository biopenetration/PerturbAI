"""Gene representations used to encode an (unseen) perturbation as model input.

Leakage note: embeddings derived ONLY from control cells or external knowledge carry no information
about perturbation effects, so they are safe to compute before train/test splitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def coexpression_embedding(Xlog_ctrl, var_names, n_components: int = 64, seed: int = 0) -> pd.DataFrame:
    """SVD of z-scored control expression (cells x genes) -> gene embeddings (genes x k)."""
    X = Xlog_ctrl.toarray() if sp.issparse(Xlog_ctrl) else np.asarray(Xlog_ctrl)
    X = X.astype(np.float32)
    X = X - X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X = X / sd
    k = min(n_components, min(X.shape) - 1)
    U, S, Vt = randomized_svd(X, n_components=k, random_state=seed)
    emb = Vt.T * np.sqrt(S)          # genes x k
    return pd.DataFrame(emb, index=pd.Index(var_names), columns=[f"coexp_{i}" for i in range(k)])


def pathway_membership_matrix(gene_sets: dict, genes, min_size: int = 5, max_size: int = 500) -> pd.DataFrame:
    genes = pd.Index(genes)
    cols = {}
    for name, members in gene_sets.items():
        inter = genes.isin(list(members))
        if min_size <= inter.sum() <= max_size:
            cols[name] = inter.astype(np.float32)
    return pd.DataFrame(cols, index=genes)


def perturbation_feature_matrix(pert_names, gene_features: pd.DataFrame):
    """Map perturbation names to gene feature rows. Returns (features DataFrame, missing names)."""
    pert_names = pd.Index(pert_names)
    present = pert_names.isin(gene_features.index)
    return gene_features.loc[pert_names[present]], list(pert_names[~present])
