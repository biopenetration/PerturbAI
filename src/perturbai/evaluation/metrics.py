"""Perturbation-response metrics computed on delta = (perturbed - control), per perturbation.

Why delta? Correlating raw expression is dominated by baseline gene expression (~0.99 for any model).
Delta metrics measure whether the model predicts the CHANGE caused by the perturbation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rowwise_pearson(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    num = (A * B).sum(1); den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def rowwise_spearman(A, B):
    return rowwise_pearson(rankdata(A, axis=1), rankdata(B, axis=1))


def per_perturbation_metrics(pred, true, ctrl_mean=None, top_k: int = 20, index=None) -> pd.DataFrame:
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    sse = ((pred - true) ** 2).sum(1)
    sst = ((true - true.mean(1, keepdims=True)) ** 2).sum(1)
    k = min(top_k, true.shape[1])
    top = np.argsort(-np.abs(true), axis=1)[:, :k]
    tp = np.take_along_axis(pred, top, 1); tt = np.take_along_axis(true, top, 1)
    out = {
        "mse_delta": ((pred - true) ** 2).mean(1),
        "mae_delta": np.abs(pred - true).mean(1),
        "r2_delta": np.where(sst > 0, 1 - sse / np.where(sst > 0, sst, 1), np.nan),
        "pearson_delta": rowwise_pearson(pred, true),
        "spearman_delta": rowwise_spearman(pred, true),
        "mse_top_deg": ((tp - tt) ** 2).mean(1),
        "direction_acc_top_deg": (np.sign(tp) == np.sign(tt)).mean(1),
    }
    if ctrl_mean is not None:
        c = np.asarray(ctrl_mean, float)[None, :]
        out["pearson_expr"] = rowwise_pearson(pred + c, true + c)   # inflated; reported for illustration
    return pd.DataFrame(out, index=index)
