"""Multi-seed, perturbation-level benchmark with uncertainty."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from perturbai.evaluation.metrics import per_perturbation_metrics
from perturbai.evaluation.splits import assert_no_leakage, split_perturbations

METRICS = ["pearson_delta", "spearman_delta", "mse_delta", "mae_delta", "r2_delta",
           "direction_acc_top_deg", "mse_top_deg"]


def run_benchmark(factories: dict[str, Callable], X: np.ndarray, Y: np.ndarray, index,
                  ctrl_mean=None, seeds=(0, 1, 2, 3, 4), test_frac=0.2, top_k=20, verbose=True) -> pd.DataFrame:
    """Unseen-perturbation benchmark. X, Y rows align with `index` (perturbation names)."""
    index = np.asarray(index).astype(str)
    pos = {p: i for i, p in enumerate(index)}
    frames = []
    for seed in seeds:
        sp_ = split_perturbations(index, test_frac=test_frac, seed=seed)
        assert_no_leakage(sp_["train"], sp_["test"])
        tr = [pos[p] for p in sp_["train"]]; te = [pos[p] for p in sp_["test"]]
        for name, make in factories.items():
            model = make()
            model.fit(X[tr], Y[tr])
            pred = model.predict(X[te])
            df = per_perturbation_metrics(pred, Y[te], ctrl_mean, top_k, index=index[te])
            df.index.name = "perturbation"
            df = df.reset_index().assign(model=name, seed=seed)
            frames.append(df)
            if verbose:
                print(f"seed {seed} {name:>8}: pearson_delta={df.pearson_delta.mean():.3f}  mse={df.mse_delta.mean():.4f}")
    return pd.concat(frames, ignore_index=True)


def summarize(long: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Mean over perturbations within seed, then mean ± sd across seeds."""
    metrics = [m for m in (metrics or METRICS) if m in long.columns]
    per_seed = long.groupby(["model", "seed"])[metrics].mean().reset_index()
    agg = per_seed.groupby("model")[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.sort_values("pearson_delta_mean", ascending=False)


def compare_to_baseline(long: pd.DataFrame, baseline="mean", metric="pearson_delta") -> pd.DataFrame:
    """Per-seed paired difference vs a baseline (positive = better for correlation-type metrics)."""
    ps = long.groupby(["model", "seed"])[metric].mean().unstack("model")
    diff = ps.sub(ps[baseline], axis=0).drop(columns=baseline)
    return pd.DataFrame({"mean_diff": diff.mean(), "sd_diff": diff.std(), "n_seeds": diff.count(),
                         "seeds_better": (diff > 0).sum()})


def evaluate_on_split(factories: dict[str, Callable], X, Y, index, train, test, ctrl_mean=None, top_k=20) -> pd.DataFrame:
    """Evaluate models on an externally defined split (e.g. the split used by GEARS)."""
    index = np.asarray(index).astype(str)
    pos = {p: i for i, p in enumerate(index)}
    assert_no_leakage(train, test)
    tr = [pos[p] for p in train if p in pos]; te = [pos[p] for p in test if p in pos]
    frames = []
    for name, make in factories.items():
        m = make().fit(X[tr], Y[tr])
        df = per_perturbation_metrics(m.predict(X[te]), Y[te], ctrl_mean, top_k, index=index[te])
        df.index.name = "perturbation"
        frames.append(df.reset_index().assign(model=name, seed=-1))
    return pd.concat(frames, ignore_index=True)
