import numpy as np
import pandas as pd

from perturbai.data.synthetic import make_synthetic_perturbseq
from perturbai.evaluation.benchmark import compare_to_baseline, run_benchmark, summarize
from perturbai.evaluation.metrics import per_perturbation_metrics, rowwise_pearson
from perturbai.evaluation.splits import (assert_no_leakage, split_cells_within_perturbation,
                                         split_combinations, split_perturbations)
from perturbai.features.de import bh_fdr, pseudobulk_de
from perturbai.features.effects import (compute_effects, effect_size_summary, knockdown_efficiency,
                                        split_half_effects)
from perturbai.features.embeddings import coexpression_embedding, perturbation_feature_matrix
from perturbai.features.graph import knn_graph, propagate
from perturbai.models.baselines import (ControlBaseline, KNNModel, MeanBaseline, RandomForestModel,
                                        RidgeModel, predict_additive)
from perturbai.preprocessing.qc import normalize_log1p_matrix
from perturbai.translational.prioritize import prioritize


def _data():
    X, obs, var, truth = make_synthetic_perturbseq(seed=1)
    Xn, Xl = normalize_log1p_matrix(X)
    return X, Xn, Xl, obs, var, truth


def test_split_no_leakage():
    perts = [f"G{i}" for i in range(50)]
    sp = split_perturbations(perts, 0.2, 0.1, seed=3)
    assert_no_leakage(sp["train"], sp["test"], sp["val"])
    assert len(sp["test"]) == 10 and len(set(sp["train"]) | set(sp["test"]) | set(sp["val"])) == 50
    try:
        assert_no_leakage(["a"], ["a"])
        raise RuntimeError("leak not detected")
    except AssertionError:
        pass


def test_cell_split_and_combo_split():
    labels = np.array(["a"] * 10 + ["b"] * 10)
    m = split_cells_within_perturbation(labels, 0.3, 0)
    assert m[:10].sum() == 3 and m[10:].sum() == 3
    sp = split_combinations(["A+B", "A+C", "C+D"], ["A", "B", "C"], 0.5, 0)
    assert all(c != "C+D" for c in sp["test"])


def test_effects_and_knockdown():
    X, Xn, Xl, obs, var, truth = _data()
    res = compute_effects(Xl, obs.perturbation.values, var.index, min_cells=20)
    assert res.delta.shape[1] == var.shape[0] and len(res.delta) == 60
    resb = compute_effects(Xl, obs.perturbation.values, var.index, batch=obs.gem_group.values, min_cells=20)
    assert resb.delta.shape == res.delta.shape
    kd = knockdown_efficiency(Xn, obs.perturbation.values, var.index)
    assert kd.knockdown[kd.assessable].median() > 0.5
    # perturbation lowers its own gene
    g = res.delta.index[0]
    assert res.delta.loc[g, g] < 0
    a, b = split_half_effects(Xl, obs.perturbation.values, var.index, seed=0, min_cells=20)
    r = rowwise_pearson(a.values, b.values)
    assert np.nanmean(r) > 0.5, "split-half reliability should be high on synthetic data"
    assert effect_size_summary(res.delta).shape[0] == len(res.delta)


def test_de_pseudobulk():
    X, Xn, Xl, obs, var, truth = _data()
    p = obs.perturbation[obs.perturbation != "control"].iloc[0]
    de = pseudobulk_de(Xl, obs.perturbation.values, var.index, p, batch=obs.gem_group.values, min_cells_rep=3)
    assert de.loc[p, "log_fc"] < 0 and de.loc[p, "padj"] < 0.05
    pv = np.random.default_rng(0).uniform(size=100)
    assert (bh_fdr(pv) >= pv - 1e-12).all()


def test_metrics_basic():
    t = np.random.default_rng(0).normal(size=(5, 100))
    m = per_perturbation_metrics(t, t)
    assert np.allclose(m.pearson_delta, 1) and np.allclose(m.mse_delta, 0) and np.allclose(m.direction_acc_top_deg, 1)
    z = per_perturbation_metrics(np.zeros_like(t), t)
    assert (z.direction_acc_top_deg == 0).all()


def test_models_beat_mean_on_unseen_perturbations():
    X, Xn, Xl, obs, var, truth = _data()
    res = compute_effects(Xl, obs.perturbation.values, var.index, min_cells=20)
    ctrl = obs.perturbation.values == "control"
    emb = coexpression_embedding(Xl[ctrl], var.index, n_components=16)
    F, missing = perturbation_feature_matrix(res.delta.index, emb)
    assert not missing
    Y = res.delta.loc[F.index].values
    long = run_benchmark({"control": ControlBaseline, "mean": MeanBaseline, "ridge": lambda: RidgeModel(10.0),
                          "knn": lambda: KNNModel(3), "rf": lambda: RandomForestModel(50, 10)},
                         F.values, Y, F.index, res.ctrl_mean.values, seeds=(0, 1, 2), verbose=False)
    s = summarize(long)
    cmp = compare_to_baseline(long, "mean")
    print(s[["pearson_delta_mean", "mse_delta_mean"]])
    print(cmp)
    assert s.loc["knn", "pearson_delta_mean"] > s.loc["mean", "pearson_delta_mean"] + 0.05
    assert s.loc["ridge", "pearson_delta_mean"] > s.loc["mean", "pearson_delta_mean"]


def test_graph_and_additive_and_prioritize():
    X, Xn, Xl, obs, var, truth = _data()
    emb = np.random.default_rng(0).normal(size=(50, 8))
    A = knn_graph(emb, 5)
    assert (A != A.T).nnz == 0
    Z = propagate(A, emb, 2, 0.5)
    assert Z.shape == emb.shape
    single = pd.DataFrame(np.random.default_rng(0).normal(size=(3, 6)), index=["A", "B", "C"])
    out = predict_additive(single, ["A+B", "A+Z"])
    assert list(out.index) == ["A+B"] and np.allclose(out.loc["A+B"], single.loc["A"] + single.loc["B"])
    ev = pd.DataFrame({"depmap_mean_effect": np.linspace(-1, 0, 3), "druggability": [0.2, 0.9, 0.5]},
                      index=["A", "B", "C"])
    pr = prioritize(single, ev)
    assert "caveat" in pr and len(pr) == 3
