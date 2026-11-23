"""Leakage-safe splitting. For unseen-perturbation tasks ALWAYS split at the perturbation level."""
from __future__ import annotations

import numpy as np


def split_perturbations(perts, test_frac=0.2, val_frac=0.0, seed=0) -> dict:
    perts = np.array(sorted(set(map(str, perts))))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(perts))
    n_test = max(1, int(round(test_frac * len(perts))))
    n_val = int(round(val_frac * len(perts)))
    test = perts[perm[:n_test]]
    val = perts[perm[n_test:n_test + n_val]]
    train = perts[perm[n_test + n_val:]]
    out = {"train": list(train), "val": list(val), "test": list(test)}
    assert_no_leakage(out["train"], out["test"], out["val"])
    return out


def kfold_perturbation_splits(perts, k=5, seed=0):
    perts = np.array(sorted(set(map(str, perts))))
    perm = np.random.default_rng(seed).permutation(len(perts))
    folds = np.array_split(perts[perm], k)
    for i in range(k):
        test = list(folds[i])
        train = [p for j, f in enumerate(folds) if j != i for p in f]
        assert_no_leakage(train, test)
        yield {"train": train, "val": [], "test": test}


def assert_no_leakage(train, test, val=()):
    tr, te, va = set(train), set(test), set(val)
    if tr & te or tr & va or va & te:
        raise AssertionError(f"Perturbation leakage: {sorted((tr & te) | (tr & va) | (va & te))[:5]}")


def split_cells_within_perturbation(labels, test_frac=0.3, seed=0) -> np.ndarray:
    """Scenario 1 (seen perturbations, unseen cells). Returns boolean test mask."""
    labels = np.asarray(labels).astype(str)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), bool)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        n_test = max(1, int(round(test_frac * len(idx)))) if len(idx) > 1 else 0
        mask[rng.choice(idx, n_test, replace=False)] = True
    return mask


def split_combinations(combos, singles_available, test_frac=0.3, seed=0, sep="+"):
    """Scenario 3: hold out double perturbations whose constituents are both seen as singles."""
    combos = sorted(set(combos))
    eligible = [c for c in combos if all(g in set(singles_available) for g in c.split(sep))]
    rng = np.random.default_rng(seed)
    test = list(rng.choice(eligible, max(1, int(round(test_frac * len(eligible)))), replace=False)) if eligible else []
    train = [c for c in combos if c not in set(test)]
    return {"train": train, "test": test}
