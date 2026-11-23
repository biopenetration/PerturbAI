"""Glue: load the aligned (features, delta) modelling task produced by notebooks 05 and 08."""
from __future__ import annotations

import numpy as np
import pandas as pd

from perturbai.features.embeddings import perturbation_feature_matrix
from perturbai.paths import Paths, artifact


def load_task(P: Paths, tag: str, feature_prefixes=("coexp_",)) -> dict:
    delta = pd.read_parquet(P.processed / artifact("effects", tag))
    ctrl = pd.read_parquet(P.processed / artifact("ctrl_mean", tag))["ctrl_mean"]
    feats = pd.read_parquet(P.processed / artifact("features", tag))
    cols = [c for c in feats.columns if c.startswith(tuple(feature_prefixes))]
    F, missing = perturbation_feature_matrix(delta.index, feats[cols])
    idx = sorted(F.index)                       # deterministic order => identical splits in every notebook
    Y = delta.loc[idx]
    return {"X": F.loc[idx].values.astype(np.float64), "Y": Y.values.astype(np.float64),
            "index": np.array(idx), "ctrl_mean": ctrl.loc[Y.columns].values,
            "genes": Y.columns, "feats": feats[cols], "missing": missing}
