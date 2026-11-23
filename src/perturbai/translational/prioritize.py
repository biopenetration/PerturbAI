"""Candidate-target prioritisation from predicted effects + external evidence.

OUTPUT IS A RANKED HYPOTHESIS LIST, NOT VALIDATED DRUG TARGETS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from perturbai.features.effects import effect_size_summary

CAVEAT = ("Computational hypothesis only: requires experimental, pharmacological and safety validation. "
          "Essentiality in one cell line does not imply a therapeutic window.")


def load_depmap_gene_effect(effect_csv, model_csv=None, cell_line: str | None = "K562") -> pd.DataFrame:
    """DepMap CRISPRGeneEffect.csv (rows=models, cols='GENE (ID)'). Returns mean effect across all lines
    and, if available, the effect in `cell_line` (more negative = more essential)."""
    ge = pd.read_csv(effect_csv, index_col=0)
    ge.columns = [c.split(" (")[0] for c in ge.columns]
    out = pd.DataFrame({"depmap_mean_effect": ge.mean(0)})
    if model_csv and cell_line:
        m = pd.read_csv(model_csv)
        col = "StrippedCellLineName" if "StrippedCellLineName" in m else m.columns[0]
        ids = m.loc[m[col].astype(str).str.upper() == cell_line.upper(), "ModelID"].tolist()
        ids = [i for i in ids if i in ge.index]
        if ids:
            out[f"depmap_{cell_line}_effect"] = ge.loc[ids[0]]
    return out


def _rank01(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) if s.notna().any() else s


def prioritize(pred_delta: pd.DataFrame, evidence: pd.DataFrame | None = None, weights: dict | None = None,
               uncertainty: pd.Series | None = None, thr: float = 0.1) -> pd.DataFrame:
    """Composite score = weighted mean of rank-normalised components. Evidence columns (optional):
       depmap_mean_effect (neg = essential), is_cancer_gene (0/1), druggability (0-1), disease_assoc (0-1)."""
    S = effect_size_summary(pred_delta, thr)
    comp = pd.DataFrame(index=S.index)
    comp["effect_size"] = _rank01(S["l2_norm"])
    if evidence is not None:
        ev = evidence.reindex(S.index)
        if "depmap_mean_effect" in ev:
            comp["essentiality"] = _rank01(-ev["depmap_mean_effect"])
        for c in ("is_cancer_gene", "druggability", "disease_assoc"):
            if c in ev:
                comp[c] = _rank01(ev[c].astype(float))
    w = weights or {c: 1.0 for c in comp.columns}
    w = {k: v for k, v in w.items() if k in comp.columns}
    score = sum(comp[k] * v for k, v in w.items()) / sum(w.values())
    out = S.join(comp.add_prefix("score_"))
    if evidence is not None:
        out = out.join(evidence.reindex(S.index))
    out["composite"] = score
    if uncertainty is not None:
        out["uncertainty"] = uncertainty.reindex(S.index)
        out["composite_adj"] = out["composite"] * (1 - _rank01(out["uncertainty"]).fillna(0) * 0.5)
    out["caveat"] = CAVEAT
    return out.sort_values("composite_adj" if "composite_adj" in out else "composite", ascending=False)
