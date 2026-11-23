"""FastAPI prediction service.  Run: uvicorn app.api:app --port 8000
GET /perturbations           -> genes that can be encoded
GET /predict?gene=G&top=20   -> predicted top up/down genes"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from perturbai.paths import artifact, get_paths

TAG = os.environ.get("PERTURBAI_TAG", "replogle_k562_essential")
P = get_paths()
app = FastAPI(title="PerturbAI API", version="0.1.0")
_B = None


def bundle():
    global _B
    if _B is None:
        f = P.models / artifact("best_model", TAG)
        if not f.exists():
            raise HTTPException(503, f"model artifact missing: {f.name}")
        _B = joblib.load(f)
    return _B


@app.get("/health")
def health():
    return {"status": "ok", "model_available": (P.models / artifact("best_model", TAG)).exists()}


@app.get("/perturbations")
def perturbations():
    B = bundle(); return {"genes": sorted(B["feature_table"].index), "measured": B["trained_on"]}


@app.get("/predict")
def predict(gene: str, top: int = 20):
    B = bundle()
    if gene not in B["feature_table"].index:
        raise HTTPException(404, f"gene {gene} has no features")
    y = pd.Series(B["model"].predict(B["feature_table"].loc[[gene]].values)[0], index=B["genes"]).sort_values()
    return {"gene": gene, "model": B["name"], "seen_in_training": gene in set(B["trained_on"]),
            "down": y.head(top).round(4).to_dict(), "up": y.tail(top)[::-1].round(4).to_dict(),
            "disclaimer": "Computational prediction; not experimental evidence."}
