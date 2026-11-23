"""Download a dataset to the data root (Drive on Colab, ~/PerturbAI_data locally, or $PERTURBAI_DATA).
Usage: python scripts/download_data.py replogle_k562_essential"""
import argparse, json
from perturbai.paths import get_paths
from perturbai.data.loaders import load_dataset, detect_columns, audit_adata, save_audit

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--no-audit", action="store_true")
a = ap.parse_args()
P = get_paths()
adata = load_dataset(a.dataset, P, backed=True)
print(adata)
if not a.no_audit:
    c = detect_columns(adata)
    out = save_audit(audit_adata(adata, c["pert_col"], c["control_label"]), a.dataset, P)
    print("audit ->", out)
