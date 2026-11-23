"""Environment-aware path resolution.

Resolution order for the data root:
  1. env var PERTURBAI_DATA   (local machines, servers, CI)
  2. Google Colab -> mounts Drive, uses MyDrive/<PERTURBAI_DRIVE_FOLDER or 'PerturbAI_data'>
  3. ~/PerturbAI_data
Notebooks never hardcode a path; they call get_paths().
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class Paths:
    root: Path
    scratch: Path

    @property
    def raw(self) -> Path: return self.root / "raw"
    @property
    def interim(self) -> Path: return self.root / "interim"
    @property
    def processed(self) -> Path: return self.root / "processed"
    @property
    def models(self) -> Path: return self.root / "models"
    @property
    def results(self) -> Path: return self.root / "results"
    @property
    def manifests(self) -> Path: return self.root / "manifests"
    @property
    def figures(self) -> Path: return self.root / "figures"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed, self.models,
                  self.results, self.manifests, self.figures, self.scratch):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _mount_drive() -> Path:
    from google.colab import drive  # type: ignore
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    folder = os.environ.get("PERTURBAI_DRIVE_FOLDER", "PerturbAI_data")
    return Path("/content/drive/MyDrive") / folder


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    env = os.environ.get("PERTURBAI_DATA")
    if env:
        root = Path(env).expanduser()
    elif in_colab():
        root = _mount_drive()
    else:
        root = Path.home() / "PerturbAI_data"
    scratch = Path("/content/scratch") if in_colab() else root / "scratch"
    return Paths(root=root, scratch=scratch).ensure()


# Artifact file names shared by notebooks and the app
ARTIFACTS = {
    "qc_adata": "{ds}_qc.h5ad",                 # interim
    "subset_adata": "{ds}_subset.h5ad",         # processed
    "effects": "{ds}_effects_delta.parquet",    # processed  (perturbations x genes)
    "ctrl_mean": "{ds}_ctrl_mean.parquet",      # processed
    "n_cells": "{ds}_n_cells.csv",              # processed
    "features": "{ds}_features_coexpr.parquet", # processed
    "benchmark": "{ds}_benchmark_long.csv",     # results
    "best_model": "{ds}_best_model.joblib",     # models
    "priority": "{ds}_target_priority.csv",     # results
}


def artifact(name: str, ds: str) -> str:
    return ARTIFACTS[name].format(ds=ds)
