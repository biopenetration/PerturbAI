from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def file_hash(path: str | Path, algo: str = "md5", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def memory_report() -> dict:
    import psutil
    vm = psutil.virtual_memory()
    return {"total_gb": round(vm.total / 1e9, 1), "available_gb": round(vm.available / 1e9, 1),
            "cpu_count": os.cpu_count()}


def run_id(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{git_hash()}"


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_config(name: str) -> dict:
    import yaml
    cfg_dir = Path(__file__).resolve().parents[2] / "configs"
    with open(cfg_dir / name) as f:
        return yaml.safe_load(f)
