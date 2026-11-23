"""Fallback runner when pytest is unavailable: python scripts/run_tests_nopytest.py"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import test_core
fails = 0
for n in sorted(dir(test_core)):
    if n.startswith("test_"):
        try:
            getattr(test_core, n)(); print("PASS", n)
        except Exception:
            fails += 1; print("FAIL", n); traceback.print_exc()
sys.exit(1 if fails else 0)
