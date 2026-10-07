"""Canonical filesystem locations, anchored to the repository root.

Every script that reads or writes ``data/`` or ``results/`` resolves its
paths through here instead of hardcoding cwd-relative literals such as
``"data/benchmark_data.parquet"``. Because the root is derived from this
module's own ``__file__`` (not the current working directory), the target
location is identical no matter which directory a bench or generator is
launched from.

This is a safety property, not a cosmetic one: a wrong-cwd launch of a
cwd-relative generator silently writes a whole parallel ``data/`` tree
(e.g. ``slides/data/``) instead of the canonical one.
That is exactly how a 24 GB nested duplicate accumulated invisibly and was
later swept in a disk cleanup. Anchoring to ``__file__`` makes that class
of accident impossible.
"""
from pathlib import Path

# src/coxbench/paths.py -> parents[0]=coxbench, [1]=src, [2]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def data_path(name):
    """Absolute path to ``name`` under the canonical ``data/`` directory.

    Creates ``data/`` if it does not exist. ``name`` may contain nested
    components (e.g. ``"survival_parts/part_000.parquet"``).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name


def results_path(name):
    """Absolute path to ``name`` under the canonical ``results/`` directory.

    Creates ``results/`` if it does not exist.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name
