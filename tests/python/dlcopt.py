"""Finds the dlc-opt binary for the Python tests (DLC_OPT, else build/dlc-opt) and runs it."""
from __future__ import annotations

import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
DLC_OPT = os.environ.get("DLC_OPT") or str(ROOT / "build" / "dlc-opt")


def available() -> bool:
    return os.access(DLC_OPT, os.X_OK)


def run(text: str, *flags: str) -> str:
    proc = subprocess.run([DLC_OPT, *flags, "-"], input=text, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(f"dlc-opt {' '.join(flags)} failed:\n{proc.stderr}")
    return proc.stdout
