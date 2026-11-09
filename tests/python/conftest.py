"""Session setup for the Python tests.

- The kernel cache (compiled code on disk) goes to a temporary directory for the session, unless MINIDLC_CACHE_DIR is
  set: tests stay hermetic and leave nothing in ~/.cache.
- MINIDLC_REQUIRE_COMPILED=1 (set by CI through ctest) turns a missing `_minidlc` module or `dlc-opt` into an error
  instead of skipped tests, so a broken build cannot pass as "all skipped".
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

_CACHE_DIR = None
if not os.environ.get("MINIDLC_CACHE_DIR"):
    _CACHE_DIR = tempfile.mkdtemp(prefix="minidlc-test-cache-")
    os.environ["MINIDLC_CACHE_DIR"] = _CACHE_DIR


def pytest_sessionstart(session):
    if os.environ.get("MINIDLC_REQUIRE_COMPILED") != "1":
        return
    try:
        import minidlc._minidlc  # noqa: F401
    except ImportError as e:
        raise pytest.UsageError(f"MINIDLC_REQUIRE_COMPILED=1 but the _minidlc module does not import: {e}")
    dlc_opt = os.environ.get("DLC_OPT", os.path.join(os.path.dirname(__file__), "..", "..", "build", "dlc-opt"))
    if not os.path.exists(dlc_opt):
        raise pytest.UsageError(f"MINIDLC_REQUIRE_COMPILED=1 but dlc-opt is not at {dlc_opt}")


def pytest_sessionfinish(session, exitstatus):
    if _CACHE_DIR:
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)
