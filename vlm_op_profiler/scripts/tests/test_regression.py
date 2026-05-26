"""Regression invariants on the real Llama-3.2-1B text smoke-test trace.

Runs against /app/results/smoke-test-text/trace.jsonl when present. Skips
the whole module otherwise so this file works inside the Docker image's
unit-test pass even without the model fetched.

`make regression-test` populates the trace by invoking the text smoke-test
first, then runs `pytest scripts/tests/test_regression.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from validate import ARCH_SPECS, validate_one


TRACE_PATH = Path(
    os.environ.get("REGRESSION_TRACE", "/app/results/smoke-test-text/trace.jsonl")
)

pytestmark = pytest.mark.skipif(
    not TRACE_PATH.is_file(),
    reason=f"regression trace not found at {TRACE_PATH}; run `make regression-test`",
)


def test_regression_invariants_pass() -> None:
    """All structural + analytical checks pass for Llama-3.2-1B."""
    rep = validate_one(TRACE_PATH, ARCH_SPECS["llama-3.2-1b"])
    assert rep.ok, "\n".join(f"{f.check}: {f.detail}" for f in rep.failed)


def test_regression_trace_size_lower_bound() -> None:
    """Smoke-test-text must produce a meaningful trace size.

    The exact line count drifts with upstream llama.cpp graph construction,
    so we only assert a lower bound that catches catastrophic regressions
    (e.g. interceptor not firing → ~0 lines).
    """
    n = sum(1 for _ in TRACE_PATH.open("r", encoding="utf-8"))
    assert n >= 1000, f"only {n} trace records — interceptor may not have fired"


def test_regression_both_phases_present() -> None:
    """At least one prefill record and one decode record must exist."""
    import json
    phases: set[str] = set()
    with TRACE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            phases.add(json.loads(line).get("phase", ""))
            if {"prefill", "decode"} <= phases:
                return
    pytest.fail(f"missing phases — saw {sorted(phases)}, want both prefill and decode")
