"""Tests for scripts/summarize.py — cross-model rollups & determinism."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

import summarize as sum_mod


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary_csv_schema_and_totals(
    fake_report_csvs: list[Path], tmp_path: Path
) -> None:
    """summary.csv has the documented schema and pct_of_total sums to ~100%."""
    out = tmp_path / "summary"
    rc = sum_mod.main([*map(str, fake_report_csvs), "--out", str(out)])
    assert rc == 0

    df = pd.read_csv(out / "summary.csv")
    assert list(df.columns) == sum_mod.SUMMARY_COLS

    # 300 + 200 + 700 + 900 = 2100 grand total → pct sums close to 100%
    assert int(df["macs"].sum()) == 2100
    assert abs(df["pct_of_total"].sum() - 100.0) < 0.05

    # First row is the largest MAC group: ModelB FLASH_ATTN_EXT @ 900.
    assert df.iloc[0]["model"] == "ModelB"
    assert int(df.iloc[0]["macs"]) == 900


def test_summary_md_present(
    fake_report_csvs: list[Path], tmp_path: Path
) -> None:
    """summary.md has both top-dtype and per-model totals tables."""
    out = tmp_path / "summary"
    rc = sum_mod.main([*map(str, fake_report_csvs), "--out", str(out)])
    assert rc == 0
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "Top dtype combinations" in md
    assert "Per-model MAC totals" in md
    assert "ModelA" in md and "ModelB" in md


def test_summary_deterministic(
    fake_report_csvs: list[Path], tmp_path: Path
) -> None:
    """Same inputs ⇒ byte-identical summary.csv and summary.md."""
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    rc = sum_mod.main([*map(str, fake_report_csvs), "--out", str(out1)])
    assert rc == 0
    rc = sum_mod.main([*map(str, fake_report_csvs), "--out", str(out2)])
    assert rc == 0
    assert _file_sha(out1 / "summary.csv") == _file_sha(out2 / "summary.csv")
    assert _file_sha(out1 / "summary.md") == _file_sha(out2 / "summary.md")


def test_summary_argument_order_independence(
    fake_report_csvs: list[Path], tmp_path: Path
) -> None:
    """CLI arg ordering must not affect output content."""
    out1 = tmp_path / "ord1"
    out2 = tmp_path / "ord2"
    rc = sum_mod.main([*map(str, fake_report_csvs), "--out", str(out1)])
    assert rc == 0
    rc = sum_mod.main([*map(str, reversed(fake_report_csvs)), "--out", str(out2)])
    assert rc == 0
    assert _file_sha(out1 / "summary.csv") == _file_sha(out2 / "summary.csv")
    assert _file_sha(out1 / "summary.md") == _file_sha(out2 / "summary.md")
