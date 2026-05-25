"""Tests for scripts/aggregate.py — schema, totals, determinism."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

import aggregate as agg_mod


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_aggregate_combined_report(fake_results: Path) -> None:
    """Combined report.csv covers every trace and totals match by hand."""
    rc = agg_mod.main([str(fake_results)])
    assert rc == 0

    report = fake_results / "report.csv"
    md = fake_results / "report.md"
    assert report.is_file()
    assert md.is_file()

    df = pd.read_csv(report)
    assert list(df.columns) == agg_mod.SCHEMA_COLS

    # Sum across the three runs: 100+300+200 + 500+400 + 700+900 = 3100
    assert int(df["macs"].sum()) == 3100

    # Both models appear; ordering is descending by macs.
    assert df["macs"].is_monotonic_decreasing
    assert set(df["model"]) == {"ModelA", "ModelB"}

    # Largest single row is ModelB attn_out / FLASH_ATTN_EXT @ 900 MACs.
    top = df.iloc[0]
    assert top["model"] == "ModelB"
    assert top["op"] == "FLASH_ATTN_EXT"
    assert int(top["macs"]) == 900


def test_aggregate_per_run_outputs(fake_results: Path) -> None:
    """--per-run writes one report.{csv,md} next to each trace.jsonl."""
    rc = agg_mod.main([str(fake_results), "--per-run"])
    assert rc == 0
    for run_dir in (
        fake_results / "ModelA" / "run-001",
        fake_results / "ModelA" / "run-002",
        fake_results / "ModelB" / "run-001",
    ):
        assert (run_dir / "report.csv").is_file()
        assert (run_dir / "report.md").is_file()

    # Per-run totals match what we wrote in conftest.
    df = pd.read_csv(fake_results / "ModelA" / "run-001" / "report.csv")
    assert int(df["macs"].sum()) == 600  # 100 + 300 + 200


def test_aggregate_deterministic(fake_results: Path, tmp_path: Path) -> None:
    """Running aggregate twice on the same inputs yields byte-identical files."""
    # First run.
    rc = agg_mod.main([str(fake_results)])
    assert rc == 0
    sha1_csv = _file_sha(fake_results / "report.csv")
    sha1_md = _file_sha(fake_results / "report.md")

    # Mutate filesystem timestamps to confirm we don't depend on mtime ordering.
    for trace in fake_results.rglob("trace.jsonl"):
        trace.touch()

    # Second run — same inputs, expect identical bytes.
    rc = agg_mod.main([str(fake_results)])
    assert rc == 0
    assert _file_sha(fake_results / "report.csv") == sha1_csv
    assert _file_sha(fake_results / "report.md") == sha1_md


def test_aggregate_layer_phase_pivot_includes_vision_encode(
    fake_results: Path,
) -> None:
    """The markdown table widens to include vision_encode when present."""
    rc = agg_mod.main([str(fake_results)])
    assert rc == 0
    md = (fake_results / "report.md").read_text(encoding="utf-8")
    assert "vision_encode" in md
    assert "vision_conv" in md  # the layer category for the CONV_2D node


def test_aggregate_run_meta_section(fake_results: Path) -> None:
    """run_meta.json content lands in the Run metadata table."""
    rc = agg_mod.main([str(fake_results)])
    assert rc == 0
    md = (fake_results / "report.md").read_text(encoding="utf-8")
    assert "Run metadata" in md
    # SHA prefix and llama commit prefix should appear.
    assert "abcdef012345" in md
    assert "deadbeefcafe" in md
    assert "ModelA.gguf" in md


def test_aggregate_handles_empty_input(tmp_path: Path) -> None:
    """When no trace.jsonl exists, exit code is 1 and nothing is written."""
    empty = tmp_path / "empty_results"
    empty.mkdir()
    rc = agg_mod.main([str(empty)])
    assert rc == 1
    assert not (empty / "report.csv").exists()
    assert not (empty / "report.md").exists()
