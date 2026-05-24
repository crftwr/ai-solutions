"""aggregate.py — Phase 6

Read all trace.jsonl files under a results directory and emit:
  - report.csv  (long-format aggregation)
  - report.md   (human-readable Markdown tables)

Usage:
    python scripts/aggregate.py <results_dir>

The results_dir is expected to contain subdirectories of the form
  <model_name>/<run_id>/trace.jsonl

report.csv and report.md are written next to each trace.jsonl in
<model_name>/<run_id>/ as well as a combined one at <results_dir>/report.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
SCHEMA_COLS = [
    "model",
    "run_id",
    "phase",
    "layer_category",
    "op",
    "src0_type",
    "src1_type",
    "dst_type",
    "calls",
    "macs",
]

GROUP_COLS = [
    "model",
    "run_id",
    "phase",
    "layer_category",
    "op",
    "src0_type",
    "src1_type",
    "dst_type",
]


# ---------------------------------------------------------------------------
# Trace reading
# ---------------------------------------------------------------------------
def read_trace(path: Path, model: str, run_id: str) -> Iterator[dict]:
    """Yield one record per line of a trace.jsonl file."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: {path}:{lineno}: JSON parse error — {exc}",
                    file=sys.stderr,
                )
                continue
            yield {
                "model": model,
                "run_id": run_id,
                "phase": rec.get("phase", ""),
                "layer_category": rec.get("layer_category", "other"),
                "op": rec.get("op", ""),
                "src0_type": rec.get("src0", {}).get("type", ""),
                "src1_type": rec.get("src1", {}).get("type", ""),
                "dst_type": rec.get("dst", {}).get("type", ""),
                "macs": int(rec.get("macs", 0)),
            }


def load_results(results_dir: Path) -> pd.DataFrame:
    """Walk results_dir and load all trace.jsonl files into a DataFrame."""
    rows: list[dict] = []
    for trace_path in sorted(results_dir.rglob("trace.jsonl")):
        parts = trace_path.relative_to(results_dir).parts
        if len(parts) < 3:
            print(
                f"WARNING: unexpected path structure: {trace_path}",
                file=sys.stderr,
            )
            continue
        model, run_id = parts[0], parts[1]
        print(f"  loading {trace_path.relative_to(results_dir)}")
        rows.extend(read_trace(trace_path, model, run_id))

    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)

    df = pd.DataFrame(rows)
    df["calls"] = 1  # each row is one node execution
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (model, run_id, phase, layer_category, op, dtype combo)."""
    if df.empty:
        return pd.DataFrame(columns=SCHEMA_COLS)
    agg = (
        df.groupby(GROUP_COLS, as_index=False)
        .agg(calls=("calls", "sum"), macs=("macs", "sum"))
        .sort_values("macs", ascending=False)
    )
    return agg[SCHEMA_COLS]


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _pct(value: int, total: int) -> str:
    if total == 0:
        return "–"
    return f"{100 * value / total:.1f}%"


def make_markdown(df: pd.DataFrame, title: str = "") -> str:
    """Generate a Markdown report from an aggregated DataFrame."""
    lines: list[str] = []
    if title:
        lines += [f"# {title}", ""]

    total_macs = df["macs"].sum()
    lines += [
        f"**Total MACs:** {total_macs:,}",
        "",
    ]

    # Top-10 dtype combinations by MAC count
    dtype_agg = (
        df.groupby(["src0_type", "src1_type", "dst_type"], as_index=False)
        .agg(macs=("macs", "sum"))
        .sort_values("macs", ascending=False)
        .head(10)
    )
    lines += ["## Top dtype combinations (by MACs)", ""]
    lines += ["| src0 | src1 | dst | MACs | % total |", "|------|------|-----|------|---------|"]
    for _, row in dtype_agg.iterrows():
        lines.append(
            f"| {row['src0_type']} | {row['src1_type']} | {row['dst_type']} "
            f"| {int(row['macs']):,} | {_pct(int(row['macs']), total_macs)} |"
        )
    lines += [""]

    # MACs by layer category
    layer_agg = (
        df.groupby(["layer_category", "phase"], as_index=False)
        .agg(macs=("macs", "sum"))
        .sort_values(["layer_category", "phase"])
    )
    lines += ["## MACs by layer category and phase", ""]
    lines += ["| Category | Phase | MACs | % total |", "|----------|-------|------|---------|"]
    for _, row in layer_agg.iterrows():
        lines.append(
            f"| {row['layer_category']} | {row['phase']} "
            f"| {int(row['macs']):,} | {_pct(int(row['macs']), total_macs)} |"
        )
    lines += [""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Root results directory")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path (default: <results_dir>/report.csv)",
    )
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: not a directory: {results_dir}", file=sys.stderr)
        return 1

    print(f"Loading traces from {results_dir} …")
    df = load_results(results_dir)
    if df.empty:
        print("No trace.jsonl files found.", file=sys.stderr)
        return 1

    agg = aggregate(df)

    out_csv = args.out or (results_dir / "report.csv")
    agg.to_csv(out_csv, index=False)
    print(f"Written: {out_csv}")

    out_md = out_csv.with_suffix(".md")
    md = make_markdown(agg, title=f"Report — {results_dir.name}")
    out_md.write_text(md, encoding="utf-8")
    print(f"Written: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
