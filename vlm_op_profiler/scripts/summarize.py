"""summarize.py — Phase 6

Read multiple report.csv files and produce a cross-model executive summary:
  - summary.csv  per-model (src0_dtype × src1_dtype → dst_dtype) breakdown
                 with % of grand-total MACs.
  - summary.md   human-readable tables.

Usage:
    python scripts/summarize.py results/*/report.csv
    python scripts/summarize.py --out summary/ results/*/report.csv

Outputs are deterministic: given the same input CSVs, summary.csv and
summary.md are byte-identical across runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SUMMARY_COLS = ["model", "src0_type", "src1_type", "dst_type", "macs", "pct_of_total"]


def load_inputs(paths: list[Path]) -> pd.DataFrame:
    """Concatenate every readable report.csv into one DataFrame.

    Missing files are warned about, not fatal — partial summaries are
    legitimate when some models have not been run yet.
    """
    frames: list[pd.DataFrame] = []
    for p in sorted(paths):
        if not p.is_file():
            print(f"WARNING: not a file, skipping: {p}", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        frames.append(df)
        print(f"  loaded {p} ({len(df)} rows)")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def per_model_dtype_breakdown(combined: pd.DataFrame) -> pd.DataFrame:
    """Group by (model, dtype combo); pct_of_total over the grand total."""
    if combined.empty:
        return pd.DataFrame(columns=SUMMARY_COLS)
    total = float(combined["macs"].sum())
    agg = (
        combined.groupby(
            ["model", "src0_type", "src1_type", "dst_type"], as_index=False
        )
        .agg(macs=("macs", "sum"))
    )
    agg["pct_of_total"] = (
        (agg["macs"] / total * 100.0).round(2) if total > 0 else 0.0
    )
    agg = agg.sort_values(
        ["macs", "model", "src0_type", "src1_type", "dst_type"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return agg[SUMMARY_COLS]


def per_model_totals(combined: pd.DataFrame) -> pd.DataFrame:
    """One row per model: (model, macs, pct_of_total)."""
    if combined.empty:
        return pd.DataFrame(columns=["model", "macs", "pct_of_total"])
    total = float(combined["macs"].sum())
    out = (
        combined.groupby("model", as_index=False)["macs"]
        .sum()
        .sort_values(
            ["macs", "model"], ascending=[False, True], kind="mergesort"
        )
        .reset_index(drop=True)
    )
    out["pct_of_total"] = (
        (out["macs"] / total * 100.0).round(2) if total > 0 else 0.0
    )
    return out


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_(no data)_", ""]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    return lines


def make_markdown(
    dtype_agg: pd.DataFrame, model_totals: pd.DataFrame, total: int
) -> str:
    """Human-readable cross-model summary."""
    lines: list[str] = ["# Cross-model MAC summary", ""]
    lines.append(f"**Total MACs (all models):** {total:,}")
    lines.append("")

    lines += ["## Top dtype combinations (top 20)", ""]
    rows = [
        [
            str(r["model"]),
            str(r["src0_type"]),
            str(r["src1_type"]),
            str(r["dst_type"]),
            f"{int(r['macs']):,}",
            f"{float(r['pct_of_total']):.2f}%",
        ]
        for _, r in dtype_agg.head(20).iterrows()
    ]
    lines += _table(
        ["Model", "src0", "src1", "dst", "MACs", "% of total"], rows
    )

    lines += ["## Per-model MAC totals", ""]
    rows = [
        [
            str(r["model"]),
            f"{int(r['macs']):,}",
            f"{float(r['pct_of_total']):.2f}%",
        ]
        for _, r in model_totals.iterrows()
    ]
    lines += _table(["Model", "Total MACs", "% of grand total"], rows)

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "csv_files", nargs="+", type=Path,
        help="One or more report.csv files to summarise",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("."),
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args(argv)

    combined = load_inputs(args.csv_files)
    if combined.empty:
        print("No usable input rows.", file=sys.stderr)
        return 1

    dtype_agg = per_model_dtype_breakdown(combined)
    totals = per_model_totals(combined)
    total_macs = int(combined["macs"].sum())

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "summary.csv"
    out_md = args.out / "summary.md"

    dtype_agg.to_csv(out_csv, index=False, lineterminator="\n")
    out_md.write_text(make_markdown(dtype_agg, totals, total_macs),
                      encoding="utf-8")
    print(f"Written: {out_csv}")
    print(f"Written: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
