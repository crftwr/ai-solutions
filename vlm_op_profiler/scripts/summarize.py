"""summarize.py — Phase 6

Read multiple report.csv files and produce a cross-model executive summary:
- summary.csv: percentage of total MACs per (src0_dtype × src1_dtype → dst_dtype)
               combination, stacked by model.
- summary.md:  human-readable tables.

Usage:
    python scripts/summarize.py results/*/report.csv
    python scripts/summarize.py --out summary/ results/*/report.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files", nargs="+", type=Path,
        help="One or more report.csv files to summarise",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("."),
        help="Output directory (default: current directory)",
    )
    args = parser.parse_args(argv)

    # Load and concatenate
    frames: list[pd.DataFrame] = []
    for p in sorted(args.csv_files):
        if not p.is_file():
            print(f"WARNING: not a file, skipping: {p}", file=sys.stderr)
            continue
        df = pd.read_csv(p)
        frames.append(df)
        print(f"  loaded {p} ({len(df)} rows)")

    if not frames:
        print("No valid CSV files found.", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True)
    total_macs = combined["macs"].sum()

    # Cross-model dtype summary
    dtype_agg = (
        combined
        .groupby(["model", "src0_type", "src1_type", "dst_type"], as_index=False)
        .agg(macs=("macs", "sum"))
    )
    dtype_agg["pct_of_total"] = (dtype_agg["macs"] / total_macs * 100).round(2)
    dtype_agg = dtype_agg.sort_values("macs", ascending=False)

    # Write summary.csv
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "summary.csv"
    dtype_agg.to_csv(out_csv, index=False)
    print(f"Written: {out_csv}")

    # Write summary.md
    out_md = args.out / "summary.md"
    lines = [
        "# Cross-model MAC summary",
        "",
        f"**Total MACs (all models):** {total_macs:,}",
        "",
        "## Top dtype combinations",
        "",
        "| Model | src0 | src1 | dst | MACs | % of total |",
        "|-------|------|------|-----|------|------------|",
    ]
    for _, row in dtype_agg.head(20).iterrows():
        lines.append(
            f"| {row['model']} | {row['src0_type']} | {row['src1_type']} "
            f"| {row['dst_type']} | {int(row['macs']):,} | {row['pct_of_total']:.2f}% |"
        )
    lines += [""]

    # Per-model totals
    model_totals = (
        combined.groupby("model")["macs"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    lines += [
        "## Per-model MAC totals",
        "",
        "| Model | Total MACs | % of grand total |",
        "|-------|------------|-----------------|",
    ]
    for _, row in model_totals.iterrows():
        pct = 100 * row["macs"] / total_macs if total_macs > 0 else 0.0
        lines.append(f"| {row['model']} | {int(row['macs']):,} | {pct:.1f}% |")
    lines += [""]

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
