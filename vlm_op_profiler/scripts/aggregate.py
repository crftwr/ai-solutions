"""aggregate.py — Phase 6

Read all trace.jsonl files under a results directory and emit:
  - report.csv  (long-format aggregation)
  - report.md   (human-readable Markdown tables)

Usage:
    python scripts/aggregate.py <results_dir>
    python scripts/aggregate.py <results_dir> --per-run

The results_dir is expected to contain subdirectories of the form
  <model_name>/<run_id>/trace.jsonl

With --per-run, a report.{csv,md} is also written next to each trace.jsonl.
The combined report covers every trace.jsonl found below results_dir and
lands at <results_dir>/report.{csv,md}.

Outputs are deterministic: the same trace.jsonl files always produce
byte-identical report.csv and report.md (rows sorted by descending MACs with
explicit lexicographic tiebreakers).
"""

from __future__ import annotations

import argparse
import json
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

# Used for stable tiebreaking — same MACs → fall back to lexicographic order
# on the grouping columns, so output ordering is deterministic across pandas
# versions / OS locale settings.
SORT_BY = ["macs"] + GROUP_COLS
SORT_ASC = [False] + [True] * len(GROUP_COLS)


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
                "phase": rec.get("phase", "") or "",
                "layer_category": rec.get("layer_category", "") or "other",
                "op": rec.get("op", "") or "",
                "src0_type": rec.get("src0", {}).get("type", "") or "",
                "src1_type": rec.get("src1", {}).get("type", "") or "",
                "dst_type": rec.get("dst", {}).get("type", "") or "",
                "macs": int(rec.get("macs", 0)),
            }


def discover_traces(results_dir: Path) -> list[tuple[Path, str, str]]:
    """Return [(trace_path, model, run_id)] for every trace.jsonl under results_dir.

    A trace at <results_dir>/<model>/<run_id>/trace.jsonl is included; traces
    whose relative path has fewer than two directory components are skipped
    with a warning. Sorted for determinism.
    """
    out: list[tuple[Path, str, str]] = []
    for trace_path in sorted(results_dir.rglob("trace.jsonl")):
        parts = trace_path.relative_to(results_dir).parts
        if len(parts) < 3:
            print(
                f"WARNING: unexpected path structure (need <model>/<run_id>/trace.jsonl): "
                f"{trace_path}",
                file=sys.stderr,
            )
            continue
        out.append((trace_path, parts[0], parts[1]))
    return out


def load_one(trace_path: Path, model: str, run_id: str) -> pd.DataFrame:
    """Load one trace.jsonl into a DataFrame (one row per node execution)."""
    rows = list(read_trace(trace_path, model, run_id))
    if not rows:
        return pd.DataFrame(columns=SCHEMA_COLS)
    df = pd.DataFrame(rows)
    df["calls"] = 1
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (model, run_id, phase, layer_category, op, dtype combo).

    Output is deterministic: rows sorted by descending macs with lexicographic
    tiebreakers on every grouping column.
    """
    if df.empty:
        return pd.DataFrame(columns=SCHEMA_COLS)
    agg = (
        df.groupby(GROUP_COLS, as_index=False)
        .agg(calls=("calls", "sum"), macs=("macs", "sum"))
    )
    agg = agg.sort_values(SORT_BY, ascending=SORT_ASC, kind="mergesort")
    return agg[SCHEMA_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt_pct(value: float, total: float) -> str:
    if total <= 0:
        return "–"
    return f"{100.0 * value / total:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table; empty rows produce a 'no data' note."""
    if not rows:
        return ["_(no data)_", ""]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    return lines


def _section_dtype_combos(df: pd.DataFrame, total: int) -> list[str]:
    """Section 1 — top dtype combinations by MAC count."""
    agg = (
        df.groupby(["src0_type", "src1_type", "dst_type"], as_index=False)
        .agg(macs=("macs", "sum"))
    )
    agg = agg.sort_values(
        ["macs", "src0_type", "src1_type", "dst_type"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).head(10)
    rows = [
        [
            str(r["src0_type"]),
            str(r["src1_type"]),
            str(r["dst_type"]),
            f"{int(r['macs']):,}",
            _fmt_pct(float(r["macs"]), total),
        ]
        for _, r in agg.iterrows()
    ]
    return ["## Top dtype combinations (by MACs)", ""] + _table(
        ["src0", "src1", "dst", "MACs", "% total"], rows
    )


def _section_layer_phase(df: pd.DataFrame, total: int) -> list[str]:
    """Section 2 — MACs by layer category, with one column per phase."""
    if df.empty:
        return ["## MACs by layer category and phase", "", "_(no data)_", ""]
    pivot = (
        df.groupby(["layer_category", "phase"], as_index=False)
        .agg(macs=("macs", "sum"))
        .pivot(index="layer_category", columns="phase", values="macs")
        .fillna(0)
        .astype("int64")
    )
    # Column order: always prefill | decode | vision_encode | …others alpha.
    preferred = [p for p in ("prefill", "decode", "vision_encode") if p in pivot.columns]
    extra = sorted(c for c in pivot.columns if c not in preferred)
    pivot = pivot[preferred + extra]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values(["total", "layer_category"], ascending=[False, True], kind="mergesort")

    header = ["Category"] + list(pivot.columns[:-1]) + ["total", "% grand total"]
    rows: list[list[str]] = []
    for category, row in pivot.iterrows():
        cells = [str(category)]
        for c in pivot.columns[:-1]:
            cells.append(f"{int(row[c]):,}")
        cells.append(f"{int(row['total']):,}")
        cells.append(_fmt_pct(float(row["total"]), total))
        rows.append(cells)
    return ["## MACs by layer category and phase", ""] + _table(header, rows)


def _section_op_breakdown(df: pd.DataFrame, total: int) -> list[str]:
    """Section 3 — all ops, sorted by descending MAC count."""
    agg = (
        df.groupby("op", as_index=False)
        .agg(calls=("calls", "sum"), macs=("macs", "sum"))
    )
    agg = agg.sort_values(
        ["macs", "op"], ascending=[False, True], kind="mergesort"
    )
    rows = [
        [
            str(r["op"]),
            f"{int(r['calls']):,}",
            f"{int(r['macs']):,}",
            _fmt_pct(float(r["macs"]), total),
        ]
        for _, r in agg.iterrows()
    ]
    return ["## Op type breakdown", ""] + _table(
        ["op", "calls", "MACs", "% total"], rows
    )


def _section_run_meta(meta_files: list[Path]) -> list[str]:
    """Section 4 — run metadata summary, one row per discovered run_meta.json."""
    if not meta_files:
        return []
    rows: list[list[str]] = []
    for path in sorted(meta_files):
        try:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read {path}: {exc}", file=sys.stderr)
            continue
        rows.append([
            str(path.parent.parent.name) + "/" + str(path.parent.name),
            (m.get("model_path") or "").split("/")[-1] or "–",
            (m.get("image_path") or "").split("/")[-1] or "–",
            (m.get("image_sha256") or "")[:12] or "–",
            (m.get("llama_commit") or "")[:12] or "–",
            (m.get("prompt") or "").replace("|", "\\|")[:60] or "–",
        ])
    return ["## Run metadata", ""] + _table(
        ["run", "model", "image", "image_sha[:12]", "llama_commit[:12]", "prompt"],
        rows,
    )


def make_markdown(
    df: pd.DataFrame,
    *,
    title: str,
    meta_files: list[Path] | None = None,
) -> str:
    """Build the full per-results report.md."""
    total = int(df["macs"].sum()) if not df.empty else 0
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"**Total MACs:** {total:,}")
    lines.append("")
    lines += _section_dtype_combos(df, total)
    lines += _section_layer_phase(df, total)
    lines += _section_op_breakdown(df, total)
    if meta_files:
        lines += _section_run_meta(meta_files)
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_reports(
    agg: pd.DataFrame,
    out_dir: Path,
    *,
    title: str,
    meta_files: list[Path] | None,
) -> tuple[Path, Path]:
    """Write report.csv + report.md into out_dir; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "report.csv"
    out_md = out_dir / "report.md"
    # lineterminator='\n' keeps the CSV byte-identical across OSes.
    agg.to_csv(out_csv, index=False, lineterminator="\n")
    out_md.write_text(make_markdown(agg, title=title, meta_files=meta_files),
                      encoding="utf-8")
    return out_csv, out_md


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path, help="Root results directory")
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Also write report.{csv,md} next to each trace.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override combined report path (default: <results_dir>/report.csv)",
    )
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        print(f"ERROR: not a directory: {results_dir}", file=sys.stderr)
        return 1

    traces = discover_traces(results_dir)
    if not traces:
        print(f"No trace.jsonl files found under {results_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(traces)} trace.jsonl file(s) under {results_dir}")

    per_run_dfs: list[pd.DataFrame] = []
    meta_files: list[Path] = []
    for trace_path, model, run_id in traces:
        print(f"  loading {trace_path.relative_to(results_dir)}")
        df = load_one(trace_path, model, run_id)
        per_run_dfs.append(df)

        meta_path = trace_path.parent / "run_meta.json"
        if meta_path.is_file():
            meta_files.append(meta_path)

        if args.per_run:
            agg = aggregate(df)
            csv, md = write_reports(
                agg, trace_path.parent,
                title=f"Report — {model}/{run_id}",
                meta_files=[meta_path] if meta_path.is_file() else None,
            )
            print(f"    wrote {csv}, {md}")

    combined = (
        pd.concat(per_run_dfs, ignore_index=True)
        if per_run_dfs and any(not d.empty for d in per_run_dfs)
        else pd.DataFrame(columns=SCHEMA_COLS)
    )
    agg = aggregate(combined)

    if args.out:
        out_csv, out_md = write_reports(
            agg, args.out.parent if args.out.suffix else args.out,
            title=f"Report — {results_dir.name}", meta_files=meta_files,
        )
    else:
        out_csv, out_md = write_reports(
            agg, results_dir,
            title=f"Report — {results_dir.name}", meta_files=meta_files,
        )
    print(f"Combined report: {out_csv}, {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
