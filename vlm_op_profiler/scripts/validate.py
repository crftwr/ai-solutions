"""validate.py — Phase 7 cross-checks on a trace.jsonl.

Verifies two classes of invariant for every trace.jsonl found below a results
directory:

1. **Structural** — architecture-agnostic checks that catch bugs in the
   classifier, graph walker, or phase tagger. Each holds for any Llama-style
   transformer (GQA + SwiGLU FFN):

   - For every MUL_MAT row, `macs == 2·m·n·k`. Cheap shape↔MAC consistency
     check; catches accumulator overflow / signed-vs-unsigned bugs.
   - `calls(attn_qkv) == 3 · calls(attn_out)` (Q, K, V projections vs O).
   - `calls(ffn_gate) == calls(ffn_up) == calls(ffn_down)` (SwiGLU).
   - Every decode-phase MUL_MAT has `m == 1`.
   - Every prefill-phase MUL_MAT has `max m > 1` across the graph.

2. **Analytical cross-check** — given an architecture spec
   (n_layers, d_model, n_kv_heads / n_heads, d_ff, vocab_size, lm_head_tied),
   re-derive the per-layer MUL_MAT MAC contribution from the recorded
   shapes and verify it matches the architecture (`2·M·d_model² ` for
   attn_out, `2·M·d_model·d_ff` for FFN, etc.). Catches model-config drift
   and double-counted graphs.

Two known specs are bundled (Llama-3.2-1B-Instruct, SmolVLM-Instruct). For
other models, pass `--arch <key>` with a key defined in ARCH_SPECS or run
without `--arch` to skip section 2.

Exit code 0 iff all enabled checks pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Architecture specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchSpec:
    """Static config we need to derive per-MAC expectations.

    n_kv_heads / n_heads gives the GQA reduction factor on K/V projections;
    when they are equal (MHA) the K/V projections are square just like Q.
    """
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab: int
    lm_head_tied: bool  # if True, lm_head reuses embed weights (no extra MUL_MAT)


# These are the two models actually exercised by `make regression-test`.
# Numbers come from the upstream model config; do not invent values for
# models we have not run end-to-end.
ARCH_SPECS: dict[str, ArchSpec] = {
    "llama-3.2-1b": ArchSpec(
        name="Llama-3.2-1B-Instruct",
        n_layers=16, d_model=2048, n_heads=32, n_kv_heads=8,
        d_ff=8192, vocab=128256, lm_head_tied=True,
    ),
    "smolvlm": ArchSpec(
        name="SmolVLM-Instruct (text body)",
        n_layers=24, d_model=2048, n_heads=32, n_kv_heads=8,
        d_ff=8192, vocab=49152, lm_head_tied=True,
    ),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    """Return every parseable record in one trace.jsonl."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: {path}:{lineno}: {exc}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Results — accumulator type
# ---------------------------------------------------------------------------

@dataclass
class Failure:
    check: str
    detail: str


@dataclass
class Report:
    trace: Path
    passed: list[str]
    failed: list[Failure]

    @property
    def ok(self) -> bool:
        return not self.failed


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def check_mac_formula(records: list[dict]) -> tuple[bool, str]:
    """Every MUL_MAT row must satisfy macs == 2*m*n*k."""
    bad: list[tuple[int, int]] = []
    for r in records:
        if r.get("op") != "MUL_MAT":
            continue
        expected = 2 * int(r["m"]) * int(r["n"]) * int(r["k"])
        if int(r["macs"]) != expected:
            bad.append((r.get("graph_id", -1), r.get("node_idx", -1)))
            if len(bad) > 5:
                break
    if bad:
        return False, f"{len(bad)} MUL_MAT rows where macs != 2·m·n·k (e.g. {bad[:3]})"
    return True, "macs == 2·m·n·k holds for every MUL_MAT row"


def _count_calls(records: list[dict], op: str, layer_category: str) -> int:
    return sum(
        1 for r in records
        if r.get("op") == op and r.get("layer_category") == layer_category
    )


def check_attention_ratio(records: list[dict]) -> tuple[bool, str]:
    """attn_qkv MUL_MAT count == 3 × attn_out MUL_MAT count."""
    qkv = _count_calls(records, "MUL_MAT", "attn_qkv")
    out = _count_calls(records, "MUL_MAT", "attn_out")
    if out == 0 and qkv == 0:
        return True, "no attention MUL_MATs to check (skipped)"
    if qkv != 3 * out:
        return False, f"attn_qkv={qkv} vs 3·attn_out={3 * out}"
    return True, f"attn_qkv ({qkv}) == 3 × attn_out ({out})"


def check_ffn_triplet(records: list[dict]) -> tuple[bool, str]:
    """ffn_gate / ffn_up / ffn_down MUL_MAT counts are equal (SwiGLU FFN)."""
    g = _count_calls(records, "MUL_MAT", "ffn_gate")
    u = _count_calls(records, "MUL_MAT", "ffn_up")
    d = _count_calls(records, "MUL_MAT", "ffn_down")
    if g == u == d == 0:
        return True, "no FFN MUL_MATs to check (skipped)"
    if not (g == u == d):
        return False, f"ffn_gate={g}, ffn_up={u}, ffn_down={d}"
    return True, f"ffn_gate == ffn_up == ffn_down == {g}"


def check_decode_m_is_one(records: list[dict]) -> tuple[bool, str]:
    """Every decode-phase MUL_MAT has m == 1 (single token per decode step)."""
    offenders: list[tuple[int, int, int]] = []
    for r in records:
        if r.get("op") != "MUL_MAT" or r.get("phase") != "decode":
            continue
        if int(r["m"]) != 1:
            offenders.append((r.get("graph_id", -1), r.get("node_idx", -1), int(r["m"])))
            if len(offenders) > 5:
                break
    if offenders:
        return False, f"{len(offenders)} decode MUL_MATs with m > 1 (e.g. {offenders[:3]})"
    return True, "every decode MUL_MAT has m == 1"


def check_prefill_max_m(records: list[dict]) -> tuple[bool, str]:
    """For each prefill graph, max(m) across MUL_MATs is > 1."""
    by_graph: dict[int, int] = {}
    for r in records:
        if r.get("op") != "MUL_MAT" or r.get("phase") != "prefill":
            continue
        gid = int(r["graph_id"])
        by_graph[gid] = max(by_graph.get(gid, 0), int(r["m"]))
    bad = [gid for gid, mx in by_graph.items() if mx <= 1]
    if not by_graph:
        return True, "no prefill graphs to check (skipped)"
    if bad:
        return False, f"{len(bad)} prefill graphs with max(m) <= 1 (e.g. {bad[:3]})"
    return True, f"all {len(by_graph)} prefill graphs have max(m) > 1"


# ---------------------------------------------------------------------------
# Analytical cross-check (given an ArchSpec)
# ---------------------------------------------------------------------------

def _sum_m_for(records: list[dict], category: str) -> int:
    """Sum of M dimensions across all MUL_MAT rows in a category.

    M is the per-row sequence dimension; summing it across rows of one
    category gives the same total regardless of how the graph was sliced.
    """
    return sum(
        int(r["m"]) for r in records
        if r.get("op") == "MUL_MAT" and r.get("layer_category") == category
    )


def _macs_for(records: list[dict], category: str) -> int:
    return sum(
        int(r["macs"]) for r in records
        if r.get("op") == "MUL_MAT" and r.get("layer_category") == category
    )


def check_arch_consistency(
    records: list[dict], arch: ArchSpec
) -> list[tuple[bool, str]]:
    """Cross-check per-category total MACs against the architecture spec.

    For every layer/graph, MAC totals reduce to a constant times sum(M):

        attn_out  → 2 · d_model²                       per row · M
        ffn_gate  → 2 · d_model · d_ff                 per row · M
        ffn_up    → 2 · d_model · d_ff                 per row · M
        ffn_down  → 2 · d_model · d_ff                 per row · M

    So macs(category) == constant(category) · sum_of_M(category).
    """
    d = arch.d_model
    f = arch.d_ff
    n_layers = arch.n_layers
    expectations = {
        "attn_out": 2 * d * d,
        "ffn_gate": 2 * d * f,
        "ffn_up":   2 * d * f,
        "ffn_down": 2 * d * f,
    }
    results: list[tuple[bool, str]] = []
    for cat, per_token in expectations.items():
        sum_m = _sum_m_for(records, cat)
        macs = _macs_for(records, cat)
        if sum_m == 0:
            results.append((True, f"{cat}: no rows (skipped)"))
            continue
        expected = per_token * sum_m
        if macs != expected:
            results.append((
                False,
                f"{cat}: measured {macs} != expected {expected} "
                f"(per-token {per_token}, sum_m {sum_m})",
            ))
        else:
            results.append((
                True,
                f"{cat}: {macs:,} MACs = {per_token:,}·sum(m)={sum_m}",
            ))

    # n_layers consistency: calls(attn_out) should be a multiple of n_layers.
    calls_out = _count_calls(records, "MUL_MAT", "attn_out")
    if calls_out == 0:
        results.append((True, "n_layers check: no attn_out calls (skipped)"))
    elif calls_out % n_layers != 0:
        results.append((
            False,
            f"calls(attn_out)={calls_out} is not a multiple of n_layers={n_layers}",
        ))
    else:
        results.append((
            True,
            f"calls(attn_out)={calls_out} is {calls_out // n_layers} × n_layers",
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def validate_one(trace: Path, arch: ArchSpec | None) -> Report:
    records = load_records(trace)
    rep = Report(trace=trace, passed=[], failed=[])

    structural = [
        ("mac_formula",     check_mac_formula(records)),
        ("attn_qkv_ratio",  check_attention_ratio(records)),
        ("ffn_triplet",     check_ffn_triplet(records)),
        ("decode_m_eq_1",   check_decode_m_is_one(records)),
        ("prefill_max_m",   check_prefill_max_m(records)),
    ]
    for name, (ok, detail) in structural:
        if ok:
            rep.passed.append(f"[{name}] {detail}")
        else:
            rep.failed.append(Failure(name, detail))

    if arch is not None:
        for ok, detail in check_arch_consistency(records, arch):
            tag = "arch_consistency"
            if ok:
                rep.passed.append(f"[{tag}] {detail}")
            else:
                rep.failed.append(Failure(tag, detail))

    return rep


def print_report(rep: Report) -> None:
    rel = str(rep.trace)
    status = "PASS" if rep.ok else "FAIL"
    print(f"\n=== {status}: {rel} ===")
    for line in rep.passed:
        print(f"  ok   {line}")
    for f in rep.failed:
        print(f"  FAIL [{f.check}] {f.detail}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "path", type=Path,
        help="trace.jsonl, or a directory to search recursively for trace.jsonl",
    )
    p.add_argument(
        "--arch", choices=sorted(ARCH_SPECS), default=None,
        help="Architecture spec for the analytical cross-check.",
    )
    args = p.parse_args(argv)

    if args.path.is_file():
        traces = [args.path]
    else:
        traces = sorted(args.path.rglob("trace.jsonl"))
    if not traces:
        print(f"ERROR: no trace.jsonl under {args.path}", file=sys.stderr)
        return 2

    arch = ARCH_SPECS[args.arch] if args.arch else None
    if arch:
        print(f"Analytical spec: {arch.name} "
              f"(n_layers={arch.n_layers}, d_model={arch.d_model}, d_ff={arch.d_ff})")

    all_ok = True
    for trace in traces:
        rep = validate_one(trace, arch)
        print_report(rep)
        all_ok = all_ok and rep.ok
    print()
    print("All checks PASSED." if all_ok else "VALIDATION FAILED.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
