"""Tests for scripts/validate.py — invariants run against synthetic traces."""

from __future__ import annotations

import json
from pathlib import Path

import validate as v_mod
from validate import (
    ArchSpec,
    check_attention_ratio,
    check_decode_m_is_one,
    check_ffn_triplet,
    check_mac_formula,
    check_prefill_max_m,
    check_arch_consistency,
    validate_one,
)


# ---------------------------------------------------------------------------
# Tiny synthetic trace builders
# ---------------------------------------------------------------------------

def _mm(m: int, n: int, k: int, *, phase: str, cat: str,
        op: str = "MUL_MAT", graph_id: int = 1) -> dict:
    return {
        "step": graph_id, "phase": phase, "graph_id": graph_id, "node_idx": 0,
        "op": op, "name": "", "layer_category": cat,
        "src0": {"type": "Q4_K", "ne": [k, n, 1, 1]},
        "src1": {"type": "F32",  "ne": [k, m, 1, 1]},
        "dst":  {"type": "F32",  "ne": [n, m, 1, 1]},
        "m": m, "n": n, "k": k,
        "macs": 2 * m * n * k,
    }


def _llama_layer_rows(m: int, *, n_layers: int, phase: str,
                      d_model: int, d_kv: int, d_ff: int) -> list[dict]:
    """A well-formed (Q,K,V,O,gate,up,down) per layer at the given m."""
    rows: list[dict] = []
    for _ in range(n_layers):
        rows.append(_mm(m, d_model, d_model, phase=phase, cat="attn_qkv"))  # Q
        rows.append(_mm(m, d_kv,    d_model, phase=phase, cat="attn_qkv"))  # K
        rows.append(_mm(m, d_kv,    d_model, phase=phase, cat="attn_qkv"))  # V
        rows.append(_mm(m, d_model, d_model, phase=phase, cat="attn_out"))
        rows.append(_mm(m, d_ff,    d_model, phase=phase, cat="ffn_gate"))
        rows.append(_mm(m, d_ff,    d_model, phase=phase, cat="ffn_up"))
        rows.append(_mm(m, d_model, d_ff,    phase=phase, cat="ffn_down"))
    return rows


# ---------------------------------------------------------------------------
# Structural checks — happy path
# ---------------------------------------------------------------------------

def test_mac_formula_ok_for_consistent_rows() -> None:
    rows = _llama_layer_rows(m=2, n_layers=4, phase="prefill",
                             d_model=128, d_kv=32, d_ff=256)
    ok, _ = check_mac_formula(rows)
    assert ok


def test_mac_formula_detects_corruption() -> None:
    rows = _llama_layer_rows(m=2, n_layers=2, phase="prefill",
                             d_model=128, d_kv=32, d_ff=256)
    rows[3]["macs"] = 0  # corrupt one row's MAC count
    ok, detail = check_mac_formula(rows)
    assert not ok
    assert "macs != 2·m·n·k" in detail


def test_attention_ratio() -> None:
    rows = _llama_layer_rows(m=2, n_layers=3, phase="prefill",
                             d_model=128, d_kv=32, d_ff=256)
    ok, _ = check_attention_ratio(rows)
    assert ok
    # Drop one Q row → 8 attn_qkv, 3 attn_out → 8 != 9
    rows = [r for i, r in enumerate(rows) if i != 0]
    ok, detail = check_attention_ratio(rows)
    assert not ok
    assert "attn_qkv" in detail


def test_ffn_triplet() -> None:
    rows = _llama_layer_rows(m=2, n_layers=2, phase="prefill",
                             d_model=128, d_kv=32, d_ff=256)
    ok, _ = check_ffn_triplet(rows)
    assert ok
    # Remove one ffn_down → triplet broken
    rows = [r for r in rows if not (r["op"] == "MUL_MAT" and r["layer_category"] == "ffn_down")][:-1] + \
           [r for r in rows if r["op"] == "MUL_MAT" and r["layer_category"] == "ffn_down"][:-1]
    ok, detail = check_ffn_triplet(rows)
    assert not ok
    assert "ffn_" in detail


def test_decode_m_is_one_detects_violation() -> None:
    rows = _llama_layer_rows(m=1, n_layers=2, phase="decode",
                             d_model=128, d_kv=32, d_ff=256)
    ok, _ = check_decode_m_is_one(rows)
    assert ok
    rows.append(_mm(2, 128, 128, phase="decode", cat="attn_out"))
    ok, detail = check_decode_m_is_one(rows)
    assert not ok
    assert "decode MUL_MATs with m > 1" in detail


def test_prefill_max_m_skipped_when_no_prefill() -> None:
    rows = _llama_layer_rows(m=1, n_layers=2, phase="decode",
                             d_model=128, d_kv=32, d_ff=256)
    ok, detail = check_prefill_max_m(rows)
    assert ok and "skipped" in detail


# ---------------------------------------------------------------------------
# Analytical cross-check
# ---------------------------------------------------------------------------

ARCH = ArchSpec(
    name="synthetic", n_layers=4,
    d_model=128, n_heads=8, n_kv_heads=2,
    d_ff=256, vocab=1024, lm_head_tied=True,
)


def test_arch_consistency_holds_for_well_formed_trace() -> None:
    rows: list[dict] = []
    # Two prefill graphs at m=2 and m=3 (different prompt lengths).
    rows += _llama_layer_rows(m=2, n_layers=ARCH.n_layers, phase="prefill",
                              d_model=ARCH.d_model,
                              d_kv=ARCH.d_model * ARCH.n_kv_heads // ARCH.n_heads,
                              d_ff=ARCH.d_ff)
    rows += _llama_layer_rows(m=3, n_layers=ARCH.n_layers, phase="prefill",
                              d_model=ARCH.d_model,
                              d_kv=ARCH.d_model * ARCH.n_kv_heads // ARCH.n_heads,
                              d_ff=ARCH.d_ff)
    for ok, _ in check_arch_consistency(rows, ARCH):
        assert ok


def test_arch_consistency_catches_wrong_d_ff() -> None:
    rows = _llama_layer_rows(m=2, n_layers=ARCH.n_layers, phase="prefill",
                             d_model=ARCH.d_model,
                             d_kv=ARCH.d_model * ARCH.n_kv_heads // ARCH.n_heads,
                             d_ff=ARCH.d_ff)
    bad_arch = ArchSpec(**{**ARCH.__dict__, "d_ff": ARCH.d_ff * 2})
    results = check_arch_consistency(rows, bad_arch)
    failed = [r for ok, _ in results for r in [ok] if not r]
    assert failed, "Wrong d_ff should fail the analytical cross-check"


# ---------------------------------------------------------------------------
# End-to-end via validate_one / trace.jsonl on disk
# ---------------------------------------------------------------------------

def _write_trace(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "trace.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_validate_one_passes_for_clean_trace(tmp_path: Path) -> None:
    rows = _llama_layer_rows(m=2, n_layers=ARCH.n_layers, phase="prefill",
                             d_model=ARCH.d_model,
                             d_kv=ARCH.d_model * ARCH.n_kv_heads // ARCH.n_heads,
                             d_ff=ARCH.d_ff)
    rows += _llama_layer_rows(m=1, n_layers=ARCH.n_layers, phase="decode",
                              d_model=ARCH.d_model,
                              d_kv=ARCH.d_model * ARCH.n_kv_heads // ARCH.n_heads,
                              d_ff=ARCH.d_ff)
    trace = _write_trace(tmp_path, rows)
    rep = validate_one(trace, ARCH)
    assert rep.ok, [f"{f.check}: {f.detail}" for f in rep.failed]


def test_validate_main_exits_zero_on_clean_trace(tmp_path: Path) -> None:
    rows = _llama_layer_rows(m=2, n_layers=ARCH.n_layers, phase="prefill",
                             d_model=ARCH.d_model,
                             d_kv=ARCH.d_model * ARCH.n_kv_heads // ARCH.n_heads,
                             d_ff=ARCH.d_ff)
    trace = _write_trace(tmp_path, rows)
    rc = v_mod.main([str(trace)])
    assert rc == 0
