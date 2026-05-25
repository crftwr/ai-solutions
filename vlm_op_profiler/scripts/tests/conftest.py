"""Shared test fixtures and path setup for the Phase 6 aggregation scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make scripts/ importable without installing the package.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _trace_record(
    *,
    step: int,
    phase: str,
    op: str,
    layer_category: str,
    src0_type: str,
    src1_type: str,
    dst_type: str,
    macs: int,
    name: str = "",
) -> dict:
    """Build a single trace.jsonl record matching the Phase 4 schema."""
    return {
        "step": step,
        "phase": phase,
        "graph_id": step,
        "node_idx": 0,
        "op": op,
        "name": name,
        "layer_category": layer_category,
        "src0": {"type": src0_type, "ne": [1, 1, 1, 1]},
        "src1": {"type": src1_type, "ne": [1, 1, 1, 1]},
        "dst":  {"type": dst_type,  "ne": [1, 1, 1, 1]},
        "m": 0, "n": 0, "k": 0,
        "macs": macs,
    }


@pytest.fixture()
def fake_results(tmp_path: Path) -> Path:
    """Two models × two runs each, with deterministic trace contents.

    Layout:
        <tmp>/results/
          ModelA/run-001/{trace.jsonl,run_meta.json}
          ModelA/run-002/trace.jsonl
          ModelB/run-001/trace.jsonl
    """
    root = tmp_path / "results"

    # ModelA / run-001: prefill + decode, two dtype combos
    rec_a1 = [
        _trace_record(step=1, phase="prefill", op="MUL_MAT",
                      layer_category="attn_qkv",
                      src0_type="Q4_K", src1_type="F32", dst_type="F32",
                      macs=100),
        _trace_record(step=1, phase="prefill", op="MUL_MAT",
                      layer_category="ffn_down",
                      src0_type="Q4_K", src1_type="F32", dst_type="F32",
                      macs=300),
        _trace_record(step=2, phase="decode", op="MUL_MAT",
                      layer_category="attn_qkv",
                      src0_type="Q4_K", src1_type="F32", dst_type="F32",
                      macs=200),
    ]
    a1_dir = root / "ModelA" / "run-001"
    a1_dir.mkdir(parents=True)
    (a1_dir / "trace.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rec_a1), encoding="utf-8"
    )
    (a1_dir / "run_meta.json").write_text(
        json.dumps({
            "model_path": "/m/ModelA.gguf",
            "image_path": "/i/example_64.jpg",
            "image_sha256": "abcdef0123456789",
            "llama_commit": "deadbeefcafef00d",
            "prompt": "Describe.",
        }) + "\n",
        encoding="utf-8",
    )

    # ModelA / run-002: a vision_encode graph too
    rec_a2 = [
        _trace_record(step=1, phase="vision_encode", op="CONV_2D",
                      layer_category="vision_conv",
                      src0_type="F16", src1_type="F32", dst_type="F32",
                      macs=500),
        _trace_record(step=2, phase="prefill", op="MUL_MAT",
                      layer_category="lm_head",
                      src0_type="Q4_K", src1_type="F32", dst_type="F32",
                      macs=400),
    ]
    a2_dir = root / "ModelA" / "run-002"
    a2_dir.mkdir(parents=True)
    (a2_dir / "trace.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rec_a2), encoding="utf-8"
    )

    # ModelB / run-001
    rec_b1 = [
        _trace_record(step=1, phase="prefill", op="MUL_MAT",
                      layer_category="attn_qkv",
                      src0_type="Q8_0", src1_type="F32", dst_type="F32",
                      macs=700),
        _trace_record(step=1, phase="prefill", op="FLASH_ATTN_EXT",
                      layer_category="attn_out",
                      src0_type="F16", src1_type="F16", dst_type="F32",
                      macs=900),
    ]
    b1_dir = root / "ModelB" / "run-001"
    b1_dir.mkdir(parents=True)
    (b1_dir / "trace.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rec_b1), encoding="utf-8"
    )

    return root


@pytest.fixture()
def fake_report_csvs(tmp_path: Path) -> list[Path]:
    """Two minimal report.csv files for summarize tests."""
    a_dir = tmp_path / "ModelA"
    b_dir = tmp_path / "ModelB"
    a_dir.mkdir()
    b_dir.mkdir()
    a_csv = a_dir / "report.csv"
    b_csv = b_dir / "report.csv"
    a_csv.write_text(
        "model,run_id,phase,layer_category,op,src0_type,src1_type,dst_type,calls,macs\n"
        "ModelA,run-001,prefill,attn_qkv,MUL_MAT,Q4_K,F32,F32,2,300\n"
        "ModelA,run-001,decode,attn_qkv,MUL_MAT,Q4_K,F32,F32,1,200\n",
        encoding="utf-8",
    )
    b_csv.write_text(
        "model,run_id,phase,layer_category,op,src0_type,src1_type,dst_type,calls,macs\n"
        "ModelB,run-001,prefill,attn_qkv,MUL_MAT,Q8_0,F32,F32,1,700\n"
        "ModelB,run-001,prefill,attn_out,FLASH_ATTN_EXT,F16,F16,F32,1,900\n",
        encoding="utf-8",
    )
    return [a_csv, b_csv]


# Ensure pandas isn't influenced by the host locale for column ordering.
os.environ.setdefault("LC_ALL", "C")
