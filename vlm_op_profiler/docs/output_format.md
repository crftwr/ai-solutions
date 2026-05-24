# Output format — vlm-op-profiler

All output files are written under `results/<model>/<run-id>/`.

---

## trace.jsonl

One JSON object per line, one line per executed `ggml` node.

### Schema

```json
{
  "step":           <uint64>,
  "phase":          "prefill" | "decode",
  "graph_id":       <uint64>,
  "node_idx":       <uint32>,
  "op":             <string>,
  "name":           <string>,
  "layer_category": <string>,
  "src0": {
    "type": <string>,
    "ne":   [<int64>, <int64>, <int64>, <int64>]
  },
  "src1": {
    "type": <string>,
    "ne":   [<int64>, <int64>, <int64>, <int64>]
  },
  "dst": {
    "type": <string>,
    "ne":   [<int64>, <int64>, <int64>, <int64>]
  },
  "m":    <int64>,
  "n":    <int64>,
  "k":    <int64>,
  "macs": <int64>
}
```

### Field reference

| Field | Type | Notes |
|-------|------|-------|
| `step` | uint64 | Same as `graph_id`; provided for readability |
| `phase` | string | `"prefill"` or `"decode"` |
| `graph_id` | uint64 | Monotonically increasing per-process counter |
| `node_idx` | uint32 | Index of this node within the graph's node array |
| `op` | string | `ggml_op_name()` value, e.g. `"MUL_MAT"`, `"SOFT_MAX"` |
| `name` | string | `ggml_tensor::name`; may be empty for unnamed nodes |
| `layer_category` | string | See [Layer categories](#layer-categories) |
| `src0.type` | string | `ggml_type_name()` value, e.g. `"Q4_K"`, `"F32"` |
| `src0.ne` | int64[4] | Element counts, innermost-first (`ne[0]` = row length) |
| `src1.*` | — | Same as `src0`; all zeros if node has no second source |
| `dst.*` | — | Destination tensor dtype and shape |
| `m`, `n`, `k` | int64 | MatMul dimensions; 0 for non-matmul ops |
| `macs` | int64 | `2*M*N*K` for MUL_MAT; extended in Phase 2 for other ops |

### Layer categories

| Category | Description |
|----------|-------------|
| `attn_qkv` | Q, K, V projection weights and biases |
| `attn_out` | Attention output projection |
| `ffn_gate` | FFN gate branch (SwiGLU / GeGLU) |
| `ffn_up` | FFN up projection (or first linear in non-gated FFN) |
| `ffn_down` | FFN down projection |
| `norm` | Layer norm / RMS norm |
| `lm_head` | Language-model head projection |
| `embd` | Token / position embedding lookup |
| `vision_conv` | Convolution ops in the vision encoder |
| `vision_attn` | Attention ops in the vision encoder |
| `vision_mlp` | MLP ops in the vision encoder |
| `projector` | Cross-modal projector (VLM bridge) |
| `other` | Unclassified; should be empty for well-supported models |

---

## report.csv

Aggregated statistics in long format.

### Schema

```
model, run_id, phase, layer_category, op, src0_type, src1_type, dst_type, calls, macs
```

| Column | Type | Notes |
|--------|------|-------|
| `model` | string | Model directory name |
| `run_id` | string | ISO-8601 timestamp of the run |
| `phase` | string | `"prefill"` or `"decode"` |
| `layer_category` | string | See above |
| `op` | string | ggml op name |
| `src0_type` | string | ggml type name, e.g. `"Q4_K"` |
| `src1_type` | string | ggml type name; empty for unary ops |
| `dst_type` | string | ggml type name |
| `calls` | uint64 | Number of graph_compute calls this row covers |
| `macs` | uint64 | Total MACs across all `calls` |

---

## report.md

Human-readable Markdown tables derived from `report.csv`.  Includes:

1. **Top-10 MACs by dtype combo** — sorted by descending MAC count.
2. **MACs by layer category and phase** — prefill / decode side-by-side.
3. **Op type breakdown** — all ops, sorted by MAC count.
4. **Run metadata summary** — model, prompt, image hash, llama.cpp commit.

---

## run_meta.json

```json
{
  "profiler_version": "0.1.0",
  "profiler_commit":  "<git sha>",
  "llama_commit":     "<git sha>",
  "model_path":       "<path>",
  "model_sha256":     "<hex>",
  "gguf_metadata":    { ... },
  "prompt":           "<string>",
  "image_path":       "<path or null>",
  "image_sha256":     "<hex or null>",
  "run_id":           "<ISO-8601>",
  "host":             "<hostname>",
  "platform":         "macOS | Linux",
  "inner_backend":    "cpu | metal | cuda"
}
```

---

## summary.csv (cross-model)

Produced by `scripts/summarize.py` from multiple `report.csv` files.

```
model, phase, src0_type, src1_type, dst_type, macs, pct_of_total
```

`pct_of_total` is the fraction of total MACs across all models accounted for
by this dtype combination.
