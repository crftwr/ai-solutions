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
| `phase` | string | `"prefill"`, `"decode"`, or `"vision_encode"` (the last only appears when `--include-vision-encode` is set) |
| `graph_id` | uint64 | Monotonically increasing per-process counter |
| `node_idx` | uint32 | Index of this node within the graph's node array |
| `op` | string | `ggml_op_name()` value, e.g. `"MUL_MAT"`, `"SOFT_MAX"` |
| `name` | string | `ggml_tensor::name`; may be empty for unnamed nodes |
| `layer_category` | string | See [Layer categories](#layer-categories) |
| `src0.type` | string | `ggml_type_name()` value, e.g. `"Q4_K"`, `"F32"` |
| `src0.ne` | int64[4] | Element counts, innermost-first (`ne[0]` = row length) |
| `src1.*` | — | Same as `src0`; all zeros if node has no second source |
| `dst.*` | — | Destination tensor dtype and shape |
| `m`, `n`, `k` | int64 | MatMul dimensions; 0 for ops with no single rectangular decomposition |
| `macs` | int64 | Multiply-accumulate count; formula depends on op (see below) |

### MAC formulas by op type

The `macs` field uses the formula below for each op.  For all other ops
`macs = 0` (no significant multiply-accumulate work).

| Op | Tensor shapes | `m` | `n` | `k` | `macs` formula |
|----|--------------|-----|-----|-----|----------------|
| `MUL_MAT` | src0=[K,N,…], src1=[K,M,…] | M=src1.ne[1] | N=src0.ne[1] | K=src0.ne[0] | `2·M·N·K` |
| `MUL_MAT_ID` | src0=[K,N,n_exp], src1=[K,e_used,n_tok], src2=[e_used,n_tok] | e_used·n_tok | N | K | `2·M·N·K` |
| `FLASH_ATTN_EXT` | q=[D,n_q,Sq,B], k=[D,n_kv,Skv,B], v=[Dv,n_kv,Skv,B] | 0 | 0 | 0 | `2·n_q·B·Sq·Skv·(D+Dv)` |
| `CONV_2D` | kernel=[KW,KH,IC,OC], data=[W,H,IC,N], dst=[OW,OH,OC,N] | OW·OH·N | OC | KW·KH·IC | `2·m·n·k` |
| `SSM_CONV` | sx=[d_conv-1+n_t,d_inner,n_s], c=[d_conv,d_inner], dst=[d_inner,n_t,n_s] | 0 | 0 | 0 | `2·d_inner·n_t·n_s·d_conv` |
| `SSM_SCAN` | state=[d_state,hdim,n_head], x=[hdim,n_head,n_t,n_seqs] | 0 | 0 | 0 | `2·d_state·hdim·n_head·n_t·n_seqs` |
| `RWKV_WKV6` | k=[S,H,n_tok] | 0 | 0 | 0 | `2·S²·H·n_tok` |
| `RWKV_WKV7` | r=[S,H,n_tok] | 0 | 0 | 0 | `2·S²·H·n_tok` |

**Notes:**

- `MUL_MAT_ID` is the MoE routed matmul (Mixtral, Qwen-MoE, etc.).  `M` is the
  total (expert_used × token) pairs across the batch; `m/n/k` are populated
  and satisfy `macs = 2·m·n·k`.

- `FLASH_ATTN_EXT` counts both the Q·K^T matmul (`2·n_q·B·Sq·Skv·D`) and the
  A·V matmul (`2·n_q·B·Sq·Skv·Dv`) in a single field.  When `D == Dv` (the
  common case) this simplifies to `4·n_q·B·Sq·Skv·D`.  `m/n/k` are left
  at 0 because the operation is not a single rectangular matmul.

- `CONV_2D` is treated as an equivalent matmul of shape
  (OW·OH·N) × OC over inner dimension KW·KH·IC, so `macs = 2·m·n·k`.

- `SSM_CONV` is the Mamba depthwise conv-1d on a rolling state window.  It is
  a per-channel 1-D dot product of length `d_conv` applied at each of
  `d_inner·n_t·n_s` positions.

- `SSM_SCAN` counts the two dominant multiply steps per timestep per head:
  the B·x outer product and the C·state contraction, each costing
  `d_state×head_dim` multiply-accumulates.

- `RWKV_WKV6/7` counts the k⊗v outer product and the r·state contraction,
  both of size S² per head per token.

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

Written by `cli/vlm_op_profiler.py` before exec'ing into `llama-mtmd-cli`.
All fields are best-effort; missing values are emitted as `null`.

```json
{
  "profiler_version":      "0.1.0-dev",
  "profiler_commit":       "<git sha or 'unknown'>",
  "llama_commit":          "<git sha or 'unknown'>",
  "model_path":            "<path or null>",
  "model_sha256":          "<hex or null>",
  "mmproj_path":           "<path or null>",
  "mmproj_sha256":         "<hex or null>",
  "gguf_metadata":         {},
  "prompt":                "<string or null>",
  "image_path":            "<path or null>",
  "image_sha256":          "<hex or null>",
  "run_id":                "<ISO-8601 UTC>",
  "host":                  "<hostname>",
  "platform":              "macOS | Linux | ...",
  "arch":                  "<machine arch, e.g. x86_64 / arm64>",
  "inner_backend":         "cpu | metal | cuda",
  "include_vision_encode": true | false
}
```

Notes:

- `profiler_commit` / `llama_commit` are baked into the Docker image at
  `make docker-build` time via `--build-arg`; outside Docker, they are
  resolved by walking up from the CLI script to the nearest `.git`.
- `gguf_metadata` is reserved for later phases; currently always `{}`.
- `include_vision_encode` reflects whether the `--include-vision-encode`
  flag was passed and therefore whether `trace.jsonl` may contain
  `phase: "vision_encode"` rows.

---

## summary.csv (cross-model)

Produced by `scripts/summarize.py` from multiple `report.csv` files.

```
model, phase, src0_type, src1_type, dst_type, macs, pct_of_total
```

`pct_of_total` is the fraction of total MACs across all models accounted for
by this dtype combination.
