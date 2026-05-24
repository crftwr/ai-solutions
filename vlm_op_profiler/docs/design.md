# Design notes — vlm-op-profiler

## Goal

Produce a **deterministic, per-op MAC and call-count breakdown** of `llama.cpp`
inference for a set of vision-language models (VLMs).  The output should tell a
hardware designer, for each architectural layer category (attention QKV, FFN
down, vision MLP, etc.) and each dtype combination, exactly how many MACs are
consumed — broken down by prefill vs decode phase.

## Core approach: profiling backend wrapper

`llama.cpp`'s `ggml` library builds a static DAG (`ggml_cgraph`) before every
compute step.  Each node in the DAG already carries full type and shape
metadata (op kind, source tensor types, shapes/strides) **before any arithmetic
runs**.  We exploit this by implementing a ggml backend wrapper:

1. **Implements `ggml_backend_i`** (the interface struct in
   `ggml/src/ggml-backend-impl.h`).
2. **Holds a pointer to an inner backend** (CPU, Metal, CUDA) in its context.
3. In **`graph_compute`**, walks every `cgraph->nodes[i]` and records
   `NodeStats` (op, dtypes, shapes, MACs), then forwards the graph to the inner
   backend for actual execution.
4. **Buffers records** and flushes them to `trace.jsonl` at the end of each
   graph_compute call.

The backend is a **shared library** (`libbackend_stats.dylib` / `.so`).

### Load mechanism

The profiling backend is injected into the llama.cpp process via:
- **macOS:** `DYLD_INSERT_LIBRARIES=<path>/libbackend_stats.dylib`
- **Linux:** `LD_PRELOAD=<path>/libbackend_stats.so`

A `__attribute__((constructor))` function calls `ggml_backend_register()` so
the backend appears in ggml's backend registry before any inference code runs.

The `vlm_op_profiler` CLI (Phase 4) sets these environment variables, adds
`PROFSTATS_OUT_DIR` and other config vars, then exec's `llama-mtmd-cli`.

### Why not patch ggml directly?

Patching `ggml_compute_forward` (e.g. adding a callback there) would be lower
overhead and give the same data, but requires maintaining a fork of `llama.cpp`.
The backend wrapper approach keeps us forward-compatible as `llama.cpp` evolves.
When the upstream `ggml_backend_i` interface changes we update our implementation
once; when llama.cpp gets new op types we extend `graph_walker.cpp`.

### Why not use `ggml_backend_sched_set_eval_callback`?

The scheduler eval callback fires after the graph has been split and allocated,
and it fires per-node rather than per-graph.  It does not easily expose the
full cgraph for a single-pass walk.  It also requires access to the
`ggml_backend_sched_t` handle inside llama, which our library cannot obtain
without further interception.  The eval callback may become the preferred
mechanism if the backend-wrapper approach proves too invasive in a future
llama.cpp version.

### Why not static graph dump?

`ggml_graph_print` / `ggml_graph_dump_dot` capture the graph once, before
execution.  They do not capture how shapes change as the KV cache grows during
decode (sequence length increases with each token), so they under-count decode
MACs.  Useful as a sanity check (Phase 7) but not as the primary mechanism.

---

## Layer classification

Tensor names in `llama.cpp` follow a consistent scheme set by the model loader.
Examples:

- `blk.0.attn_q.weight` — LLaMA attention Q projection, layer 0
- `vision_model.encoder.layers.5.mlp.fc1` — CLIP/LLaVA vision MLP, layer 5
- `mm_projector.1.weight` — cross-modal projector

`layer_classifier.cpp` maps tensor name substrings to categories.  The mapping
is a simple ordered table; architecture-specific patterns are grouped by
comment block to make updates easy.

---

## Phase tracking

A "prefill" graph processes the full prompt (M > 1 for MUL_MAT nodes where M
is the sequence length).  A "decode" graph processes a single new token (M = 1).
The `PhaseTracker` inspects the maximum M dimension observed across MUL_MAT
nodes in each graph to classify it.

Vision-encoder graphs run once during prefill; they are tagged `prefill` with
`layer_category` values in the `vision_*` / `projector` namespace.

---

## MAC counting

For `GGML_OP_MUL_MAT(A, B) → C`:

```
src[0] = A  (weights):      ne = [K, N, ...]
src[1] = B  (activations):  ne = [K, M, ...]
dst    = C:                 ne = [N, M, ...]
macs = 2 * M * N * K
```

For `GGML_OP_MUL_MAT_ID` (used for MoE expert gating), the formula is the same
but applied per-expert; M is the number of tokens routed to that expert.

Other ops with non-trivial multiply-accumulate patterns (convolutions,
RWKV WKV, SSM scans) are extended in Phase 2 and documented in
`docs/output_format.md`.

---

## Validation methodology (Phase 7)

1. Run the profiler on a 7B LLaMA model with a batch of 1 token (decode).
   Total MACs should equal `2 * seq_len * (32 * (4096² + 2 * 4096 * 11008))` ≈
   published FLOP counts (within 1%).
2. Cross-check `trace.jsonl` node list against `--verbose` ggml graph dump from
   `llama-cli`.
3. Run a regression test on every commit: smallest model, fixed prompt, fixed
   seed, compare `report.csv` hash to a golden reference.

---

## Non-goals

- Latency, energy, or memory-bandwidth estimation.
- Non-ggml runtimes (PyTorch, TensorRT-LLM, vLLM).
- Model quantisation or re-encoding.
