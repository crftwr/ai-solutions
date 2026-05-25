# vlm-op-profiler

A tooling project for collecting tensor-operation statistics from `llama.cpp` when running vision-language models (VLMs). For each model, it produces a per-op breakdown of `ggml` nodes executed, the input/output dtype combinations involved, shapes, MAC counts, and call frequencies. The intended consumer is hardware/kernel design work that needs to know which low-precision dot-product patterns (int4, int8, mxfp4, fp8, bf16, with int32 or bf16 accumulation) actually dominate inference compute on real VLM workloads.

This solution lives under `vlm_op_profiler/` inside the `ai-solutions` monorepo. It is a **measurement tool only**: it does not ship model weights, does not run any inference engine of its own, and does not modify `llama.cpp`'s compute kernels — it only observes.

---

## What the tool produces

For each (model, prompt, image) input it emits, under `results/<model>/<run-id>/`:

- `trace.jsonl` — one line per executed `ggml` node, with op kind, src0/src1/dst dtype and shape, M/N/K (for MatMul), MAC count, layer category, and phase (prefill / decode).
- `report.csv` + `report.md` — aggregations: MAC count and call count grouped by op type, dtype combination, layer category (attention QKV, attention output, FFN gate/up/down, LM head, vision encoder, projector), and phase.
- `run_meta.json` — model SHA, GGUF metadata, prompt, image hash, `llama.cpp` commit, profiler commit.

A separate `scripts/summarize.py` reads multiple per-run `report.csv` files and produces a cross-model executive summary: percentage of total MACs accounted for by each (src0_dtype × src1_dtype → dst_dtype) combination, stacked by model.

---

## Approach

`llama.cpp`'s `ggml` library compiles all inference into a static DAG (`ggml_cgraph`) before dispatching to a backend. Every node already carries full type and shape metadata: op kind (`GGML_OP_MUL_MAT`, `GGML_OP_SOFT_MAX`, …), source tensor types (`GGML_TYPE_Q4_K`, `GGML_TYPE_Q8_0`, `GGML_TYPE_BF16`, …), and shapes/strides. We exploit this via **LD_PRELOAD function interposition** without patching llama.cpp:

1. `libbackend_stats.so` (Linux) / `libbackend_stats.dylib` (macOS) interposes `ggml_backend_sched_graph_compute` and `ggml_backend_sched_graph_compute_async` using `dlsym(RTLD_NEXT, …)` to obtain the real symbols at library-load time.
2. On each intercepted call, the library walks every node in `cgraph->nodes`, records per-node stats (op, dtypes, shapes, MACs, layer category, phase), then calls through to the real scheduler — compute is **bit-identical** to an unmodified run.
3. Records are buffered in memory and flushed to `trace.jsonl` after each prefill/decode step (capped by `PROFSTATS_MAX_STEPS`).
4. Output directory is configured via `PROFSTATS_OUT_DIR`.

The CLI front-end `vlm-op-profiler` is a thin Python script (`cli/vlm_op_profiler.py`) that sets `LD_PRELOAD`/`DYLD_INSERT_LIBRARIES` and the output-path env vars, then `exec`s `llama-mtmd-cli` (VLMs) or `llama-cli` (text-only). It accepts the same arguments as those inner binaries.

### Phase tracker heuristic

`phase_tracker.cpp` tags each graph as `prefill` (max M dimension across MUL_MAT nodes > 1) or `decode` (max M == 1). This is a structural heuristic — it works for standard transformer decode where each step processes a single token.

### Alternatives considered

- **`ggml_backend_i` wrapper (originally planned).** Implements the backend interface and delegates to an inner backend. Abandoned because the interface changes frequently across llama.cpp versions and the per-backend registration approach requires patching the scheduler dispatch loop. LD_PRELOAD interposition at the scheduler level is more stable.
- **Patching `ggml_compute_forward` with a callback.** Lower-overhead and gives runtime call counts directly, but requires maintaining a fork of `llama.cpp`. Rejected.
- **Static graph dump via `ggml_graph_print`.** Cheapest to prototype but does not capture how `K`/sequence-length grow during decode. Useful as a sanity check only.

---

## Directory layout

All paths below are relative to `vlm_op_profiler/` inside the `ai-solutions` monorepo.

```
vlm_op_profiler/
├── src/
│   ├── backend_stats.cpp    LD_PRELOAD interceptor: shadows ggml_backend_sched_graph_compute*,
│   │                        walks cgraph, records per-node stats, calls through to real symbol
│   ├── backend_stats.h      Internal types and env-var names (PROFSTATS_OUT_DIR, PROFSTATS_MAX_STEPS)
│   ├── graph_walker.cpp     Per-op stat extraction: dtype, shape, M/N/K, MAC count
│   ├── layer_classifier.cpp Heuristic tensor-name → layer-category table (44 patterns)
│   └── phase_tracker.cpp    Tags each graph as prefill (max_m > 1) or decode (max_m == 1)
│
├── cli/
│   └── vlm_op_profiler.py   Sets LD_PRELOAD/DYLD_INSERT_LIBRARIES + PROFSTATS_* env vars,
│                            then exec's llama-mtmd-cli (VLM) or llama-cli (text)
│
├── scripts/
│   ├── fetch_models_hf.py     Download GGUF VLM weights from Hugging Face (suite mode)
│   ├── generate_test_images.py  Render the 3 synthetic JPEGs in assets/
│   ├── run_suite.py           Execute the profiler across the model × prompt × image matrix
│   ├── aggregate.py           Read trace.jsonl files, emit report.csv + report.md
│   ├── summarize.py           Cross-model executive summary
│   └── tests/                 pytest coverage for aggregate.py + summarize.py
│
├── assets/                  3 synthetic JPEGs (64/224/448 px) used by run_suite + smoke tests
│
├── third_party/
│   └── llama.cpp/           git submodule (registered in ai-solutions/.gitmodules),
│                            pinned to a known-good upstream commit
│
├── models/                  gitignored; populated by make fetch-model-text / fetch-models
├── results/                 gitignored; one subdirectory per run
├── .env                     gitignored; set HF_TOKEN=hf_... for gated models (see .env.example)
└── docs/
    ├── design.md            Design notes (approach, alternatives, validation methodology)
    ├── output_format.md     Schema for trace.jsonl, report.csv, summary.md
    └── supported_models.md  VLMs known to work end-to-end, with per-model notes
```

> **Monorepo note:** the `llama.cpp` git submodule is registered in `ai-solutions/.gitmodules` (the monorepo root), not inside `vlm_op_profiler/`. To register it:
> ```bash
> # From the ai-solutions repo root:
> git submodule add https://github.com/ggerganov/llama.cpp vlm_op_profiler/third_party/llama.cpp
> ```
> Thereafter, the standard `git submodule update --init --recursive` from the repo root initialises it.

---

## Build & dependencies

- **Docker:** all C/C++ toolchain, cmake, and Python dependencies are bundled inside the Docker image. No local toolchain is required.
- **llama.cpp:** vendored as a git submodule at `vlm_op_profiler/third_party/llama.cpp` (registered in the `ai-solutions` repo root); built inside Docker with `-DGGML_BACKEND_DL=ON`.
- **Python:** 3.11+ with `pandas`, `numpy`, `pyarrow`, `jinja2` (see `requirements.txt`) — bundled in the image; only needed locally if running aggregation scripts outside Docker.
- **Disk:** ~80 GB free for the default model suite at int8 quantization.

Bootstrap (run from `vlm_op_profiler/`):

```bash
# From the ai-solutions repo root — initialise the llama.cpp submodule:
git submodule update --init --recursive

# Then from vlm_op_profiler/:
make docker-build   # builds the image (~10–20 min first time)
```

---

## Implementation plan

Phased so each step produces a usable artifact.

### Phase 0 — repo skeleton ✅

Repo scaffolding, CMake, submodule pin. `cmake --build` green, `vlm-op-profiler --help` prints usage.

### Phase 1 — LD_PRELOAD interceptor ✅

LD_PRELOAD interposition on `ggml_backend_sched_graph_compute` (and `_async`). For each graph, walks nodes and emits one JSONL line per `MUL_MAT` node (op, dtypes, shape, M/N/K, MACs, layer category, phase). Phase tagged by max M dimension heuristic. Layer classifier covers 44 tensor-name patterns across LLaMA, SmolVLM, LLaVA, Qwen2-VL, and common variants.

**Validated:** Llama-3.2-1B-Instruct produces 5030 records across 2 prefill + 10 decode steps, 0 unclassified MUL_MAT, 100.38 GMACs total. See `docs/supported_models.md` for per-model status.

Env vars read by `libbackend_stats`:
- `PROFSTATS_OUT_DIR` — output directory for `trace.jsonl`
- `PROFSTATS_MAX_STEPS` — stop after N decode steps (0 = unlimited)

### Phase 2 — MAC accounting for all significant ops ✅

Extended `graph_walker.cpp` to compute `macs` for every op with non-trivial multiply-accumulate cost:

| Op | Formula |
|----|---------|
| `GGML_OP_MUL_MAT` | `2·M·N·K` (Phase 1) |
| `GGML_OP_MUL_MAT_ID` | `2·K·N·n_expert_used·n_tokens` (M=n_expert_used×n_tokens) |
| `GGML_OP_FLASH_ATTN_EXT` | `2·n_q·B·Sq·Skv·(D+Dv)` — counts both QK and AV matmuls |
| `GGML_OP_CONV_2D` | `2·OW·OH·N·OC·KW·KH·IC` (treated as equivalent matmul) |
| `GGML_OP_SSM_CONV` | `2·d_inner·n_t·n_s·d_conv` (Mamba depthwise conv) |
| `GGML_OP_SSM_SCAN` | `2·d_state·head_dim·n_head·n_t·n_seqs` (Mamba selective scan) |
| `GGML_OP_RWKV_WKV6/7` | `2·S²·H·n_tokens` (outer product + contraction per head) |

Full shape conventions and derivation in `docs/output_format.md § MAC formulas by op type`.

### Phase 3 — layer classification & phase tagging ✅

`layer_classifier.cpp` extended from 44 → 48 name patterns plus a new
`classify_by_op` op-type fallback.  New coverage added:

| Pattern / fallback | Category | Covers |
|--------------------|----------|--------|
| `cache_k`, `cache_v` | `attn_qkv` | KV-cache SET_ROWS / VIEW / PERMUTE |
| `kq_mask` | `attn_qkv` | causal-mask CPY ops |
| `__fattn__` | `attn_out` | FLASH_ATTN_EXT nodes |
| `ffn_inp` | `attn_out` | residual ADD at FFN input |
| `l_out` | `ffn_down` | residual ADD at layer output |
| `ffn_swiglu` | `ffn_gate` | SwiGLU activation (GLU op) |
| `GLU` op | `ffn_gate` | any remaining GLU nodes |
| `GET_ROWS` op | `embd` | unnamed embedding lookups |
| `ROPE` op | `attn_qkv` | rotary position embedding |
| `SOFT_MAX` op | `attn_qkv` | attention score normalisation |
| `RMS_NORM`/`LAYER_NORM`/`GROUP_NORM` op | `norm` | unnamed norm nodes |

Fixed fallback: was returning `""` for unclassified nodes; now returns `"other"`.

**Validated:**
- Llama-3.2-1B-Instruct: 2012 records, **0 unclassified**.
- SmolVLM-Instruct: 751 prefill records captured (inference blocked by token-ID
  mismatch — see `docs/supported_models.md`), **0 unclassified** in captured trace.

Phase tagging (max-M heuristic) unchanged from Phase 1; full CLI-boundary
tagging deferred to Phase 4.

### Phase 4 — VLM CLI wrapper ✅

`cli/vlm_op_profiler.py` accepts the same args as `llama-mtmd-cli` plus the
profiler-specific flags below; everything else is forwarded unchanged.
Outputs land in `<out-dir>/{trace.jsonl, run_meta.json}`.

| Flag | Behaviour |
|------|-----------|
| `--out-dir <path>` | Output directory (created if missing); also exported as `PROFSTATS_OUT_DIR` |
| `--steps <N>` | Cap on decode graphs to record (`0` = unlimited); exported as `PROFSTATS_MAX_STEPS` |
| `--include-vision-encode` | Also record vision-encoder graphs (detected by `GGML_OP_CONV_2D` presence); exported as `PROFSTATS_INCLUDE_VISION_ENCODE`. Default skips them so `trace.jsonl` reflects the LLM body only. |
| `--mtmd-cli <path>` | Explicit path to `llama-mtmd-cli` (auto-located in the Docker image) |

`run_meta.json` is written *before* exec into `llama-mtmd-cli` so it lands
even when inference subsequently fails. Fields include profiler/llama-cpp
commit SHAs (baked at `make docker-build` time via `--build-arg`),
SHA-256 of model/mmproj/image, prompt, ISO-8601 `run_id`, host, platform,
arch, and best-effort `inner_backend`. Full schema in
`docs/output_format.md § run_meta.json`.

Vision-encoder distinction uses a simple structural heuristic: a graph is
tagged `phase = "vision_encode"` iff it contains a `CONV_2D` node, which
in llama.cpp is exclusive to image patch embedding. Tracked separately
from prefill/decode counts in `PhaseTracker`.

### Phase 5 — model suite & runner ✅

**Model fetch** — `scripts/fetch_models_hf.py` (Python, run via
`make fetch-models-suite SUITE=minimal|default|edge|full`) downloads the
selected suite into `models/`. Each entry pairs a main GGUF with its mmproj;
missing / gated repos report per-file failures without aborting the suite,
so a partial download still produces a usable set.

| Suite | Contents | Approx size (int4) |
|-------|----------|-------------------|
| `minimal` | SmolVLM-Instruct only — smoke-test fixture | ~2 GB |
| `default` | minimal + 7 architecturally diverse general-purpose VLMs (LLaVA-1.6, Qwen2-VL-7B, MiniCPM-V 2.6, Phi-3.5-V, Pixtral-12B, Llama-3.2-11B-Vision*, Idefics3-8B) | ~80 GB |
| `edge` | minimal + edge / physical-AI / robotics models (InternVL2-2B, Qwen2.5-VL-3B, moondream2, PaliGemma 2 3B, Florence-2 base, Gemma 3 4B-it) + Q8_0 / IQ3_M quants for SmolVLM / Phi-3.5-V / InternVL2-2B | ~25 GB |
| `full` | every spec in the registry | ~180 GB |

\* Llama-3.2-11B-Vision is gated; populate `HF_TOKEN` in `.env`.
`HF_TOKEN` flows from `.env` → Makefile → container via `HF_TOKEN_ARG`.

The `edge` suite intentionally targets Jetson-class and Qualcomm AI SoC
deployments — small models that fit in 4–8 GB plus extra quantization
variants so the profiler captures the dominant low-precision dot-product
pattern on the target hardware (int4 / int8 / sub-4-bit). See
`docs/supported_models.md § Quantization variants on edge` for the
per-quant rationale.

Run-suite parity: `scripts/run_suite.py` carries the same model list, so
once an edge model is fetched it is automatically exercised by
`make run-suite`. Missing files are silently skipped per combination.

**Test fixtures** — `scripts/generate_test_images.py` (run via
`make gen-test-images`) produces three synthetic JPEGs at sizes that
exercise different vision-encoder code paths:

| File | Size | Pattern |
|------|------|---------|
| `assets/example_64.jpg` | 64×64 | deterministic xorshift noise |
| `assets/example_224.jpg` | 224×224 | 8-cell color-block grid (typical CLIP ViT input) |
| `assets/example_448.jpg` | 448×448 | smooth radial gradient (typical LLaVA-1.6 patch) |

These are checked in so `make smoke-test` works without re-running the
generator. Re-run only after changing the generator.

**Suite runner** — `scripts/run_suite.py` (run via `make run-suite`,
optional `SUITE_ARGS="--only SmolVLM --dry-run"`) iterates
models × 3 prompts × 3 images and writes
`results/<model>/<YYYYMMDDTHHMMSSZ>_<image>_<prompt>/`. Each combination
is launched as its own subprocess so a single failure (e.g. SmolVLM's
token-ID issue) doesn't abort the suite; combination directories are
self-contained for easy re-runs. A `results/suite_<timestamp>.json`
summary captures per-run exit codes for later inspection.

### Phase 6 — aggregation & reporting ✅

**`scripts/aggregate.py`** walks a results directory, loads every
`<model>/<run_id>/trace.jsonl`, and writes:

- `report.csv` — long-format, columns: `model, run_id, phase, layer_category,
  op, src0_type, src1_type, dst_type, calls, macs`.
- `report.md` — four sections:
  1. **Top dtype combinations** (top 10 by descending MACs)
  2. **MACs by layer category and phase** — pivot with one column per phase
     (`prefill | decode | vision_encode`) plus a row total / % grand-total.
  3. **Op type breakdown** — every op, sorted by descending MACs.
  4. **Run metadata** — one row per discovered `run_meta.json` with model,
     image, image-SHA prefix, llama_commit prefix, and prompt.

Outputs are **deterministic**: rows are sorted by `(macs desc, then every
group column ascending)` with `kind="mergesort"`; CSV uses `lineterminator="\n"`
so files are byte-identical across operating systems. The aggregate test
suite hashes outputs and re-runs to enforce this.

The combined report lands at `<results_dir>/report.{csv,md}`. Passing
`--per-run` (the Makefile default) also writes a report next to each
`trace.jsonl` so individual runs are self-describing.

**`scripts/summarize.py`** reads multiple per-run `report.csv` files and
emits a cross-model executive summary at `--out/`:

- `summary.csv` — columns: `model, src0_type, src1_type, dst_type, macs,
  pct_of_total`, sorted by descending MACs.
- `summary.md` — top-20 dtype combinations table + per-model MAC totals.

CLI argument order does not affect output — input paths are sorted before
processing. The summarize test suite verifies this and hashes outputs across
two runs to enforce determinism.

**Makefile**

| Target | Purpose |
|--------|---------|
| `make aggregate` | Runs `aggregate.py --per-run /app/results` inside the Docker image. Forward extras via `AGGREGATE_ARGS=...`. |
| `make summarize` | Roll up every `results/**/report.csv` (excluding the combined one) into `results/summary.{csv,md}`. Run `make aggregate` first. |
| `make test`     | Runs `pytest scripts/tests/` inside the image — covers schema, totals, vision_encode handling, deterministic output, and argument-order independence. |

### Phase 7 — validation & docs (1 day)

Cross-check totals against `llama.cpp`'s own `--verbose` graph dump for at least two models. Fill in `docs/supported_models.md` (GGUF source, preprocessing quirks, known op-name patterns). Add a regression test that runs the smallest model end-to-end on every commit.

**Total: ~10 working days for a first usable end-to-end pipeline.**

---

## Output schema (summary)

`trace.jsonl` — one record per executed `ggml` node:

```json
{
  "step": 142,
  "phase": "decode",
  "graph_id": 142,
  "node_idx": 37,
  "op": "MUL_MAT",
  "name": "blk.5.attn_q.weight*x",
  "layer_category": "attn_qkv",
  "src0": {"type": "Q4_K", "ne": [4096, 4096, 1, 1]},
  "src1": {"type": "F32",  "ne": [4096, 1, 1, 1]},
  "dst":  {"type": "F32",  "ne": [4096, 1, 1, 1]},
  "m": 4096, "n": 1, "k": 4096,
  "macs": 33554432
}
```

Full schema in `docs/output_format.md`.

---

## Conventions

- **Language:** all code, comments, commit messages, docs, and issue text in English.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`); one logical change per commit; reference the phase from the implementation plan in the body when relevant.
- **C++:** C++17, `clang-format` with LLVM style, no exceptions in the backend hot path (matches `ggml`'s conventions), prefer `std::string_view` for tensor-name parsing.
- **Python:** `ruff` for lint and format, type hints required on public functions, `pytest` for the aggregation scripts.
- **Determinism:** the profiler must not introduce nondeterminism. Any feature that requires sampling is gated behind an explicit `--sample` flag.
- **No model weights or run outputs in git.** `models/` and `results/` are `.gitignore`d. Anything reproducible from a script must not be checked in.
- **Upstream tracking:** the `llama.cpp` submodule is updated only via a deliberate bump commit made from the `ai-solutions` root. The upstream commit hash and any required patches go in the bump commit's body, and the regression test is re-run before merge.

---

## Non-goals

- Does not modify `ggml`'s compute kernels.
- Does not estimate latency, energy, or memory bandwidth. It counts ops and MACs only; downstream tooling can combine MAC counts with roofline parameters.
- Does not support non-`ggml` runtimes (PyTorch, TensorRT-LLM, vLLM). A parallel tool with the same output schema could be built for those; see `docs/design.md`.

---

## External references

- llama.cpp: https://github.com/ggerganov/llama.cpp
- ggml backend interface: `third_party/llama.cpp/ggml/include/ggml-backend.h`
- GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Multimodal CLI (`llama-mtmd-cli`): https://github.com/ggerganov/llama.cpp/tree/master/examples/llava
