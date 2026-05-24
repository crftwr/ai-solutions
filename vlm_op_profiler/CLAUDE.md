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
│   ├── fetch_models_hf.py   Download GGUF VLM weights from Hugging Face (suite mode)
│   ├── run_suite.sh         Execute the profiler across the model x prompt x image matrix
│   ├── aggregate.py         Read trace.jsonl files, emit report.csv + report.md
│   └── summarize.py         Cross-model executive summary
│
├── assets/
│   └── example.jpg          64×64 synthetic JPEG for smoke tests (committed to repo)
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

### Phase 2 — MatMul MAC accounting (1 day)

For `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`, compute M, N, K from `src0->ne` and `src1->ne` (respecting transposition conventions) and record `macs = 2 * M * N * K`. For other reduction ops (`GGML_OP_RWKV_WKV`, `GGML_OP_SSM_*`, conv variants), document the formula in `docs/output_format.md` and apply it. Exit criterion: total MACs reported for a 7B LLM decode step is within 1% of a published FLOP count for that model.

### Phase 3 — layer classification & phase tagging (1.5 days)

Parse `tensor->name` (e.g. `blk.0.attn_q.weight`, `vision_model.encoder.layers.0.mlp.fc1`) and bucket ops into: `attn_qkv`, `attn_out`, `ffn_gate`, `ffn_up`, `ffn_down`, `norm`, `lm_head`, `vision_conv`, `vision_attn`, `vision_mlp`, `projector`, `other`. Keep the classifier table in `layer_classifier.cpp` so per-architecture extensions are local. Tag each graph as `prefill` (first forward of a request) or `decode` (subsequent single-token forwards) via `llama_decode` boundaries in the CLI wrapper. Exit criterion: every node in the default suite gets a non-`other` category, or is explicitly listed in `docs/supported_models.md` as a known gap.

### Phase 4 — VLM CLI wrapper (1 day)

`vlm_op_profiler` accepts the same args as `llama-mtmd-cli` plus `--out-dir <path>`, `--steps <N>` (cap on decode tokens), `--include-vision-encode` (profile the image-encoder graph independently of the LLM body). Output goes to `<out-dir>/{trace.jsonl, run_meta.json}`.

### Phase 5 — model suite & runner (1 day)

`scripts/fetch_models_hf.py` (invoked via `make fetch-models-suite`) downloads the default suite into `models/`. Default suite is chosen for architectural diversity:

- LLaVA-style (e.g. LLaVA-1.6)
- Qwen2-VL / Qwen2.5-VL family
- Llama 3.2-Vision
- MiniCPM-V 2.6
- Pixtral
- Phi-3.5-Vision
- SmolVLM
- One late-fusion variant (e.g. Idefics3) for contrast

`run_suite.sh` iterates models x 3 representative prompts x 3 representative images and writes results under `results/<model>/<run-id>/`. Each combination is runnable in isolation for re-runs.

### Phase 6 — aggregation & reporting (1.5 days)

`aggregate.py` reads all `trace.jsonl` under a results directory and produces `report.csv` (long format: model, phase, layer_category, op, src0_dtype, src1_dtype, dst_dtype, calls, macs) and `report.md` (human-readable tables). `summarize.py` reads multiple `report.csv` and produces a cross-model executive summary. Outputs must be deterministic given the same inputs.

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
