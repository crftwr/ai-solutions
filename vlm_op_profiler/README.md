# vlm-op-profiler

Collects tensor-operation statistics from [`llama.cpp`](https://github.com/ggerganov/llama.cpp) while running vision-language models (VLMs). For each model it produces a per-op breakdown of `ggml` nodes executed — input/output dtype combinations, shapes, MAC counts, and call frequencies — so that hardware and kernel designers can see which low-precision dot-product patterns actually dominate inference compute on real workloads.

This is a **measurement tool only**. It does not ship model weights, run its own inference engine, or modify `llama.cpp`'s compute kernels.

---

## Quick start

```bash
# From the ai-solutions repo root — initialise the llama.cpp submodule:
git submodule update --init --recursive

# Then from vlm_op_profiler/:
make docker-build        # builds the image (~10–20 min first time)

# Download a small public text model and run a validated smoke test:
make fetch-model-text    # downloads Llama-3.2-1B-Instruct-IQ3_M.gguf (~660 MB)
make smoke-test-text     # runs llama-cli + LD_PRELOAD, verifies trace.jsonl

# Download the SmolVLM GGUF files and run the VLM smoke test:
make fetch-models        # downloads SmolVLM Q4_K_M + mmproj (~2 GB)
make smoke-test          # requires assets/example.jpg (see below)
```

#### Gated models (LLaVA etc.)

Create `vlm_op_profiler/.env` (gitignored) with your HuggingFace token:

```bash
HF_TOKEN=hf_...
```

`make fetch-models` and `make fetch-model-text` automatically pass the token when present.

#### Running on your own model

```bash
docker run --rm \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/assets:/app/assets:ro" \
  vlm-op-profiler \
  --model   /app/models/llava-1.6-mistral-7b.Q4_K_M.gguf \
  --mmproj  /app/models/llava-1.6-mistral-7b-mmproj.gguf \
  --image   /app/assets/example.jpg \
  --out-dir /app/results/llava-test \
  -p "Describe this image in detail."
```

---

## What it produces

Under `results/<run>/`:

| File | Contents |
|------|----------|
| `trace.jsonl` | One line per executed `ggml` node: op, src0/src1/dst dtype & shape, M/N/K, MACs, layer category, phase |
| `report.csv` | MACs + call count grouped by op, dtype combo, layer category, and prefill/decode phase |
| `report.md` | Human-readable tables derived from `report.csv` |
| `run_meta.json` | Model SHA, GGUF metadata, prompt, image hash, `llama.cpp` and profiler commit |

```bash
# Aggregate a single run:
python scripts/aggregate.py results/my-run

# Cross-model executive summary:
python scripts/summarize.py results/*/report.csv
```

---

## Test assets

`assets/example.jpg` is a small synthetic 64×64 JPEG used by `make smoke-test`. It is committed to the repo; no download needed. Replace it with any representative image for real profiling runs.

---

## Model suite

Download the default model suite (VLMs only, ~80 GB at int8 quantisation):

```bash
make fetch-models-suite          # full suite
```

Default suite covers LLaVA-1.6, Qwen2-VL, Llama-3.2-Vision, MiniCPM-V 2.6, Pixtral, Phi-3.5-Vision, SmolVLM, and Idefics3.

See [docs/supported_models.md](docs/supported_models.md) for per-model validation status and known issues.

---

## Architecture overview

The profiler uses **LD_PRELOAD function interposition** (on Linux; `DYLD_INSERT_LIBRARIES` on macOS) to intercept `ggml_backend_sched_graph_compute` and `ggml_backend_sched_graph_compute_async` without patching `llama.cpp`. The interceptor:

1. Resolves the real symbol via `dlsym(RTLD_NEXT, ...)` at library-load time.
2. On each call, walks every node in the `ggml_cgraph` to record per-node stats (op, dtypes, shape, MACs, layer category, phase).
3. Calls through to the real scheduler function — compute is bit-identical to an unmodified run.
4. Buffers records in memory and flushes them to `trace.jsonl` after each prefill/decode step.

The `vlm-op-profiler` CLI is a thin wrapper around `llama-mtmd-cli` (for VLMs) or `llama-cli` (for text-only models) that sets `LD_PRELOAD` / `DYLD_INSERT_LIBRARIES` and the output-path env vars, then `exec`s the inner binary.

See [docs/design.md](docs/design.md) for the full design and alternatives considered.

---

## Build dependencies

| Dependency | Version | Notes |
|-----------|---------|-------|
| Docker | ≥ 20.10 | Required — bundles the C++ toolchain, cmake, llama.cpp, and Python env |
| Python | ≥ 3.11 | For running aggregation scripts outside Docker (optional) |
| pandas / numpy / pyarrow / jinja2 | see `requirements.txt` | Only needed when running scripts outside Docker |

---

## Makefile targets

| Target | Action |
|--------|--------|
| `make docker-build` | Build the Docker image (primary path) |
| `make docker-run` | Run `vlm-op-profiler --help` in Docker |
| `make docker-shell` | Interactive shell in the container |
| `make docker-clean` | Remove the Docker image |
| `make fetch-model-text` | Download Llama-3.2-1B-Instruct-IQ3_M.gguf (~660 MB, public) |
| `make fetch-models` | Download SmolVLM-Instruct Q4_K_M + mmproj (~2 GB, public) |
| `make fetch-models-suite` | Download full VLM suite via `scripts/fetch_models_hf.py` |
| `make smoke-test-text` | End-to-end validation with `llama-cli` + text model; verifies prefill+decode in `trace.jsonl` |
| `make smoke-test` | End-to-end test with SmolVLM + `assets/example.jpg` |
| `make run-suite` | Run profiler across default model × prompt × image matrix |
| `make aggregate` | Run `scripts/aggregate.py` over `results/` |
| `make test` | Run Python unit tests (`pytest`) |

---

## Output schema

`trace.jsonl` example record:

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

Full schema: [docs/output_format.md](docs/output_format.md).

---

## Conventions

- All code, comments, commits, and docs in **English**.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/).
- C++: C++17, LLVM `clang-format` style, no exceptions in the backend hot path.
- Python: `ruff` lint/format, type hints on public functions, `pytest`.
- `models/` and `results/` are gitignored — never commit weights or run outputs.
- `.env` is gitignored — never commit tokens; see `.env.example` for the expected format.
