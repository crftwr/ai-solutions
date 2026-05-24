# vlm-op-profiler

Collects tensor-operation statistics from [`llama.cpp`](https://github.com/ggerganov/llama.cpp) while running vision-language models (VLMs). For each model it produces a per-op breakdown of `ggml` nodes executed — input/output dtype combinations, shapes, MAC counts, and call frequencies — so that hardware and kernel designers can see which low-precision dot-product patterns actually dominate inference compute on real workloads.

This is a **measurement tool only**. It does not ship model weights, run its own inference engine, or modify `llama.cpp`'s compute kernels.

---

## Quick start

### Docker (recommended — no local cmake required)

```bash
# From the ai-solutions repo root — initialise the llama.cpp submodule:
git submodule update --init --recursive

# Then from vlm_op_profiler/:
make docker-build        # builds the image (~10–20 min first time)
make docker-run          # prints usage

# Run on a single model (mount your models/ dir):
docker run --rm \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/results:/app/results" \
  vlm-op-profiler \
  --model  /app/models/llava-v1.6-mistral-7b.Q4_K_M.gguf \
  --image  /app/models/example.jpg \
  --out-dir /app/results/llava-v1.6-test \
  "Describe this image in detail."
```

### Local build (requires cmake ≥ 3.22 and clang/gcc)

```bash
git submodule update --init --recursive  # from repo root
make setup   # cmake configure + Python venv
make build   # compile backend + CLI

./build/vlm-op-profiler --help
```

## What it produces

Under `results/<model>/<run-id>/`:

| File | Contents |
|------|----------|
| `trace.jsonl` | One line per executed `ggml` node: op, src0/src1/dst dtype & shape, M/N/K, MACs, layer category, phase |
| `report.csv` | MACs + call count grouped by op, dtype combo, layer category, and prefill/decode phase |
| `report.md` | Human-readable tables derived from `report.csv` |
| `run_meta.json` | Model SHA, GGUF metadata, prompt, image hash, `llama.cpp` and profiler commit |

```bash
# Aggregate across all runs under results/:
python scripts/aggregate.py results/llava-v1.6-test

# Cross-model executive summary:
python scripts/summarize.py results/*/report.csv
```

## Model suite

`scripts/fetch_models.sh` downloads the default suite (~80 GB at int8 quantisation):

```bash
make fetch-models
```

Default suite covers LLaVA-1.6, Qwen2-VL, Llama-3.2-Vision, MiniCPM-V 2.6, Pixtral, Phi-3.5-Vision, SmolVLM, and Idefics3.

## Architecture overview

The profiler injects a **stats backend wrapper** into `llama.cpp` at runtime (via `DYLD_INSERT_LIBRARIES` on macOS, `LD_PRELOAD` on Linux). The wrapper implements `ggml_backend_i`, holds a pointer to the real inner backend (CPU / Metal / CUDA), and in `graph_compute` walks every node in the `ggml_cgraph` to record statistics before forwarding the graph to the inner backend.

See [docs/design.md](docs/design.md) for the full design and alternatives considered.

## Build dependencies

| Dependency | Version | Notes |
|-----------|---------|-------|
| Docker | ≥ 20.10 | **Recommended** — bundles all build deps |
| clang / gcc | — | C++17; only needed for local build |
| cmake | ≥ 3.22 | Only needed for local build |
| Python | ≥ 3.11 | For aggregation scripts |
| pandas / numpy / pyarrow / jinja2 | see `requirements.txt` | |

## Makefile targets

| Target | Action |
|--------|--------|
| `make docker-build` | Build the Docker image (primary path) |
| `make docker-run` | Run `vlm-op-profiler --help` in Docker |
| `make docker-shell` | Interactive shell in the container |
| `make docker-clean` | Remove the Docker image |
| `make setup` | Local: submodule init + cmake configure + `.venv` |
| `make build` | Local: `cmake --build build -j` |
| `make fetch-models` | Run `scripts/fetch_models.sh` |
| `make run-suite` | Run profiler across default model × prompt × image matrix |
| `make aggregate` | Run `scripts/aggregate.py` over `results/` |
| `make test` | Run Python unit tests (`pytest`) |
| `make clean` | Remove `build/` and `.venv/` |

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

## Conventions

- All code, comments, commits, and docs in **English**.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/).
- C++: C++17, LLVM `clang-format` style, no exceptions in the backend hot path.
- Python: `ruff` lint/format, type hints on public functions, `pytest`.
- `models/` and `results/` are gitignored — never commit weights or run outputs.
