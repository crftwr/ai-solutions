################################################################################
# vlm-op-profiler — Makefile
# All paths relative to vlm_op_profiler/ (this file's directory).
#
# WORKFLOW (Docker):
#   make docker-build    Build the Docker image
#   make docker-run      vlm-op-profiler --help inside the image
#   make docker-shell    Drop into the container for exploration
################################################################################

SHELL   := /bin/bash
REPO_ROOT := $(shell git -C "$(dir $(abspath $(lastword $(MAKEFILE_LIST))))" rev-parse --show-toplevel)

# Load .env if present (for HF_TOKEN and other secrets)
-include .env
export HF_TOKEN

# Docker
DOCKER_IMAGE  ?= vlm-op-profiler
DOCKER_TAG    ?= latest
MODELS_DIR    ?= $(abspath models)
RESULTS_DIR   ?= $(abspath results)

.PHONY: all docker-build docker-run docker-shell docker-clean \
        fetch-models fetch-models-suite fetch-model-text \
        gen-test-images smoke-test smoke-test-text run-suite aggregate summarize \
        validate regression-test test lint help

all: docker-build

# ---------------------------------------------------------------------------
## Docker targets (primary workflow)
# ---------------------------------------------------------------------------

## docker-build: build the Docker image (includes llama.cpp + backend + CLI)
##   Bakes the current profiler + llama.cpp commit SHAs into the image so
##   that run_meta.json reports correct provenance at runtime.
PROFILER_COMMIT := $(shell git -C "$(REPO_ROOT)" rev-parse HEAD 2>/dev/null || echo unknown)
LLAMA_COMMIT    := $(shell git -C "$(REPO_ROOT)/vlm_op_profiler/third_party/llama.cpp" rev-parse HEAD 2>/dev/null || echo unknown)

docker-build:
	docker build \
	    --build-arg PROFILER_COMMIT=$(PROFILER_COMMIT) \
	    --build-arg LLAMA_COMMIT=$(LLAMA_COMMIT) \
	    -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

## docker-run: run vlm-op-profiler --help inside the container
docker-run: docker-build
	docker run --rm \
	    -v "$(MODELS_DIR):/app/models:ro" \
	    -v "$(RESULTS_DIR):/app/results" \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) $(ARGS)

## docker-shell: interactive shell in the container for debugging
docker-shell: docker-build
	docker run --rm -it \
	    -v "$(MODELS_DIR):/app/models:ro" \
	    -v "$(RESULTS_DIR):/app/results" \
	    --entrypoint /bin/bash \
	    $(DOCKER_IMAGE):$(DOCKER_TAG)

## docker-clean: remove the Docker image
docker-clean:
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true

# ---------------------------------------------------------------------------
## Model & suite targets
# ---------------------------------------------------------------------------

## fetch-models: download SmolVLM (minimal smoke-test VLM, ~2 GB) via Docker.
##   Uses ggml-org/SmolVLM-Instruct-GGUF which is public (no HF token needed).
##   Downloads both the main model and the mmproj (vision projector) file.
##   Reads HF_TOKEN from .env if present (needed for gated models like LLaVA).
HF_TOKEN_ARG := $(if $(HF_TOKEN),-e HF_TOKEN=$(HF_TOKEN),)
SMOLVLM_MODEL  := SmolVLM-Instruct-Q4_K_M.gguf
SMOLVLM_MMPROJ := mmproj-SmolVLM-Instruct-f16.gguf
SMOLVLM_REPO   := ggml-org/SmolVLM-Instruct-GGUF

fetch-models:
	docker run --rm \
	    -v "$(MODELS_DIR):/models" \
	    $(HF_TOKEN_ARG) \
	    --entrypoint bash \
	    python:3.12-slim \
	    -c "pip install -q 'huggingface_hub' && \
	        mkdir -p /models && \
	        python3 -c \"\
import os, huggingface_hub as hf; \
tok = os.environ.get('HF_TOKEN') or None; \
repo = '$(SMOLVLM_REPO)'; \
print('Downloading $(SMOLVLM_MODEL)...'); \
hf.hf_hub_download(repo, '$(SMOLVLM_MODEL)', local_dir='/models', token=tok); \
print('Downloading $(SMOLVLM_MMPROJ)...'); \
hf.hf_hub_download(repo, '$(SMOLVLM_MMPROJ)', local_dir='/models', token=tok); \
print('Done.')\" && \
	        echo 'Models saved to $(MODELS_DIR)/'"

## fetch-models-suite: download the VLM model suite (SUITE=minimal|default|full)
##   Uses the project's Docker image so huggingface_hub matches requirements.txt.
##   Reads HF_TOKEN from .env for gated repos (e.g. Llama-3.2-Vision).
fetch-models-suite: docker-build
	docker run --rm \
	    -v "$(MODELS_DIR):/app/models" \
	    $(HF_TOKEN_ARG) \
	    --entrypoint /app/.venv/bin/python \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/fetch_models_hf.py \
	        --suite $${SUITE:-default} \
	        --out-dir /app/models

## smoke-test: end-to-end test with SmolVLM (downloads model first if needed)
##   Runs with --steps 2 to limit decode graphs, -n 20 to cap output tokens.
##   Verifies that trace.jsonl is produced under results/smoke-test/.
SMOKE_MODEL  ?= $(MODELS_DIR)/$(SMOLVLM_MODEL)
SMOKE_MMPROJ ?= $(MODELS_DIR)/$(SMOLVLM_MMPROJ)
SMOKE_IMAGE  ?= $(abspath assets/example_64.jpg)
SMOKE_PROMPT ?= "Describe this image briefly."
SMOKE_OUT    ?= $(RESULTS_DIR)/smoke-test

smoke-test: docker-build
	@echo "==> Smoke test: checking files..."
	@test -f "$(SMOKE_MODEL)"  || { echo "Model not found: $(SMOKE_MODEL)"; echo "Run: make fetch-models"; exit 1; }
	@test -f "$(SMOKE_MMPROJ)" || { echo "mmproj not found: $(SMOKE_MMPROJ)"; echo "Run: make fetch-models"; exit 1; }
	@test -f "$(SMOKE_IMAGE)"  || { echo "Image not found: $(SMOKE_IMAGE)"; echo "Run: make gen-test-images"; exit 1; }
	@mkdir -p "$(SMOKE_OUT)"
	@rm -f "$(SMOKE_OUT)/trace.jsonl" "$(SMOKE_OUT)/run_meta.json"
	# Tolerate non-zero exit from llama-mtmd-cli — known SmolVLM token-ID
	# issue still produces a valid warmup trace + run_meta.json. See
	# docs/supported_models.md § SmolVLM-Instruct.
	docker run --rm \
	    -v "$(MODELS_DIR):/app/models:ro" \
	    -v "$(RESULTS_DIR):/app/results" \
	    -v "$(abspath assets):/app/assets:ro" \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    --out-dir /app/results/smoke-test \
	    --steps 2 \
	    --include-vision-encode \
	    --model  /app/models/$(notdir $(SMOKE_MODEL)) \
	    --mmproj /app/models/$(notdir $(SMOKE_MMPROJ)) \
	    --image  /app/assets/example_64.jpg \
	    -n 20 \
	    -p $(SMOKE_PROMPT); true
	@echo ""
	@echo "==> Checking trace.jsonl..."
	@test -f "$(SMOKE_OUT)/trace.jsonl" && \
	    echo "  trace.jsonl exists (lines: $$(wc -l < $(SMOKE_OUT)/trace.jsonl))" || \
	    { echo "FAIL: trace.jsonl NOT found in $(SMOKE_OUT)/"; exit 1; }
	@echo "==> Checking run_meta.json..."
	@test -f "$(SMOKE_OUT)/run_meta.json" || \
	    { echo "FAIL: run_meta.json NOT found in $(SMOKE_OUT)/"; exit 1; }
	@python3 -c "import json,sys; \
	    m=json.load(open('$(SMOKE_OUT)/run_meta.json')); \
	    req=['profiler_version','profiler_commit','llama_commit','model_path','model_sha256','image_path','image_sha256','run_id','platform','inner_backend','include_vision_encode']; \
	    missing=[k for k in req if k not in m]; \
	    sys.exit(f'missing keys: {missing}') if missing else print('  run_meta.json has all required keys; profiler_commit=%s llama_commit=%s' % (m['profiler_commit'][:12], m['llama_commit'][:12]))"
	@echo "==> Smoke test PASSED."

## fetch-model-text: download Llama-3.2-1B (public, 800 MB) for text-only smoke tests
TEXT_MODEL      := Llama-3.2-1B-Instruct-IQ3_M.gguf
TEXT_MODEL_REPO := bartowski/Llama-3.2-1B-Instruct-GGUF

fetch-model-text:
	docker run --rm \
	    -v "$(MODELS_DIR):/models" \
	    $(HF_TOKEN_ARG) \
	    --entrypoint bash \
	    python:3.12-slim \
	    -c "pip install -q 'huggingface_hub' && \
	        mkdir -p /models && \
	        python3 -c \"\
import os, huggingface_hub as hf; \
tok = os.environ.get('HF_TOKEN') or None; \
print('Downloading $(TEXT_MODEL)...'); \
hf.hf_hub_download('$(TEXT_MODEL_REPO)', '$(TEXT_MODEL)', local_dir='/models', token=tok); \
print('Done.')\" && \
	        echo 'Model saved to $(MODELS_DIR)/$(TEXT_MODEL)'"

## smoke-test-text: Phase 1 validation using llama-cli + a text model
##   Shows BOTH prefill and decode phases in trace.jsonl.
##   Requires: make fetch-model-text  (and make docker-build)
TEXT_SMOKE_MODEL  ?= $(MODELS_DIR)/$(TEXT_MODEL)
TEXT_SMOKE_OUT    ?= $(RESULTS_DIR)/smoke-test-text

smoke-test-text: docker-build
	@echo "==> Text smoke test: checking model..."
	@test -f "$(TEXT_SMOKE_MODEL)" || { \
	    echo "Model not found: $(TEXT_SMOKE_MODEL)"; \
	    echo "Run: make fetch-model-text"; \
	    exit 1; }
	@mkdir -p "$(TEXT_SMOKE_OUT)"
	@rm -f "$(TEXT_SMOKE_OUT)/trace.jsonl"
	# Send prompt then /exit so llama-cli completes generation and exits cleanly.
	printf '/exit\n' | docker run --rm -i \
	    -v "$(MODELS_DIR):/app/models:ro" \
	    -v "$(RESULTS_DIR):/app/results" \
	    --env PROFSTATS_OUT_DIR=/app/results/smoke-test-text \
	    --env PROFSTATS_MAX_STEPS=2 \
	    --env LD_PRELOAD=/app/libbackend_stats.so \
	    --env LD_LIBRARY_PATH=/app \
	    --entrypoint /app/llama-cli \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	        --model /app/models/$(TEXT_MODEL) \
	        -p "The capital of France is" \
	        -n 10; true
	@echo ""
	@echo "==> Checking trace.jsonl..."
	@test -f "$(TEXT_SMOKE_OUT)/trace.jsonl" && { \
	    LINES=$$(wc -l < "$(TEXT_SMOKE_OUT)/trace.jsonl"); \
	    PHASES=$$(python3 -c "import json; recs=[json.loads(l) for l in open('$(TEXT_SMOKE_OUT)/trace.jsonl')]; \
	        from collections import Counter; p=Counter(r['phase'] for r in recs); print(dict(p))"); \
	    echo "  trace.jsonl: $${LINES} lines, phases=$${PHASES}"; \
	} || { echo "FAIL: trace.jsonl NOT found in $(TEXT_SMOKE_OUT)/"; exit 1; }
	@echo "==> Text smoke test PASSED."

## gen-test-images: render the synthetic JPEGs used by run-suite into assets/.
##   Outputs are checked into git so smoke-test works without re-running this;
##   re-run only after intentionally changing the generator.
gen-test-images: docker-build
	docker run --rm \
	    -v "$(abspath assets):/app/assets" \
	    --entrypoint /app/.venv/bin/python \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/generate_test_images.py --out-dir /app/assets

## run-suite: run profiler over default model × prompt × image matrix.
##   Per-run outputs land under results/<model>/<run_id>/{trace.jsonl,run_meta.json}.
##   Pass extra args via SUITE_ARGS (e.g. SUITE_ARGS="--only SmolVLM --dry-run").
run-suite: docker-build
	docker run --rm \
	    -v "$(MODELS_DIR):/app/models:ro" \
	    -v "$(RESULTS_DIR):/app/results" \
	    -v "$(abspath assets):/app/assets:ro" \
	    --entrypoint /app/.venv/bin/python \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/run_suite.py \
	        --models-dir /app/models \
	        --images-dir /app/assets \
	        --out-root  /app/results \
	        --profiler  /app/vlm_op_profiler.py \
	        --include-vision-encode \
	        $(SUITE_ARGS)

## aggregate: aggregate trace.jsonl files into report.csv / report.md.
##   Also writes per-run reports next to each trace.jsonl (--per-run).
##   Pass AGGREGATE_ARGS="..." to forward extra args.
aggregate: docker-build
	docker run --rm \
	    -v "$(RESULTS_DIR):/app/results" \
	    --entrypoint /app/.venv/bin/python \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/aggregate.py /app/results --per-run $(AGGREGATE_ARGS)

## summarize: roll up every per-run report.csv into summary.csv/.md.
##   Run `make aggregate` first to produce the per-run CSVs at
##   results/<model>/<run_id>/report.csv.
summarize: docker-build
	docker run --rm \
	    -v "$(RESULTS_DIR):/app/results" \
	    --entrypoint bash \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    -lc 'shopt -s nullglob globstar; \
	         files=(/app/results/**/report.csv); \
	         filtered=(); \
	         for f in "$${files[@]}"; do [ "$$f" = "/app/results/report.csv" ] || filtered+=("$$f"); done; \
	         if [ $${#filtered[@]} -eq 0 ]; then \
	             echo "No per-run /app/results/<model>/<run_id>/report.csv — run \"make aggregate\" first."; \
	             exit 1; \
	         fi; \
	         /app/.venv/bin/python scripts/summarize.py --out /app/results "$${filtered[@]}"'

# ---------------------------------------------------------------------------
## Dev / test targets
# ---------------------------------------------------------------------------

## test: run fast Python unit tests (no model inference needed).
test: docker-build
	docker run --rm \
	    --entrypoint /app/.venv/bin/pytest \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/tests/ -v --deselect scripts/tests/test_regression.py

## validate: run structural + analytical MAC cross-checks on an existing trace.
##   Pass VALIDATE_PATH and VALIDATE_ARCH:
##     make validate VALIDATE_PATH=/app/results/smoke-test-text VALIDATE_ARCH=llama-3.2-1b
VALIDATE_PATH ?= /app/results
VALIDATE_ARGS ?=
validate: docker-build
	docker run --rm \
	    -v "$(RESULTS_DIR):/app/results" \
	    --entrypoint /app/.venv/bin/python \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/validate.py $(VALIDATE_PATH) $(if $(VALIDATE_ARCH),--arch $(VALIDATE_ARCH)) $(VALIDATE_ARGS)

## regression-test: end-to-end gate, suitable for every commit.
##   Runs the text smoke-test (fast — ~1 GB model, ~30s), then validate.py +
##   pytest scripts/tests/test_regression.py against the resulting trace.jsonl.
##   Requires `make fetch-model-text` to have populated the model once.
regression-test: docker-build smoke-test-text
	@echo "==> regression-test: validating trace.jsonl ..."
	docker run --rm \
	    -v "$(RESULTS_DIR):/app/results" \
	    --entrypoint /app/.venv/bin/python \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/validate.py /app/results/smoke-test-text/trace.jsonl \
	        --arch llama-3.2-1b
	@echo "==> regression-test: pytest invariants ..."
	docker run --rm \
	    -v "$(RESULTS_DIR):/app/results" \
	    --entrypoint /app/.venv/bin/pytest \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/tests/test_regression.py -v
	@echo "==> regression-test PASSED."

## lint: run ruff on Python sources (inside Docker)
lint:
	docker run --rm \
	    --entrypoint /app/.venv/bin/ruff \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) check scripts/

## help: print this message
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
