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
        smoke-test smoke-test-text run-suite aggregate test lint help

all: docker-build

# ---------------------------------------------------------------------------
## Docker targets (primary workflow)
# ---------------------------------------------------------------------------

## docker-build: build the Docker image (includes llama.cpp + backend + CLI)
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

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

## fetch-models-suite: download full VLM suite (uses SUITE env var)
fetch-models-suite:
	docker run --rm \
	    -v "$(MODELS_DIR):/models" \
	    -v "$(abspath scripts):/scripts:ro" \
	    $(HF_TOKEN_ARG) \
	    --entrypoint bash \
	    python:3.12-slim \
	    -c "pip install -q 'huggingface_hub' && \
	        python3 /scripts/fetch_models_hf.py --suite $${SUITE:-default} --out-dir /models"

## smoke-test: end-to-end test with SmolVLM (downloads model first if needed)
##   Runs with --steps 2 to limit decode graphs, -n 20 to cap output tokens.
##   Verifies that trace.jsonl is produced under results/smoke-test/.
SMOKE_MODEL  ?= $(MODELS_DIR)/$(SMOLVLM_MODEL)
SMOKE_MMPROJ ?= $(MODELS_DIR)/$(SMOLVLM_MMPROJ)
SMOKE_IMAGE  ?= $(abspath assets/example.jpg)
SMOKE_PROMPT ?= "Describe this image briefly."
SMOKE_OUT    ?= $(RESULTS_DIR)/smoke-test

smoke-test: docker-build
	@echo "==> Smoke test: checking files..."
	@test -f "$(SMOKE_MODEL)"  || { echo "Model not found: $(SMOKE_MODEL)"; echo "Run: make fetch-models"; exit 1; }
	@test -f "$(SMOKE_MMPROJ)" || { echo "mmproj not found: $(SMOKE_MMPROJ)"; echo "Run: make fetch-models"; exit 1; }
	@test -f "$(SMOKE_IMAGE)"  || { echo "Image not found: $(SMOKE_IMAGE)"; echo "Add an image to assets/example.jpg"; exit 1; }
	@mkdir -p "$(SMOKE_OUT)"
	docker run --rm \
	    -v "$(MODELS_DIR):/app/models:ro" \
	    -v "$(RESULTS_DIR):/app/results" \
	    -v "$(abspath assets):/app/assets:ro" \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    --out-dir /app/results/smoke-test \
	    --steps 2 \
	    --model  /app/models/$(notdir $(SMOKE_MODEL)) \
	    --mmproj /app/models/$(notdir $(SMOKE_MMPROJ)) \
	    --image  /app/assets/example.jpg \
	    -n 20 \
	    -p $(SMOKE_PROMPT)
	@echo ""
	@echo "==> Checking trace.jsonl..."
	@test -f "$(SMOKE_OUT)/trace.jsonl" && \
	    echo "  trace.jsonl exists (lines: $$(wc -l < $(SMOKE_OUT)/trace.jsonl))" || \
	    { echo "FAIL: trace.jsonl NOT found in $(SMOKE_OUT)/"; exit 1; }
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

## run-suite: run profiler over default model × prompt × image matrix
run-suite:
	docker run --rm \
	    -v "$(MODELS_DIR):/app/models:ro" \
	    -v "$(RESULTS_DIR):/app/results" \
	    --entrypoint bash \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/run_suite.sh \
	        --backend-lib /app/libbackend_stats.so \
	        --cli /app/llama-mtmd-cli \
	        --out-dir /app/results

## aggregate: aggregate trace.jsonl files into report.csv / report.md
aggregate:
	docker run --rm \
	    -v "$(RESULTS_DIR):/app/results" \
	    --entrypoint /app/.venv/bin/python \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/aggregate.py /app/results

# ---------------------------------------------------------------------------
## Dev / test targets
# ---------------------------------------------------------------------------

## test: run Python unit tests (inside Docker)
test: docker-build
	docker run --rm \
	    --entrypoint /app/.venv/bin/pytest \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) \
	    scripts/tests/ -v

## lint: run ruff on Python sources (inside Docker)
lint:
	docker run --rm \
	    --entrypoint /app/.venv/bin/ruff \
	    $(DOCKER_IMAGE):$(DOCKER_TAG) check scripts/

## help: print this message
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
