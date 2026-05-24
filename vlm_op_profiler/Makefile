################################################################################
# vlm-op-profiler — Makefile
# All paths relative to vlm_op_profiler/ (this file's directory).
#
# PRIMARY WORKFLOW (Docker — no local cmake required):
#   make docker-build    Build the Docker image
#   make docker-run      vlm-op-profiler --help inside the image
#   make docker-shell    Drop into the container for exploration
#
# LOCAL WORKFLOW (requires cmake ≥ 3.22 and clang/gcc installed):
#   make setup           git submodule update + cmake configure + .venv
#   make build           cmake --build
################################################################################

SHELL   := /bin/bash
REPO_ROOT := $(shell git -C "$(dir $(abspath $(lastword $(MAKEFILE_LIST))))" rev-parse --show-toplevel)

VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff

BUILD_DIR   := build
BUILD_TYPE  ?= Release
JOBS        ?= $(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# Detect OS for dynamic library extension
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	LIB_EXT := dylib
	INJECT_VAR := DYLD_INSERT_LIBRARIES
else
	LIB_EXT := so
	INJECT_VAR := LD_PRELOAD
endif

BACKEND_LIB := $(BUILD_DIR)/libbackend_stats.$(LIB_EXT)
CLI_BIN     := $(BUILD_DIR)/vlm-op-profiler

# Docker
DOCKER_IMAGE  ?= vlm-op-profiler
DOCKER_TAG    ?= latest
MODELS_DIR    ?= $(abspath models)
RESULTS_DIR   ?= $(abspath results)

.PHONY: all docker-build docker-run docker-shell docker-clean \
        setup build fetch-models run-suite aggregate test lint clean help

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
## Local build targets (requires cmake ≥ 3.22 locally)
# ---------------------------------------------------------------------------

## setup: initialise submodule, configure CMake, create Python venv
setup:
	@echo "==> Initialising llama.cpp submodule..."
	git -C "$(REPO_ROOT)" submodule update --init --recursive
	@echo "==> Configuring CMake..."
	cmake -S . -B $(BUILD_DIR) \
	      -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	      -DGGML_BACKEND_DL=ON \
	      -DLLAMA_BUILD_TESTS=OFF \
	      -DLLAMA_BUILD_EXAMPLES=ON
	@echo "==> Creating Python venv..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "==> Setup complete. Run 'make build' next."

## build: compile backend_stats shared library and CLI binary (local)
build:
	cmake --build $(BUILD_DIR) -j $(JOBS)

# ---------------------------------------------------------------------------
## Model & suite targets
# ---------------------------------------------------------------------------

## fetch-models: download default VLM suite into models/
fetch-models:
	bash scripts/fetch_models.sh

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

## lint: run ruff on Python sources (local venv or Docker)
lint:
	@if [ -x "$(RUFF)" ]; then \
	    $(RUFF) check scripts/; \
	else \
	    docker run --rm \
	        --entrypoint /app/.venv/bin/ruff \
	        $(DOCKER_IMAGE):$(DOCKER_TAG) check scripts/; \
	fi

## clean: remove local build artefacts and Python venv
clean:
	rm -rf $(BUILD_DIR) $(VENV)

## help: print this message
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
