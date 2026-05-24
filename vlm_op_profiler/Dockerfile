# ============================================================================
# vlm-op-profiler — multi-stage Dockerfile
#
# Stage 1 (builder): compiles llama.cpp + backend_stats + CLI
# Stage 2 (runtime): minimal image with the built binaries + Python scripts
#
# Usage:
#   docker build -t vlm-op-profiler .
#   docker run --rm vlm-op-profiler vlm-op-profiler --help
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy the project (excluding build artefacts; see .dockerignore)
COPY . .

# Ensure the llama.cpp submodule is present.
# (The submodule should already be initialised on the host; COPY brings it in.)
RUN test -f third_party/llama.cpp/CMakeLists.txt || \
    { echo "ERROR: llama.cpp submodule not initialised."; \
      echo "Run: git submodule update --init --recursive"; exit 1; }

# Configure
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_BACKEND_DL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=ON \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++

# Build (backend_stats + vlm-op-profiler + llama-mtmd-cli)
RUN cmake --build build -j"$(nproc)" \
      --target backend_stats vlm-op-profiler

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Binaries and shared library from the builder stage
COPY --from=builder /src/build/vlm-op-profiler     ./vlm-op-profiler
COPY --from=builder /src/build/libbackend_stats.so ./libbackend_stats.so

# llama-mtmd-cli is only built when LLAMA_BUILD_EXAMPLES=ON and all its deps
# are present.  Copy it if it exists; skip gracefully otherwise (Phase 4 will
# need it, Phase 0 does not).
RUN --mount=from=builder,source=/src/build/bin,target=/build_bin \
    cp /build_bin/llama-mtmd-cli . 2>/dev/null || echo "[info] llama-mtmd-cli not found in build/bin — skipping (Phase 0 OK)"

# Python scripts and requirements
COPY scripts/ ./scripts/
COPY requirements.txt .
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

# Models and results directories are expected to be mounted
VOLUME ["/app/models", "/app/results"]

# Default: show help
ENTRYPOINT ["./vlm-op-profiler"]
CMD ["--help"]
