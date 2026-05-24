# ============================================================================
# vlm-op-profiler — multi-stage Dockerfile
#
# Stage 1 (builder): compiles llama.cpp + backend_stats + CLI + llama-mtmd-cli
# Stage 2 (runtime): ships the binaries, shared libs, and Python scripts
#
# Usage:
#   docker build -t vlm-op-profiler .
#   docker run --rm vlm-op-profiler --help
#   docker run --rm \
#     -v "$(pwd)/models:/app/models:ro" \
#     -v "$(pwd)/results:/app/results" \
#     vlm-op-profiler \
#     --out-dir /app/results/test \
#     --model   /app/models/llava-1.6.Q4_K_M.gguf \
#     --image   /app/models/example.jpg \
#     "Describe this image."
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy source (see .dockerignore for exclusions)
COPY . .

# Verify the llama.cpp submodule was initialised on the host
RUN test -f third_party/llama.cpp/CMakeLists.txt || \
    { echo "ERROR: llama.cpp submodule not initialised."; \
      echo "Run from the ai-solutions root: git submodule update --init --recursive"; \
      exit 1; }

# Configure — build backend_stats + llama-mtmd-cli + llama-cli.
# The vlm-op-profiler CLI is a Python script (cli/vlm_op_profiler.py) and
# needs no compilation step.
# LLAMA_BUILD_TOOLS=ON enables tools/ (contains llama-mtmd-cli and support libs).
# LLAMA_BUILD_COMMON=ON is a prerequisite for tools.
# LLAMA_BUILD_SERVER=ON enables tools/cli/ which builds llama-cli (text inference).
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_BACKEND_DL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_COMMON=ON \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++

# Build our C++ target (interceptor shared library only)
RUN cmake --build build -j"$(nproc)" \
      --target backend_stats

# Build llama-mtmd-cli (VLM inference binary we exec into)
RUN cmake --build build -j"$(nproc)" \
      --target llama-mtmd-cli

# Build llama-cli (standard text-only inference; useful for smoke tests)
RUN cmake --build build -j"$(nproc)" \
      --target llama-cli 2>/dev/null || true

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
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- CLI script -------------------------------------------------------------

# vlm-op-profiler is a Python script; no compiled binary needed.
# HERE (os.path.realpath(__file__) dir) resolves to /app, which is also where
# libbackend_stats.so and llama-mtmd-cli land — the script finds them automatically.
COPY --chmod=755 cli/vlm_op_profiler.py ./vlm_op_profiler.py

# Stats interceptor shared library (loaded via LD_PRELOAD)
COPY --from=builder /src/build/libbackend_stats.so ./libbackend_stats.so

# llama-mtmd-cli — the VLM inference binary
COPY --from=builder /src/build/bin/llama-mtmd-cli  ./llama-mtmd-cli

# llama-cli — standard text-only inference binary (useful for smoke tests)
# Silently skip if not present (tools list may vary by llama.cpp version).
RUN --mount=from=builder,source=/src/build/bin,target=/bb \
    cp /bb/llama-cli /app/llama-cli 2>/dev/null || true

# ---- All runtime shared libraries from build/bin ----------------------------
# Copy every .so* file (including versioned sonames like libggml.so.0) so that
# the dynamic linker can find them when LD_LIBRARY_PATH=/app is set.
# Also copies libbackend_stats.so and llama-mtmd-cli if already present there.
RUN --mount=from=builder,source=/src/build/bin,target=/bb \
    cp /bb/*.so*  /app/ 2>/dev/null || true; \
    cp /bb/*.a    /app/ 2>/dev/null || true

# libbackend_stats.so lives directly under build/ (not build/bin/)
COPY --from=builder /src/build/libbackend_stats.so ./libbackend_stats.so

# ---- Python scripts ---------------------------------------------------------
COPY scripts/      ./scripts/
COPY requirements.txt .
RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- Runtime config ---------------------------------------------------------
# Set LD_LIBRARY_PATH so all .so files in /app are found at runtime.
ENV LD_LIBRARY_PATH=/app

# Models and results directories are expected to be mounted at runtime.
VOLUME ["/app/models", "/app/results"]

ENTRYPOINT ["python3", "./vlm_op_profiler.py"]
CMD ["--help"]
