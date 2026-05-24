#!/usr/bin/env bash
# run_suite.sh
#
# Execute vlm-op-profiler across the default model × prompt × image matrix.
# Each combination is run independently so individual runs can be re-triggered.
#
# Usage:
#   bash scripts/run_suite.sh [options]
#
# Options:
#   --backend-lib <path>   Path to libbackend_stats.dylib / .so
#   --cli <path>           Path to vlm-op-profiler binary
#   --out-dir <path>       Root output directory (default: results)
#   --steps <N>            Max decode tokens per run (default: 64)
#   --model <name>         Run only this model (matches basename without extension)
#   --dry-run              Print commands without executing them

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BACKEND_LIB="build/libbackend_stats.dylib"
if [[ "$(uname)" != "Darwin" ]]; then BACKEND_LIB="build/libbackend_stats.so"; fi
CLI="build/vlm-op-profiler"
OUT_DIR="results"
STEPS=64
FILTER_MODEL=""
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend-lib) BACKEND_LIB="$2"; shift 2 ;;
        --cli)         CLI="$2";         shift 2 ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        --steps)       STEPS="$2";       shift 2 ;;
        --model)       FILTER_MODEL="$2";shift 2 ;;
        --dry-run)     DRY_RUN=true;     shift   ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# *//'
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
if [[ ! -f "$BACKEND_LIB" ]]; then
    echo "ERROR: backend library not found: $BACKEND_LIB"
    echo "Run 'make build' first."
    exit 1
fi

if [[ ! -f "$CLI" ]]; then
    echo "ERROR: CLI binary not found: $CLI"
    echo "Run 'make build' first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Model, prompt, and image matrix
# ---------------------------------------------------------------------------
MODELS=(
    "models/SmolVLM-Instruct-Q4_K_M.gguf"
    "models/ggml-model-q4_k.gguf"           # LLaVA-1.6
    "models/qwen2-vl-7b-instruct-q4_k_m.gguf"
    "models/Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf"
    "models/MiniCPM-V-2_6-Q4_K_M.gguf"
    "models/Phi-3.5-vision-instruct-Q4_K_M.gguf"
    "models/pixtral-12b-Q4_K_M.gguf"
    "models/Idefics3-8B-Llama3-Q4_K_M.gguf"
)

# Representative prompts
PROMPTS=(
    "Describe this image in detail."
    "What objects are visible in the image?"
    "Summarise what is happening in the scene."
)

# Representative test images (shipped in docs/test_images/ or downloaded separately)
IMAGES=(
    "docs/test_images/natural_scene.jpg"
    "docs/test_images/document_page.png"
    "docs/test_images/chart.png"
)

# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------
run_count=0
skip_count=0
timestamp=$(date -u +"%Y%m%dT%H%M%SZ")

for model_path in "${MODELS[@]}"; do
    [[ -f "$model_path" ]] || { echo "[skip] model not found: $model_path"; ((skip_count++)); continue; }
    model_name=$(basename "$model_path" .gguf)
    [[ -n "$FILTER_MODEL" && "$model_name" != *"$FILTER_MODEL"* ]] && continue

    for prompt in "${PROMPTS[@]}"; do
        for image_path in "${IMAGES[@]}"; do
            [[ -f "$image_path" ]] || { echo "[skip] image not found: $image_path"; continue; }

            prompt_slug=$(echo "$prompt" | tr ' ' '_' | tr -dc '[:alnum:]_' | head -c 20)
            image_slug=$(basename "$image_path" | cut -d. -f1)
            run_id="${timestamp}_${prompt_slug}_${image_slug}"
            run_out="${OUT_DIR}/${model_name}/${run_id}"

            if $DRY_RUN; then
                echo "[dry-run] $model_name / $prompt_slug / $image_slug"
                echo "  out: $run_out"
                continue
            fi

            echo "==> $model_name | $image_slug | $prompt_slug"
            mkdir -p "$run_out"

            UNAME=$(uname)
            if [[ "$UNAME" == "Darwin" ]]; then
                INJECT_VAR="DYLD_INSERT_LIBRARIES"
            else
                INJECT_VAR="LD_PRELOAD"
            fi

            env \
                "$INJECT_VAR"="$(realpath "$BACKEND_LIB")" \
                PROFSTATS_OUT_DIR="$(realpath "$run_out")" \
                PROFSTATS_MAX_STEPS="$STEPS" \
                "$CLI" \
                    --model  "$model_path" \
                    --image  "$image_path" \
                    --out-dir "$run_out" \
                    --steps  "$STEPS" \
                    "$prompt" \
            && echo "    [ok]" \
            || echo "    [FAILED] $model_name / $prompt_slug / $image_slug"

            ((run_count++))
        done
    done
done

echo ""
echo "Done: $run_count runs, $skip_count models skipped."
