#!/usr/bin/env bash
# fetch_models.sh
#
# Download the default VLM GGUF model suite into models/.
# Requires: huggingface-cli (pip install huggingface_hub[cli])
#
# Usage:
#   bash scripts/fetch_models.sh [--suite minimal|default|full]
#
# Suites:
#   minimal  — SmolVLM only (~3 GB); for CI / quick smoke tests
#   default  — 8 architecturally diverse models (~80 GB at int8)
#   full     — default + larger variants (~160 GB)

set -euo pipefail
cd "$(dirname "$0")/.."

SUITE="${SUITE:-default}"
MODEL_DIR="models"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) SUITE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# *//'
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------
if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found."
    echo "Install with: pip install 'huggingface_hub[cli]'"
    exit 1
fi

mkdir -p "$MODEL_DIR"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
# Each entry: "repo_id  filename  suite_level"
# suite_level: minimal | default | full
MODELS=(
    # SmolVLM — smallest; used for CI smoke tests
    "HuggingFaceTB/SmolVLM-Instruct-GGUF  SmolVLM-Instruct-Q4_K_M.gguf  minimal"

    # LLaVA-1.6 Mistral 7B
    "cmp-nct/gguf-llava-v1.6-mistral-7b  ggml-model-q4_k.gguf  default"

    # Qwen2-VL 7B
    "Qwen/Qwen2-VL-7B-Instruct-GGUF  qwen2-vl-7b-instruct-q4_k_m.gguf  default"

    # Llama 3.2 Vision 11B
    "lmstudio-community/Llama-3.2-11B-Vision-Instruct-GGUF  Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf  default"

    # MiniCPM-V 2.6
    "openbmb/MiniCPM-V-2_6-gguf  MiniCPM-V-2_6-Q4_K_M.gguf  default"

    # Phi-3.5 Vision
    "bartowski/Phi-3.5-vision-instruct-GGUF  Phi-3.5-vision-instruct-Q4_K_M.gguf  default"

    # Pixtral 12B
    "bartowski/pixtral-12b-GGUF  pixtral-12b-Q4_K_M.gguf  default"

    # Idefics3 (late-fusion contrast model)
    "bartowski/Idefics3-8B-Llama3-GGUF  Idefics3-8B-Llama3-Q4_K_M.gguf  default"
)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
downloaded=0
skipped=0

for entry in "${MODELS[@]}"; do
    read -r repo_id filename level <<< "$entry"

    # Check suite level
    if [[ "$SUITE" == "minimal" && "$level" != "minimal" ]]; then continue; fi
    if [[ "$SUITE" == "default" && "$level" == "full" ]];    then continue; fi

    dest="$MODEL_DIR/$filename"
    if [[ -f "$dest" ]]; then
        echo "[skip] $filename already exists"
        ((skipped++))
        continue
    fi

    echo "[download] $repo_id / $filename"
    huggingface-cli download "$repo_id" "$filename" --local-dir "$MODEL_DIR" --quiet
    echo "[ok] $filename"
    ((downloaded++))
done

echo ""
echo "Done: $downloaded downloaded, $skipped skipped."
echo "Models are in: $MODEL_DIR/"
