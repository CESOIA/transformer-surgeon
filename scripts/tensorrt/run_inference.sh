#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER="$SCRIPT_DIR/inference_exported_test.py"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-artifacts}"
PROMPT="${PROMPT:-Tell me one short fact about France.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

ENGINE_FILES=(
    "$OUT_DIR/qwen2_tensorrt_full_seed0.pt2"
    "$OUT_DIR/qwen2_tensorrt_random_seed1.pt2"
    "$OUT_DIR/qwen2_tensorrt_random_seed2.pt2"
    "$OUT_DIR/qwen2_tensorrt_random_seed3.pt2"
)

for engine in "${ENGINE_FILES[@]}"; do
    if [[ ! -f "$engine" ]]; then
        echo "--- SKIP (not found): $engine ---"
        echo ""
        continue
    fi
    echo "=== Running inference: $(basename "$engine") ==="
    python "$INFER" \
        --engine-path "$engine" \
        --model-name "$MODEL_NAME" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_NEW_TOKENS"
    echo ""
done
