#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER="$SCRIPT_DIR/inference_exported_test.py"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-artifacts}"
PROMPT="${PROMPT:-Tell me one short fact about France.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

PTE_FILES=(
    "$OUT_DIR/qwen2_xnnpack_full.pte"
    "$OUT_DIR/qwen2_xnnpack_w8_hard.pte"
    "$OUT_DIR/qwen2_xnnpack_w8_a8_hard.pte"
    "$OUT_DIR/qwen2_xnnpack_w4_hard.pte"
    "$OUT_DIR/qwen2_xnnpack_w4_a8_hard.pte"
    "$OUT_DIR/qwen2_xnnpack_w4_a4_hard.pte"
)

for pte in "${PTE_FILES[@]}"; do
    if [[ ! -f "$pte" ]]; then
        echo "--- SKIP (not found): $pte ---"
        echo ""
        continue
    fi
    echo "=== Running inference: $(basename "$pte") ==="
    python "$INFER" \
        --pte-path "$pte" \
        --model-name "$MODEL_NAME" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_NEW_TOKENS"
    echo ""
done
