#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER="$SCRIPT_DIR/inference_exported_test.py"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-artifacts}"
PROMPT="${PROMPT:-Tell me one short fact about France.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

# KV-cache geometry for the io_* engines, matching Qwen2-0.5B-Instruct's config
# (hidden_size=896, num_attention_heads=14 -> head_dim=64, num_key_value_heads=2,
# num_hidden_layers=24). Override if MODEL_NAME points at a different model.
NUM_LAYERS="${NUM_LAYERS:-24}"
KV_NUM_HEADS="${KV_NUM_HEADS:-2}"
HEAD_DIM="${HEAD_DIM:-64}"
MAX_CACHE_LEN="${MAX_CACHE_LEN:-1024}"
CACHE_DTYPE="${CACHE_DTYPE:-float16}"

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

# --- inout KV-cache engines (cache-impl=io_scatter / io_concat) ---
IO_ENGINE_FILES=(
    "$OUT_DIR/qwen2_tensorrt_full_seed0_io_scatter.pt2:io_scatter"
    "$OUT_DIR/qwen2_tensorrt_full_seed0_io_concat.pt2:io_concat"
)

for entry in "${IO_ENGINE_FILES[@]}"; do
    engine="${entry%%:*}"
    cache_impl="${entry##*:}"
    if [[ ! -f "$engine" ]]; then
        echo "--- SKIP (not found): $engine ---"
        echo ""
        continue
    fi
    echo "=== Running inference: $(basename "$engine") (cache-impl=$cache_impl) ==="
    python "$INFER" \
        --engine-path "$engine" \
        --model-name "$MODEL_NAME" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --cache-impl "$cache_impl" \
        --num-layers "$NUM_LAYERS" \
        --kv-num-heads "$KV_NUM_HEADS" \
        --head-dim "$HEAD_DIM" \
        --max-cache-len "$MAX_CACHE_LEN" \
        --cache-dtype "$CACHE_DTYPE"
    echo ""
done
