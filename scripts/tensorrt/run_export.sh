#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORTER="$SCRIPT_DIR/exporter_function_test.py"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-artifacts}"
DEVICE="${DEVICE:-cuda:0}"

echo "=== [1/4] tensorrt full (no compression, baseline) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --full-only \
    --seed 0 \
    --verbose

echo ""
echo "=== [2/4] tensorrt random mixed compression (seed 1) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --seed 1 \
    --verbose

echo ""
echo "=== [3/4] tensorrt random mixed compression (seed 2) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --seed 2 \
    --verbose

echo ""
echo "=== [4/4] tensorrt random mixed compression, int4 allowed (seed 3) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --device "$DEVICE" \
    --seed 3 \
    --allow-int4 \
    --verbose

echo ""
echo "All exports done. Artifacts in: $OUT_DIR"
