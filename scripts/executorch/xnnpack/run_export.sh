#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORTER="$SCRIPT_DIR/exporter_function_test.py"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-artifacts}"
MODE="${MODE:-hf}"

echo "=== [1/6] xnnpack full (no quantization) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --mode "$MODE"

echo ""
echo "=== [2/6] xnnpack w8 (weight-only INT8) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --mode "$MODE" \
    --quant-mlp \
    --quant-precision 8 \
    --quant-mode hard

echo ""
echo "=== [3/6] xnnpack a8w8 (weight INT8 + activation INT8, WikiText-2 calibration) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --mode "$MODE" \
    --quant-mlp \
    --quant-precision 8 \
    --quant-mode hard \
    --calibrate \
    --act-precision 8

echo ""
echo "=== [4/6] xnnpack w4 (weight-only INT4) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --mode "$MODE" \
    --quant-mlp \
    --quant-precision 4 \
    --quant-mode hard

echo ""
echo "=== [5/6] xnnpack a8w4 (weight INT4 + activation INT8, WikiText-2 calibration) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --mode "$MODE" \
    --quant-mlp \
    --quant-precision 4 \
    --quant-mode hard \
    --calibrate \
    --act-precision 8

echo ""
echo "=== [6/6] xnnpack a4w4 (weight INT4 + activation INT4, WikiText-2 calibration) ==="
python "$EXPORTER" \
    --model-name "$MODEL_NAME" \
    --out-dir "$OUT_DIR" \
    --mode "$MODE" \
    --quant-mlp \
    --quant-precision 4 \
    --quant-mode hard \
    --calibrate \
    --act-precision 4

echo ""
echo "All exports done. Artifacts in: $OUT_DIR"
