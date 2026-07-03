# TensorRT Export

Lowers to a TensorRT engine via `torch-tensorrt`'s Dynamo path. Reuses all the
backend-agnostic machinery in [`export.common`](common.md); only the PT2E
quantizer and the compile/save step are TensorRT-specific.

!!! note "Requirements"
    Requires the `tensorrt` extra (`pip install -e ".[tensorrt]"`) and a CUDA
    device. Tests live under `test/tensorrt_tests/`; CLI runner at
    `scripts/tensorrt/run_export.sh`.

::: transformersurgeon.export.tensorrt.tensorrt_export

::: transformersurgeon.export.tensorrt.quantizer
