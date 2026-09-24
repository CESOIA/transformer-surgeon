# TensorRT Export

Exports through [ONNX](onnx.md) and builds a strongly typed TensorRT engine from
it. The ONNX file plus its manifest are the portable artifact; build the engine
on the device that will run it
(`python -m transformersurgeon.export.tensorrt.tensorrt_export model.manifest.json`).

!!! note "Requirements"
    `pip install -e ".[onnx]"`, the `tensorrt` Python package matching your CUDA
    (e.g. `tensorrt-cu13`) and a CUDA device. CLI:
    `scripts/tensorrt/export_and_generate.py`. Background and benchmarks:
    `docs/investigations/TENSORRT_ONNX_EXPORT.md`.

::: transformersurgeon.export.tensorrt.tensorrt_export

::: transformersurgeon.export.tensorrt.engine

::: transformersurgeon.export.tensorrt.session

::: transformersurgeon.export.tensorrt.edgellm_int4
