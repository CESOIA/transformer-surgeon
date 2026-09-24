# ONNX Export

Target-agnostic ONNX export of converted decoders: portable `model.onnx` plus a
`model.manifest.json` I/O contract, in-place KV cache (`cache_impl="io_inplace"`,
ONNX `TensorScatter`), optional prompt prefill (`max_input_len > 1`), and
weight-only Q/DQ for quantized layers.

::: transformersurgeon.export.onnx.onnx_export

::: transformersurgeon.export.onnx.quantization
