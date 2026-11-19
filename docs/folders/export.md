# `export/` — from compressed model to device artifact

```python
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.backends import QNNExportConfig

result = export_to_backend(model, QNNExportConfig(output_path="out/model.pte", soc_model="SM8650"))
print(result.output_path, result.manifest_path)
```

## Three layers

1. **Generic steps** shared by every backend: `convert.py` (HF model → `blocks/` stacks), `wrapper.py`
   (`LLMWrapper`, the traced signature), `quant_info.py` (per-layer quantisation decided by compression),
   `pt2e.py` and `pt2e_linear_quantizer.py` (PyTorch 2 export quantisation), `checks.py`, `manifest.py`
   (the runtime contract written next to the artifact), `result.py`.
2. **The operations dictionary** `operations.py::BACKEND_OPERATIONS` (below).
3. **One exporter per backend** in `backends/`: subclasses of `Exporter` registered with `@register_exporter`.
   `export_to_backend` picks one by `config.backend`; `Exporter.export` completes the options, converts,
   moves components to CPU and calls the backend's `lower()`.

| Backend | Config | Output |
|---|---|---|
| `xnnpack` | `XNNPACKExportConfig` | ExecuTorch `.pte` (CPU) |
| `qnn` | `QNNExportConfig` | ExecuTorch `.pte` (Qualcomm NPU) |
| `onnx` | `ONNXExportConfig` | `model.onnx` + manifest |
| `tensorrt` | `TensorRTExportConfig` | ONNX + TensorRT engine |

## Encoder models

A model whose dictionary has an encoder component (`structure="transformer_encoder"`) and no decoder is exported
as an `EncoderWrapper`: input embedding → converted encoder → pooling (model key `pooling`) → output head, one
static call per input. Only `onnx` and `tensorrt` accept it (`Exporter.supports_encoder`).

```python
from transformersurgeon import Dinov2ForImageClassificationCompress
from transformersurgeon.export.backends import TensorRTExportConfig

model = Dinov2ForImageClassificationCompress.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer").eval()
result = export_to_backend(model, TensorRTExportConfig(output_path="out/model.onnx", float_type=torch.float16,
                                                       input_shape=(1, 3, 224, 224)))
# graph: pixel_values (1, 3, 224, 224) float16 -> logits (1, 1000); manifest format "tsurgeon-encoder/1"
```

## The operations dictionary

For every block type and backend: the default implementation and the supported alternatives, each with the
device operation it becomes. Missing options take the backend default (`resolve_convert_options`); unlisted
choices are rejected before tracing.

--8<-- "docs/generated/backend_operations.md"

Backend details (numerics, quantisation paths, speed options) are in
[AGENTS.md](https://github.com/CESOIA/transformer-surgeon/blob/main/AGENTS.md#backend-details).

API: [export](../api/export.md).
