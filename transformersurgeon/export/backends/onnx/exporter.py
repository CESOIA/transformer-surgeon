"""
Target-agnostic ONNX exporter for converted decoders.

Produces a portable ``model.onnx`` (weights as external data) plus a JSON
manifest describing its I/O contract, so any ONNX consumer -- TensorRT on the
build host or on a Jetson, or another runtime -- can drive it without
tsurgeon-specific knowledge. Backend-specific steps (engine building,
runtimes) live in the consuming backend package (e.g. ``export/tensorrt``).

Graph contract (names are fixed; shapes/dtypes are recorded in the manifest):

  inputs   input_ids   (seq,) int64        seq = 1 for decode; up to
                                           max_input_len for prompt prefill
           pos_id      (1,)   int64        cache slot of the first new token
           past_key_i / past_value_i       one per decoder layer
  outputs  logits      (vocab,)            for the last input token
           present_key_i / present_value_i updated caches

With ``kv_precision="nvfp4"`` every cache has a second tensor of block scales,
past_key_scale_i / past_value_scale_i in and present_*_scale_i out.

With ``cache_impl="io_inplace"`` the cache write is an opset-24 TensorScatter:
consumers should bind each present_* to the same buffer as its past_* (TensorRT
requires it -- it aliases them and updates in place).

With ``cache_impl="io_rows"`` the graph only reads past_*: present_* hold the new
rows, ``(1, kv_heads, seq, width)``, which the consumer writes into its cache at
pos_id (``export.manifest.write_cache_rows``). FP8 rows (``kv_precision="fp8"``)
are QuantizeLinear/DequantizeLinear with a unit scale; NVFP4 rows
(``kv_precision="nvfp4"``) are FLOAT4E2M1 with FLOAT8E4M3FN block scales, written by
a QuantizeLinear of the scales and a blocked QuantizeLinear of the rows and read by
the two DequantizeLinear (ONNX Runtime 1.31 has no FLOAT4E2M1 DequantizeLinear;
TensorRT runs them).

With ``cache_impl="io_paged"`` each layer has one paged K+V pool instead
(past_kv_i / present_kv_i, also updated in place) and attention is the fused
``tsurgeon::paged_attention`` op, which has no standard ONNX form: the target
must map it (``custom_translations``; the TensorRT backend emits Edge-LLM's
AttentionPlugin).
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ...config import BackendExportConfig
from ...exporter import Exporter, register_exporter
from ...manifest import build_llm_manifest, cache_scales, graph_io_names
from ...quant_info import extract_layer_quant_info
from ...result import BackendExportResult, finalize_export_result
from ...wrapper import resolve_components_and_wrapper
from .quantization import apply_activation_quantization, apply_weight_quantization


def _tensor_scatter(cache, update, write_index):
    from onnxscript import values

    return values.Opset("", 24).TensorScatter(cache, update, write_index, axis=2)


def _fp8_quantize(x):
    from onnxscript import ir, values

    op = values.Opset("", 24)
    scale = op.CastLike(op.Constant(value=ir.tensor(1.0)), x)
    return op.QuantizeLinear(x, scale, output_dtype=ir.DataType.FLOAT8E4M3FN)


def _fp8_dequantize(rows):
    from onnxscript import ir, values

    op = values.Opset("", 24)
    return op.DequantizeLinear(rows, op.Constant(value=ir.tensor(1.0)))


def _nvfp4_quantize(x):
    from onnxscript import ir, values

    from ....blocks.attention.kv_precision.nvfp4 import E2M1_MAX, NVFP4_BLOCK

    op = values.Opset("", 24)
    one = op.Constant(value=ir.tensor(1.0))   # the per-tensor scale
    x = op.Cast(x, to=ir.DataType.FLOAT)
    blocked = op.Concat(op.Shape(x, end=-1), op.Constant(value=ir.tensor([-1, NVFP4_BLOCK], dtype=ir.DataType.INT64)),
                        axis=0)
    amax = op.ReduceMax(op.Abs(op.Reshape(x, blocked)), op.Constant(value=ir.tensor([-1], dtype=ir.DataType.INT64)),
                        keepdims=0)
    scales = op.QuantizeLinear(op.Div(amax, op.Constant(value=ir.tensor(E2M1_MAX))), one,
                               output_dtype=ir.DataType.FLOAT8E4M3FN)
    step = op.DequantizeLinear(scales, one)
    step = op.Where(op.Greater(step, op.Constant(value=ir.tensor(0.0))), step, one)   # all-zero block
    rows = op.QuantizeLinear(x, step, axis=-1, block_size=NVFP4_BLOCK, output_dtype=ir.DataType.FLOAT4E2M1)
    return rows, scales


def _nvfp4_dequantize(rows, scales):
    from onnxscript import ir, values

    from ....blocks.attention.kv_precision.nvfp4 import NVFP4_BLOCK

    op = values.Opset("", 24)
    step = op.DequantizeLinear(scales, op.Constant(value=ir.tensor(1.0)))
    return op.DequantizeLinear(rows, step, axis=-1, block_size=NVFP4_BLOCK)


def default_translations() -> dict[Any, Callable]:
    """ATen/custom-op -> onnxscript translations every ONNX export needs."""
    from ....blocks.attention.kv_cache import io_inplace  # noqa: F401  (registers torch.ops.tsurgeon.kv_cache_update)
    from ....blocks.attention.kv_precision import fp8, nvfp4  # noqa: F401  (registers kv_fp8_* and kv_nvfp4_*)

    return {torch.ops.tsurgeon.kv_cache_update.default: _tensor_scatter,
            torch.ops.tsurgeon.kv_fp8_quantize.default: _fp8_quantize,
            torch.ops.tsurgeon.kv_fp8_dequantize.default: _fp8_dequantize,
            torch.ops.tsurgeon.kv_nvfp4_quantize.default: _nvfp4_quantize,
            torch.ops.tsurgeon.kv_nvfp4_dequantize.default: _nvfp4_dequantize}


@dataclass
class ONNXExportResult(BackendExportResult):
    """``output_path`` is the .onnx file; ``manifest_path`` its I/O contract."""

    @property
    def onnx_path(self) -> str:
        return self.output_path


@dataclass
class ONNXExportConfig(BackendExportConfig):
    """Configuration for the ONNX exporter.

    output_path        - path of the .onnx file (external weights go next to it).
    max_input_len      - >1 makes the sequence dim of input_ids dynamic
                         (1..max_input_len) so a whole prompt is one call;
                         needs cache_impl="io_inplace" or "io_rows". 1 = decode-only graph.
    opset              - ONNX opset; 24 is required for TensorScatter.
    optimize           - run the exporter's ONNX optimizer (constant folding,
                         dead-code removal).
    custom_translations - extra {torch op overload: onnxscript fn} entries,
                         merged over default_translations(); lets a target
                         map ops to its own custom ONNX nodes.
    graph_passes       - callables (onnx.ModelProto, manifest dict) -> None run
                         on the exported model before it is saved (target-
                         specific rewrites).
    int4_block_size    - INT4 weight DQ block size along in_features. None
                         keeps tsurgeon's per-channel scales exactly; a block
                         size (e.g. 128, what INT4 GEMM kernels expect)
                         re-derives max-abs scales per block.
    """

    backend: str = "onnx"
    max_input_len: int = 1
    opset: int = 24
    optimize: bool = True
    int4_block_size: int | None = None
    custom_translations: dict[Any, Callable] = field(default_factory=dict)
    graph_passes: list[Callable] = field(default_factory=list)


# Graph I/O names are defined once, next to the manifest that records them.
io_names = graph_io_names


@register_exporter
class ONNXExporter(Exporter):
    """Exports the wrapper with ``torch.onnx`` (dynamo), then rewrites quantised layers as Q/DQ."""

    backend = "onnx"
    config_class = ONNXExportConfig

    def lower(self, components, config):
        return export_onnx(components, config)


def export_onnx(components: dict, config: ONNXExportConfig) -> ONNXExportResult:
    """ONNX lowering of resolved export components (shared with the TensorRT exporter)."""
    import onnx

    out_dir = os.path.dirname(os.path.abspath(config.output_path))
    wrapper, model_config, example_inputs = resolve_components_and_wrapper(components)
    cache_impl = wrapper.cache_impl
    if config.max_input_len > 1 and cache_impl not in ("io_inplace", "io_rows"):
        raise ValueError("max_input_len > 1 (prompt prefill) requires cache_impl='io_inplace' or 'io_rows'.")

    # Per-layer quantization metadata; hard-quantized weights are dequantized
    # in place here and re-expressed as Q/DQ after export.
    layer_info = extract_layer_quant_info(wrapper)

    num_layers = len(wrapper.decoder.blocks)
    input_names, output_names = io_names(num_layers, paged=cache_impl == "io_paged",
                                         scales=cache_scales(wrapper.decoder))
    dynamic_shapes = None
    if config.max_input_len > 1:
        # torch.export specializes size-1 dims, so trace with a 2-token prompt.
        vocab = int(getattr(model_config, "vocab_size", 100) or 100)
        # The caches are static. torch.onnx turns the tuples of dynamic_shapes into lists, so
        # (rows, scales) cache pairs are traced as lists.
        keys, values = ([list(c) if isinstance(c, tuple) else c for c in caches] for caches in example_inputs[2:])
        example_inputs = (torch.randint(0, vocab, (2,), dtype=torch.long), example_inputs[1], keys, values)
        seq = torch.export.Dim("seq", min=1, max=config.max_input_len)
        static = [None if isinstance(c, torch.Tensor) else [None] * len(c) for c in keys]
        dynamic_shapes = ({0: seq}, None, static, static)

    with torch.no_grad():
        logits_dtype = wrapper(*example_inputs)[0].dtype

    translations = {**default_translations(), **config.custom_translations}
    if cache_impl == "io_paged" and torch.ops.tsurgeon.paged_attention.default not in translations:
        raise ValueError("cache_impl='io_paged' has no portable ONNX form: export with a backend that maps "
                         "tsurgeon::paged_attention (backend='tensorrt').")
    program = torch.onnx.export(
        wrapper,
        example_inputs,
        dynamo=True,
        opset_version=config.opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        custom_translation_table=translations,
        optimize=config.optimize,
        verbose=config.verbose,
    )
    model = program.model_proto

    quantized = (apply_weight_quantization(model, layer_info, int4_block_size=config.int4_block_size)
                 if layer_info else {})
    act_quantized = apply_activation_quantization(model, layer_info) if layer_info else {}
    manifest = build_llm_manifest(
        wrapper, model_config, backend=config.backend, artifact=config.output_path,
        precision="mixed" if quantized else "full",
        onnx_file=os.path.basename(config.output_path),
        opset=config.opset,
        max_input_len=int(config.max_input_len),
        logits_dtype=str(logits_dtype).replace("torch.", ""),
        inputs={"input_ids": {"shape": ["seq"], "dtype": "int64"}, "pos_id": {"shape": [1], "dtype": "int64"}},
        quantized_layers={name: {"precision": q["precision"], "granularity": q["granularity"],
                                 **act_quantized.get(name, {})}
                          for name, q in quantized.items()},
    )
    for graph_pass in config.graph_passes:
        graph_pass(model, manifest)

    data_file = os.path.basename(config.output_path) + ".data"
    if os.path.exists(os.path.join(out_dir, data_file)):
        os.remove(os.path.join(out_dir, data_file))  # save_model appends to existing data files
    onnx.save_model(model, config.output_path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location=data_file, size_threshold=1024)
    onnx.checker.check_model(config.output_path)

    return finalize_export_result(
        output_path=config.output_path,
        backend=config.backend,
        precision="mixed" if quantized else "full",
        wrapper=wrapper,
        exported_for_mismatch=None,
        run_weight_mismatch_check=False,
        weight_mismatch_eps=config.weight_mismatch_eps,
        verbose=config.verbose,
        result_cls=ONNXExportResult,
        model_config=model_config,
        manifest=manifest,
    )


__all__ = [
    "ONNXExportConfig",
    "ONNXExportResult",
    "ONNXExporter",
    "default_translations",
    "export_onnx",
    "io_names",
]
