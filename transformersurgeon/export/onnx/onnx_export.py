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

With ``cache_impl="io_inplace"`` the cache write is an opset-24 TensorScatter:
consumers should bind each present_* to the same buffer as its past_* (TensorRT
requires it -- it aliases them and updates in place).
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ..common import (
    MANIFEST_FORMAT,
    BackendExportResult,
    ExporterConfig,
    build_llm_manifest,
    extract_layer_quant_info,
    finalize_export_result,
    resolve_components_and_wrapper,
)
from .quantization import apply_weight_quantization


def _tensor_scatter(cache, update, write_index):
    from onnxscript import values

    return values.Opset("", 24).TensorScatter(cache, update, write_index, axis=2)


def default_translations() -> dict[Any, Callable]:
    """ATen/custom-op -> onnxscript translations every ONNX export needs."""
    from ...blocks import kv_cache_ops  # noqa: F401  (registers torch.ops.tsurgeon)

    return {torch.ops.tsurgeon.kv_cache_update.default: _tensor_scatter}


@dataclass
class ONNXExportResult(BackendExportResult):
    """``output_path`` is the .onnx file; ``manifest_path`` its I/O contract."""

    @property
    def onnx_path(self) -> str:
        return self.output_path


@dataclass
class ONNXExportConfig(ExporterConfig):
    """Configuration for the ONNX exporter.

    output_path        - path of the .onnx file (external weights go next to it).
    max_input_len      - >1 makes the sequence dim of input_ids dynamic
                         (1..max_input_len) so a whole prompt is one call;
                         needs cache_impl="io_inplace". 1 = decode-only graph.
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

    max_input_len: int = 1
    opset: int = 24
    optimize: bool = True
    int4_block_size: int | None = None
    custom_translations: dict[Any, Callable] = field(default_factory=dict)
    graph_passes: list[Callable] = field(default_factory=list)


def io_names(num_layers: int) -> tuple[list[str], list[str]]:
    """(input_names, output_names) in torch.onnx's flattened arg/output order."""
    inputs = ["input_ids", "pos_id"]
    inputs += [f"past_key_{i}" for i in range(num_layers)]
    inputs += [f"past_value_{i}" for i in range(num_layers)]
    outputs = ["logits"]
    outputs += [f"present_key_{i}" for i in range(num_layers)]
    outputs += [f"present_value_{i}" for i in range(num_layers)]
    return inputs, outputs


def export_with_onnx(model_or_graph: Any, *, config: ONNXExportConfig) -> ONNXExportResult:
    import onnx

    out_dir = os.path.dirname(os.path.abspath(config.output_path))
    os.makedirs(out_dir, exist_ok=True)

    wrapper, model_config, example_inputs = resolve_components_and_wrapper(model_or_graph, config=config)
    cache_impl = wrapper.cache_impl
    if cache_impl == "mutable":
        raise ValueError(
            "ONNX has no mutable state: convert with convert_options['cache_impl'] set to "
            "'io_inplace' (recommended; in-place TensorScatter), 'io_scatter' or 'io_concat'."
        )
    if config.max_input_len > 1 and cache_impl != "io_inplace":
        raise ValueError("max_input_len > 1 (prompt prefill) requires cache_impl='io_inplace'.")
    if wrapper.add_batch_dim:
        raise ValueError("add_batch_dim is not supported by the ONNX exporter (input_ids is (seq,)).")

    # Per-layer quantization metadata; hard-quantized weights are dequantized
    # in place here and re-expressed as Q/DQ after export.
    layer_info = extract_layer_quant_info(wrapper)

    num_layers = len(wrapper.decoder.blocks)
    input_names, output_names = io_names(num_layers)
    dynamic_shapes = None
    if config.max_input_len > 1:
        # torch.export specializes size-1 dims, so trace with a 2-token prompt.
        vocab = int(getattr(model_config, "vocab_size", 100) or 100)
        example_inputs = (torch.randint(0, vocab, (2,), dtype=torch.long), *example_inputs[1:])
        seq = torch.export.Dim("seq", min=1, max=config.max_input_len)
        dynamic_shapes = ({0: seq}, None, [None] * num_layers, [None] * num_layers)

    with torch.no_grad():
        logits_dtype = wrapper(*example_inputs)[0].dtype

    translations = {**default_translations(), **config.custom_translations}
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
    manifest = build_llm_manifest(
        wrapper, model_config, backend=config.backend, artifact=config.output_path,
        precision="mixed" if quantized else "full",
        onnx_file=os.path.basename(config.output_path),
        opset=config.opset,
        max_input_len=int(config.max_input_len),
        logits_dtype=str(logits_dtype).replace("torch.", ""),
        inputs={"input_ids": {"shape": ["seq"], "dtype": "int64"}, "pos_id": {"shape": [1], "dtype": "int64"}},
        quantized_layers={name: {"precision": q["precision"], "granularity": q["granularity"]}
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
    "MANIFEST_FORMAT",
    "ONNXExportConfig",
    "ONNXExportResult",
    "default_translations",
    "export_with_onnx",
    "io_names",
]
