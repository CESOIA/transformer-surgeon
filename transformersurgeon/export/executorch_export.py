import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.export import Dim, export as torch_export

from ..blocks import TransformerDecoder
from ..utils import convert_for_export


@dataclass
class QuantizationPlan:
    """Quantization plan placeholder for future mixed-precision support.

    Today we only apply a single global precision in PT2E (default int4).
    In the future this structure can be populated from model config to drive
    per-layer iterative quantization (int4/int8/fp16 by block).
    """

    global_precision: str = "int4"
    layer_precisions: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecuTorchExportResult:
    pte_path: str
    backend: str
    precision: str
    weight_mismatches: list[dict[str, Any]]
    inference_stats: dict[str, float] | None = None


@dataclass
class ExportAdapter:
    """Adapter hooks to support non-text models in future exports.

    Current default behavior remains text-only (embedding+decoder+final_layer),
    while this structure allows vision/multimodal pipelines to inject their
    own component resolution and wrapper/input logic without rewriting the
    core PT2E -> ExecuTorch flow.
    """

    resolver: Any | None = None
    wrapper_builder: Any | None = None
    example_input_builder: Any | None = None
    calibration_runner: Any | None = None
    dynamic_shapes_builder: Any | None = None
    exported_weight_extractor: Any | None = None
    original_weight_extractor: Any | None = None


class EmbeddingDecoderFinalWrapper(nn.Module):
    """Single wrapper that chains embedding -> decoder -> final layer."""

    def __init__(self, embedding: nn.Module, decoder: nn.Module, final_layer: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.final_layer = final_layer

    def forward(
        self,
        input_ids: torch.LongTensor,
        cache_len_tensor: torch.Tensor,
    ) -> torch.Tensor:
        last_pos = cache_len_tensor.size(0)
        hidden = self.decoder(self.embedding(input_ids), last_pos=last_pos)
        logits = self.final_layer(hidden[:, -1, :])
        return logits


class EmbeddingDecoderFinalStaticSeqLen1Wrapper(nn.Module):
    """Static export wrapper for decode with fixed seq_len=1.

    Input contract:
    - input_ids: shape (batch, 1)
    - seq_pos_tensor: tensor containing a single long value with the token position
      (0-based). Decoder cache_len is computed as seq_pos + 1.
    """

    def __init__(self, embedding: nn.Module, decoder: nn.Module, final_layer: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.final_layer = final_layer

    def forward(
        self,
        input_ids: torch.LongTensor,
        seq_pos_tensor: torch.Tensor,
    ) -> torch.Tensor:
        seq_pos = seq_pos_tensor.reshape(()).to(torch.long)
        hidden = self.decoder(self.embedding(input_ids), last_pos=seq_pos, static=True)
        logits = self.final_layer(hidden[:, -1, :])
        return logits


def _resolve_partitioner(backend: str, allow_fallback: bool = True):
    backend = backend.lower()
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        return backend, XnnpackPartitioner()

    if backend == "qnn":
        # QNN is lowered through Qualcomm helper APIs, not via generic partitioner.
        return backend, None

    raise ValueError(f"Unsupported backend '{backend}'. Use 'xnnpack' or 'qnn'.")


def _build_qnn_compiler_specs(qnn_soc_model: str = "SM8650", qnn_use_fp16: bool = False):
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
    )
    try:
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    except Exception:
        # Keep compatibility with older/newer ExecuTorch layouts.
        from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import QcomChipset

    if not hasattr(QcomChipset, qnn_soc_model):
        raise ValueError(f"Unknown QNN SoC '{qnn_soc_model}'.")

    backend_options = generate_htp_compiler_spec(use_fp16=qnn_use_fp16)
    return generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, qnn_soc_model),
        backend_options=backend_options,
    )


def _build_quant_plan(
    model_config: Any | None,
    requested_precision: str = "int4",
) -> QuantizationPlan:
    # Placeholder for config-driven mixed-precision policies.
    # Example future behavior:
    # - read block-level bitwidth map from model config
    # - iteratively quantize only selected blocks
    # - keep some blocks in int8/fp16
    _ = model_config
    return QuantizationPlan(global_precision=requested_precision, layer_precisions={})


def _is_tsurgeon_decoder(module: nn.Module) -> bool:
    return isinstance(module, TransformerDecoder)


def _resolve_model_components(
    model_or_graph: Any,
    *,
    convert_options: dict[str, Any] | None = None,
    verbose: bool = False,
) -> tuple[nn.Module, nn.Module, nn.Module, Any | None]:
    """Resolve (embedding, decoder, final_layer, model_config).

    Accepted inputs:
    - HF/TSurgeon full model with get_input_embeddings + lm_head + indexing
    - dict with keys: embedding, decoder, final_layer
    - tuple/list: (embedding, decoder, final_layer)
    """
    convert_options = convert_options or {"use_sdpa": False}

    if isinstance(model_or_graph, dict):
        embedding = model_or_graph["embedding"]
        decoder = model_or_graph["decoder"]
        final_layer = model_or_graph["final_layer"]
        config = model_or_graph.get("config")
        return embedding, decoder, final_layer, config

    if isinstance(model_or_graph, (tuple, list)) and len(model_or_graph) == 3:
        embedding, decoder, final_layer = model_or_graph
        return embedding, decoder, final_layer, None

    if hasattr(model_or_graph, "get_input_embeddings") and hasattr(model_or_graph, "lm_head"):
        embedding = model_or_graph.get_input_embeddings()
        final_layer = model_or_graph.lm_head

        decoder = None
        if _is_tsurgeon_decoder(model_or_graph):
            decoder = model_or_graph
        else:
            converted = convert_for_export(
                model_or_graph,
                options=convert_options,
                verbose=verbose,
            )
            decoder = converted.get("text")

        if decoder is None:
            raise ValueError(
                "Could not resolve decoder graph model from input model. "
                "Expected convert_for_export(...)[\"text\"] to be available."
            )

        return embedding, decoder, final_layer, getattr(model_or_graph, "config", None)

    raise TypeError(
        "Unsupported model input. Pass either: full model, "
        "dict {embedding, decoder, final_layer}, or "
        "tuple (embedding, decoder, final_layer)."
    )


def _default_wrapper_builder(components: Any, _model_config: Any | None) -> nn.Module:
    if isinstance(components, dict):
        embedding = components["embedding"]
        decoder = components["decoder"]
        final_layer = components["final_layer"]
    elif isinstance(components, (tuple, list)) and len(components) == 3:
        embedding, decoder, final_layer = components
    else:
        raise TypeError(
            "Default wrapper builder expects dict {embedding, decoder, final_layer} "
            "or tuple/list (embedding, decoder, final_layer)."
        )
    return EmbeddingDecoderFinalWrapper(embedding, decoder, final_layer)


def _build_quantizer(precision: str):
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    if precision == "int4":
        qconfig = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
            weight_qmin=-8,
            weight_qmax=7,
        )
    elif precision == "int8":
        qconfig = get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
        )
    else:
        raise ValueError(f"Unsupported precision '{precision}'. Use 'int4' or 'int8'.")

    return XNNPACKQuantizer().set_global(qconfig)


def _dequant_from_exported_state_dict(state_dict: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Extract all dequantized frozen params from PT2E exported state dict."""
    dequantized = []
    for key, int_w in state_dict.items():
        if not key.startswith("_frozen_param"):
            continue

        suffix = key.replace("_frozen_param", "")
        scale_key = f"_scale_{suffix}"
        zp_key = f"_zero_point_{suffix}"
        if scale_key not in state_dict:
            continue

        scale = state_dict[scale_key].float()
        zp = state_dict.get(zp_key)
        if zp is None:
            zp = torch.zeros_like(scale, dtype=torch.float32)
        else:
            zp = zp.float()

        dq = (int_w.float() - zp.unsqueeze(1)) * scale.unsqueeze(1)
        dequantized.append(dq)

    return dequantized


def _default_original_weight_extractor(wrapper: nn.Module) -> list[tuple[str, torch.Tensor]]:
    candidates: list[tuple[str, torch.Tensor]] = []
    for name, tensor in wrapper.state_dict().items():
        if tensor.ndim == 2 and tensor.is_floating_point():
            candidates.append((name, tensor.detach().float()))
    return candidates


def _default_exported_weight_extractor(quantized_exported) -> list[torch.Tensor]:
    return _dequant_from_exported_state_dict(quantized_exported.state_dict)


def _find_weight_mismatches(
    wrapper: nn.Module,
    quantized_exported,
    eps: float,
    *,
    exported_weight_extractor=None,
    original_weight_extractor=None,
) -> list[dict[str, Any]]:
    """Best-effort match of exported dequantized tensors against original weights."""
    original_weight_extractor = original_weight_extractor or _default_original_weight_extractor
    exported_weight_extractor = exported_weight_extractor or _default_exported_weight_extractor
    original_candidates = original_weight_extractor(wrapper)
    dequantized = exported_weight_extractor(quantized_exported)

    mismatches = []
    for idx, dq in enumerate(dequantized):
        same_shape = [(n, t) for n, t in original_candidates if list(t.shape) == list(dq.shape)]
        if not same_shape:
            continue

        best = None
        best_max = None
        best_mean = None
        for name, t in same_shape:
            err = (t - dq).abs()
            max_err = float(err.max().item())
            mean_err = float(err.mean().item())
            if best_max is None or max_err < best_max:
                best = name
                best_max = max_err
                best_mean = mean_err

        if best_max is not None and best_max > eps:
            mismatches.append(
                {
                    "exported_index": idx,
                    "matched_weight": best,
                    "max_abs_err": best_max,
                    "mean_abs_err": best_mean,
                    "eps": eps,
                }
            )

    return mismatches


def _run_simple_inference_stats(
    wrapper: nn.Module,
    pte_path: str,
    inference_inputs: tuple[Any, ...],
) -> dict[str, float]:
    from executorch.runtime import Runtime

    wrapper.eval()
    with torch.no_grad():
        y_ref = wrapper(*inference_inputs)

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    y_et = method.execute(list(inference_inputs))[0]
    if not isinstance(y_et, torch.Tensor):
        y_et = torch.tensor(y_et)

    err = (y_ref - y_et).abs()
    mse = ((y_ref - y_et) ** 2).mean()
    return {
        "max_abs_err": float(err.max().item()),
        "mean_abs_err": float(err.mean().item()),
        "mse": float(mse.item()),
        "rmse": float(torch.sqrt(mse).item()),
    }


def export_to_executorch(
    model_or_graph: Any,
    *,
    output_path: str,
    backend: str = "xnnpack",
    precision: str = "int4",
    static_seq_len_1: bool = False,
    calibration_data: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    example_input_ids: torch.Tensor | None = None,
    example_cache_len_tensor: torch.Tensor | None = None,
    max_seq_len: int = 2048,
    convert_options: dict[str, Any] | None = None,
    check_ir_validity: bool = True,
    verbose: bool = False,
    weight_mismatch_eps: float = 1e-4,
    allow_backend_fallback: bool = True,
    adapter: ExportAdapter | None = None,
    example_inputs: tuple[Any, ...] | None = None,
    dynamic_shapes: dict[str, Any] | None = None,
    run_weight_mismatch_check: bool = True,
    qnn_soc_model: str = "SM8650",
    qnn_use_fp16: bool = False,
) -> ExecuTorchExportResult:
    """Export model to a single ExecuTorch program.

    Features:
    - Accepts full model and converts to TSurgeon graph decoder, or direct
      (embedding, decoder, final_layer) components.
    - Uses one combined wrapper containing embedding + decoder + final layer.
    - PT2E export to XNNPACK/QNN with optional calibration data.
    - No custom fake quantization is applied.
    - Int4 enabled now with placeholder quant plan for future mixed precision.
    - If verbose, prints simple inference error statistics.
    - Warns when original-vs-exported(dequantized) weight mismatch > eps.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    adapter = adapter or ExportAdapter()

    resolver = adapter.resolver or _resolve_model_components
    components_result = resolver(
        model_or_graph,
        convert_options=convert_options,
        verbose=verbose,
    )
    if not isinstance(components_result, (tuple, list)) or len(components_result) != 4:
        raise ValueError(
            "Resolver must return a 4-item tuple: "
            "(components_or_embedding, decoder_or_none, final_layer_or_none, model_config)."
        )
    components_or_embedding, decoder, final_layer, model_config = components_result
    if decoder is not None and final_layer is not None:
        components = {
            "embedding": components_or_embedding,
            "decoder": decoder,
            "final_layer": final_layer,
        }
    else:
        components = components_or_embedding

    if adapter.wrapper_builder is not None:
        wrapper = adapter.wrapper_builder(components, model_config)
    else:
        if isinstance(components, dict):
            embedding = components["embedding"]
            decoder = components["decoder"]
            final_layer = components["final_layer"]
        elif isinstance(components, (tuple, list)) and len(components) == 3:
            embedding, decoder, final_layer = components
        else:
            raise TypeError(
                "Default wrapper builder expects dict {embedding, decoder, final_layer} "
                "or tuple/list (embedding, decoder, final_layer)."
            )

        if static_seq_len_1:
            wrapper = EmbeddingDecoderFinalStaticSeqLen1Wrapper(embedding, decoder, final_layer)
        else:
            wrapper = EmbeddingDecoderFinalWrapper(embedding, decoder, final_layer)
    wrapper.eval()

    # Placeholder for future mixed precision. Today we only enforce global int4/int8.
    quant_plan = _build_quant_plan(model_config, requested_precision=precision)
    if quant_plan.layer_precisions:
        warnings.warn(
            "Per-layer precision overrides are not implemented yet; using global precision only.",
            stacklevel=2,
        )

    if example_inputs is None:
        if adapter.example_input_builder is not None:
            example_inputs = adapter.example_input_builder(model_config)
        else:
            if example_input_ids is None:
                vocab_size = int(getattr(model_config, "vocab_size", 32000)) if model_config is not None else 32000
                default_seq_len = 1 if static_seq_len_1 else 2
                example_input_ids = torch.randint(0, vocab_size, (1, default_seq_len), dtype=torch.long)
            if example_cache_len_tensor is None:
                if static_seq_len_1:
                    example_cache_len_tensor = torch.tensor([0], dtype=torch.long)
                else:
                    example_cache_len_tensor = torch.ones(5)
            example_inputs = (example_input_ids, example_cache_len_tensor)

    if dynamic_shapes is None:
        if static_seq_len_1:
            dynamic_shapes = None
        elif adapter.dynamic_shapes_builder is not None:
            dynamic_shapes = adapter.dynamic_shapes_builder(max_seq_len)
        else:
            dyn_seq_len = Dim("dyn_seq_len", min=1, max=max_seq_len - 1)
            dyn_cache_len = Dim("dyn_cache_len", min=1, max=max_seq_len - 1)
            dynamic_shapes = {
                "input_ids": (Dim.STATIC, dyn_seq_len),
                "cache_len_tensor": (dyn_cache_len,),
            }

    resolved_backend, partitioner = _resolve_partitioner(
        backend,
        allow_fallback=allow_backend_fallback,
    )

    exported = torch_export(
        wrapper,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    )

    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    quantizer = _build_quantizer(quant_plan.global_precision)
    prepared = prepare_pt2e(exported.module(), quantizer)

    with torch.no_grad():
        if calibration_data:
            if adapter.calibration_runner is not None:
                adapter.calibration_runner(prepared, calibration_data)
            else:
                for cal_inputs in calibration_data:
                    if isinstance(cal_inputs, (tuple, list)):
                        prepared(*cal_inputs)
                    else:
                        prepared(cal_inputs)
        else:
            prepared(*example_inputs)

    converted = convert_pt2e(prepared)
    quantized_exported = torch_export(
        converted,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    )

    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    if resolved_backend == "qnn":
        try:
            from executorch.backends.qualcomm.utils.utils import (
                to_edge_transform_and_lower_to_qnn,
            )

            compiler_specs = _build_qnn_compiler_specs(
                qnn_soc_model=qnn_soc_model,
                qnn_use_fp16=qnn_use_fp16,
            )
            edge = to_edge_transform_and_lower_to_qnn(
                converted,
                example_inputs,
                compiler_specs,
                dynamic_shapes=dynamic_shapes,
            )
        except Exception as exc:
            if not allow_backend_fallback:
                raise RuntimeError("QNN backend is not available in this environment") from exc
            warnings.warn(
                "QNN backend is not available, falling back to XNNPACK",
                stacklevel=2,
            )
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )

            resolved_backend = "xnnpack"
            edge = to_edge_transform_and_lower(
                quantized_exported,
                compile_config=EdgeCompileConfig(_check_ir_validity=check_ir_validity),
                partitioner=[XnnpackPartitioner()],
            )
    else:
        edge = to_edge_transform_and_lower(
            quantized_exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=check_ir_validity),
            partitioner=[partitioner],
        )

    et_program = edge.to_executorch()
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    mismatches = []
    if run_weight_mismatch_check:
        try:
            mismatches = _find_weight_mismatches(
                wrapper,
                quantized_exported,
                eps=weight_mismatch_eps,
                exported_weight_extractor=adapter.exported_weight_extractor,
                original_weight_extractor=adapter.original_weight_extractor,
            )
        except Exception as exc:
            warnings.warn(
                f"Weight mismatch check skipped due to extractor incompatibility: {exc}",
                stacklevel=2,
            )
    if mismatches:
        worst = max(mismatches, key=lambda x: x["max_abs_err"])
        warnings.warn(
            "Weight mismatch above eps after export: "
            f"count={len(mismatches)} "
            f"worst_weight={worst['matched_weight']} "
            f"worst_max_abs_err={worst['max_abs_err']:.6g} "
            f"(eps={weight_mismatch_eps:.6g})",
            stacklevel=2,
        )

    stats = None
    if verbose:
        stats = _run_simple_inference_stats(
            wrapper,
            output_path,
            example_inputs,
        )
        print("[ExecuTorch export] Simple inference error statistics")
        print(f"  max_abs_err : {stats['max_abs_err']:.8f}")
        print(f"  mean_abs_err: {stats['mean_abs_err']:.8f}")
        print(f"  mse         : {stats['mse']:.10f}")
        print(f"  rmse        : {stats['rmse']:.8f}")

    return ExecuTorchExportResult(
        pte_path=output_path,
        backend=resolved_backend,
        precision=quant_plan.global_precision,
        weight_mismatches=mismatches,
        inference_stats=stats,
    )
