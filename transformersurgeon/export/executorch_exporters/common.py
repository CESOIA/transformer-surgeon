import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.export import Dim

from ..config import BackendExportConfig


@dataclass
class QuantizationPlan:
    """Quantization plan placeholder for future mixed-precision support."""

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
    """Adapter hooks to support non-text models in future exports."""

    wrapper_builder: Any | None = None
    example_input_builder: Any | None = None
    calibration_runner: Any | None = None
    dynamic_shapes_builder: Any | None = None
    exported_weight_extractor: Any | None = None
    original_weight_extractor: Any | None = None


@dataclass
class BaseExporterConfig(BackendExportConfig, ABC):
    """Abstract base config shared by backend-specific exporters."""

    adapter: ExportAdapter = field(default_factory=ExportAdapter)

    @classmethod
    @abstractmethod
    def from_kwargs(
        cls,
        *,
        output_path: str,
        backend: str,
        adapter: ExportAdapter | None = None,
        **kwargs,
    ) -> "BaseExporterConfig":
        """Build backend config from keyword arguments."""


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
    """Static export wrapper for decode with fixed seq_len=1."""

    def __init__(self, embedding: nn.Module, decoder: nn.Module, final_layer: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.final_layer = final_layer

    def forward(
        self,
        input_ids: torch.LongTensor,
        last_pos_tensor: torch.Tensor,
    ) -> torch.Tensor:
        last_pos = last_pos_tensor.to(torch.long)
        hidden = self.decoder(self.embedding(input_ids), last_pos=last_pos, static=True)
        logits = self.final_layer(hidden[:, -1, :])
        return logits


def _build_quant_plan(
    model_config: Any | None,
    requested_precision: str = "int4",
) -> QuantizationPlan:
    _ = model_config
    return QuantizationPlan(global_precision=requested_precision, layer_precisions={})


def build_wrapper(
    components: Any,
    *,
    static_seq_len_1: bool,
    adapter: ExportAdapter,
    model_config: Any | None,
) -> nn.Module:
    if adapter.wrapper_builder is not None:
        return adapter.wrapper_builder(components, model_config)

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
        return EmbeddingDecoderFinalStaticSeqLen1Wrapper(embedding, decoder, final_layer)

    return EmbeddingDecoderFinalWrapper(embedding, decoder, final_layer)


def build_example_inputs(
    model_config: Any | None,
    *,
    config: Any,
) -> tuple[Any, ...]:
    if config.example_inputs is not None:
        return config.example_inputs

    if config.adapter.example_input_builder is not None:
        return config.adapter.example_input_builder(model_config)

    example_input_ids = config.example_input_ids
    example_cache_len_tensor = config.example_cache_len_tensor

    if example_input_ids is None:
        vocab_size = int(getattr(model_config, "vocab_size", 32000)) if model_config is not None else 32000
        default_seq_len = 1 if config.static_seq_len_1 else 2
        example_input_ids = torch.randint(0, vocab_size, (1, default_seq_len), dtype=torch.long)

    if example_cache_len_tensor is None:
        if config.static_seq_len_1:
            example_cache_len_tensor = torch.tensor([1], dtype=torch.long)
        else:
            example_cache_len_tensor = torch.ones(5)

    return (example_input_ids, example_cache_len_tensor)


def build_dynamic_shapes(config: Any) -> dict[str, Any] | None:
    if config.dynamic_shapes is not None:
        return config.dynamic_shapes

    if config.static_seq_len_1:
        return None

    if config.adapter.dynamic_shapes_builder is not None:
        return config.adapter.dynamic_shapes_builder(config.max_seq_len)

    dyn_seq_len = Dim("dyn_seq_len", min=1, max=config.max_seq_len - 1)
    dyn_cache_len = Dim("dyn_cache_len", min=1, max=config.max_seq_len - 1)
    return {
        "input_ids": (Dim.STATIC, dyn_seq_len),
        "cache_len_tensor": (dyn_cache_len,),
    }


def build_quantizer(precision: str):
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


def dequant_from_exported_state_dict(state_dict: dict[str, torch.Tensor]) -> list[torch.Tensor]:
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


def default_original_weight_extractor(wrapper: nn.Module) -> list[tuple[str, torch.Tensor]]:
    candidates: list[tuple[str, torch.Tensor]] = []
    for name, tensor in wrapper.state_dict().items():
        if tensor.ndim == 2 and tensor.is_floating_point():
            candidates.append((name, tensor.detach().float()))
    return candidates


def default_exported_weight_extractor(quantized_exported) -> list[torch.Tensor]:
    return dequant_from_exported_state_dict(quantized_exported.state_dict)


def find_weight_mismatches(
    wrapper: nn.Module,
    quantized_exported,
    eps: float,
    *,
    exported_weight_extractor=None,
    original_weight_extractor=None,
) -> list[dict[str, Any]]:
    """Best-effort match of exported dequantized tensors against original weights."""
    original_weight_extractor = original_weight_extractor or default_original_weight_extractor
    exported_weight_extractor = exported_weight_extractor or default_exported_weight_extractor
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


def run_simple_inference_stats(
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


def finalize_export_result(
    *,
    pte_path: str,
    backend: str,
    precision: str,
    wrapper: nn.Module,
    example_inputs: tuple[Any, ...],
    exported_for_mismatch: Any | None,
    run_weight_mismatch_check: bool,
    weight_mismatch_eps: float,
    adapter: ExportAdapter,
    verbose: bool,
) -> ExecuTorchExportResult:
    """Run backend-agnostic post-export checks and build the final result."""
    mismatches: list[dict[str, Any]] = []
    if run_weight_mismatch_check and exported_for_mismatch is not None:
        try:
            mismatches = find_weight_mismatches(
                wrapper,
                exported_for_mismatch,
                eps=weight_mismatch_eps,
                exported_weight_extractor=adapter.exported_weight_extractor,
                original_weight_extractor=adapter.original_weight_extractor,
            )
        except Exception as exc:
            warnings.warn(
                f"Weight mismatch check skipped due to extractor incompatibility: {exc}",
                stacklevel=2,
            )
    elif run_weight_mismatch_check and precision == "full":
        warnings.warn(
            "Weight mismatch check is skipped for precision='full' (no quantized frozen params).",
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
        stats = run_simple_inference_stats(
            wrapper,
            pte_path,
            example_inputs,
        )
        print("[ExecuTorch export] Simple inference error statistics")
        print(f"  max_abs_err : {stats['max_abs_err']:.8f}")
        print(f"  mean_abs_err: {stats['mean_abs_err']:.8f}")
        print(f"  mse         : {stats['mse']:.10f}")
        print(f"  rmse        : {stats['rmse']:.8f}")

    return ExecuTorchExportResult(
        pte_path=pte_path,
        backend=backend,
        precision=precision,
        weight_mismatches=mismatches,
        inference_stats=stats,
    )


def resolve_components_and_wrapper(
    model_or_graph: Any,
    *,
    config: Any,
) -> tuple[nn.Module, Any | None, tuple[Any, ...], dict[str, Any] | None, QuantizationPlan]:
    if isinstance(model_or_graph, dict):
        components = {
            "embedding": model_or_graph["embedding"],
            "decoder": model_or_graph["decoder"],
            "final_layer": model_or_graph["final_layer"],
        }
        model_config = model_or_graph.get("config")
    elif isinstance(model_or_graph, (tuple, list)) and len(model_or_graph) == 3:
        components = model_or_graph
        model_config = None
    else:
        raise TypeError(
            "Unsupported model input at backend layer. "
            "Pass export-ready components {embedding, decoder, final_layer} "
            "or call export_to_backend for full-model conversion."
        )

    wrapper = build_wrapper(
        components,
        static_seq_len_1=config.static_seq_len_1,
        adapter=config.adapter,
        model_config=model_config,
    )
    wrapper.eval()

    quant_plan = _build_quant_plan(model_config, requested_precision=config.precision)
    if quant_plan.global_precision not in {"int4", "int8", "full"}:
        raise ValueError(
            f"Unsupported precision '{quant_plan.global_precision}'. "
            "Use 'int4', 'int8', or 'full'."
        )
    if quant_plan.layer_precisions:
        warnings.warn(
            "Per-layer precision overrides are not implemented yet; using global precision only.",
            stacklevel=2,
        )

    example_inputs = build_example_inputs(model_config, config=config)
    dynamic_shapes = build_dynamic_shapes(config)

    return wrapper, model_config, example_inputs, dynamic_shapes, quant_plan
