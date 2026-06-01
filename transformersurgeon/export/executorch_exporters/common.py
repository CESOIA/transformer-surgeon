import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ..config import BackendExportConfig


@dataclass
class QuantizationPlan:
    """Quantization plan placeholder for future mixed-precision support."""

    global_precision: str = "int4"
    layer_precisions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        model_config: Any | None,
        requested_precision: str = "int4",
    ) -> "QuantizationPlan":
        _ = model_config
        # TODO: implement per-layer precision logic based on model_config and requested_precision
        return cls(global_precision=requested_precision, layer_precisions={})


@dataclass
class ExecuTorchExportResult:
    pte_path: str
    backend: str
    precision: str
    weight_mismatches: list[dict[str, Any]]
    inference_stats: dict[str, float] | None = None


@dataclass
class ExecutorchExporterConfig(BackendExportConfig, ABC):
    """Abstract base config shared by backend-specific exporters."""

    precision: str = "int4"
    dynamic_shapes: dict[str, Any] | None = None


class LLMWrapper(nn.Module):
    """Static export wrapper for decode with fixed seq_len=1."""

    def __init__(self, embedding: nn.Module, decoder: nn.Module, final_layer: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.final_layer = final_layer

    def forward(
        self,
        input_ids: torch.LongTensor,
        last_pos_tensor: torch.LongTensor,
    ) -> torch.Tensor:
        last_pos = last_pos_tensor
        hidden = self.decoder(self.embedding(input_ids), last_pos=last_pos)
        logits = self.final_layer(hidden[-1, :])
        return logits


def build_wrapper(
    components: Any,
    *,
    model_config: Any | None,
) -> nn.Module:
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

    return LLMWrapper(embedding, decoder, final_layer)


def build_example_inputs(
    model_config: Any | None,
    *,
    config: Any,
) -> tuple[Any, ...]:
    vocab_size = int(getattr(model_config, "vocab_size", 100)) if model_config is not None else 100
    example_input_ids = torch.randint(0, vocab_size, (1,), dtype=torch.long)
    example_cache_len_tensor = torch.tensor([1], dtype=torch.long)

    max_seq_len = getattr(config, "max_seq_len", None)
    if isinstance(max_seq_len, int) and max_seq_len > 0:
        cache_len_value = int(example_cache_len_tensor.reshape(-1)[0].item())
        if cache_len_value > max_seq_len:
            raise ValueError(
                f"example_cache_len_tensor value ({cache_len_value}) exceeds max_seq_len ({max_seq_len})."
            )

    return (example_input_ids, example_cache_len_tensor)


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


def find_weight_mismatches(
    wrapper: nn.Module,
    quantized_exported,
    eps: float,
) -> list[dict[str, Any]]:
    """Best-effort match of exported dequantized tensors against original weights."""
    # Restrict matching to float 2D tensors to compare exported linear-like weights.
    original_candidates: list[tuple[str, torch.Tensor]] = []
    for name, tensor in wrapper.state_dict().items():
        if tensor.ndim == 2 and tensor.is_floating_point():
            original_candidates.append((name, tensor.detach().float()))

    exported_state_dict_attr = getattr(quantized_exported, "state_dict", None)
    exported_state_dict = (
        exported_state_dict_attr()
        if callable(exported_state_dict_attr)
        else exported_state_dict_attr
    )
    if not isinstance(exported_state_dict, dict):
        raise TypeError("exported object does not expose a dict-like state_dict")
    dequantized = dequant_from_exported_state_dict(exported_state_dict)

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
) -> tuple[nn.Module, Any | None, tuple[Any, ...], QuantizationPlan]:
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
        model_config=model_config,
    )
    wrapper.eval()

    quant_plan = QuantizationPlan.build(model_config, requested_precision=config.precision)
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
    if config.dynamic_shapes is not None:
        warnings.warn(
            "dynamic_shapes is ignored; exporter now uses static seq_len=1 contract.",
            stacklevel=2,
        )

    return wrapper, model_config, example_inputs, quant_plan
