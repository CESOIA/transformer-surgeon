"""
Common utilities shared by all ExecuTorch backend exporters.

Architecture overview
---------------------
Export pipeline (called by each backend exporter):

  1. resolve_components_and_wrapper()
       Normalises the model input (full HF model, dict, or tuple) into an
       LLMWrapper — a minimal nn.Module with a fixed-length forward signature
       that torch.export can trace.

  2. extract_layer_quant_info()   [quantised models only]
       Scans the wrapper for layers tagged by the compression pipeline:
         - hard-quantised  (_torchao_precision + torchao AffineQuantizedTensor
                            OR pre-dequantised weight + _torchao_scale stash)
         - soft-quantised  (_soft_quant_precision)
       Returns a per-layer dict with scale, precision, granularity, and any
       stored activation calibration values.

  3. build_quantizer_from_layer_info()   [quantised models only]
       Builds an XNNPACKQuantizer with one per-module qconfig per quantised
       layer.  Unquantised layers are NOT included → they stay float.

  4. prepare_pt2e / calibration pass / inject_scales_into_pt2e_observers()
       Standard PT2E flow.  For hard-quantised layers the observer min/max
       values are overridden with the exact torchao scales so no re-calibration
       is needed.  For soft-quantised layers the calibration-derived values are
       kept.  Stored activation calibration values are injected similarly.

  5. convert_pt2e / torch.export   [quantised models only]

  6. finalize_export_result()
       Optional weight-mismatch check and inference stats, then wraps
       everything into ExecuTorchExportResult.
"""

import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx
import torch.nn as nn

from ..config import BackendExportConfig


# ---------------------------------------------------------------------------
# Quantization range tables
# ---------------------------------------------------------------------------

# XNNPACK symmetric per-channel weight quantization ranges by bit-width.
_PRECISION_TO_QRANGE: dict[int, dict[str, int]] = {
    4: {"weight_qmin": -8,   "weight_qmax": 7},
    8: {"weight_qmin": -127, "weight_qmax": 127},
}

# Denominator for converting a per-channel scale back to observer min/max.
# Derived from (weight_qmax - weight_qmin) / 2 for symmetric quantization.
_PRECISION_TO_EFFECTIVE_DENOM: dict[int, float] = {
    4: 7.5,
    8: 127.0,
}

# Human-readable precision identifiers shared across backends and CLI args.
# weight_bits / activation_bits = None means that axis is not quantised.
SUPPORTED_QUANT_CONFIGS: dict[str, dict] = {
    "full":  {"weight_bits": None, "activation_bits": None, "description": "No quantization (FP32)"},
    "w4":    {"weight_bits": 4,    "activation_bits": None, "description": "4-bit weight-only"},
    "w8":    {"weight_bits": 8,    "activation_bits": None, "description": "8-bit weight-only"},
    "a8w8":  {"weight_bits": 8,    "activation_bits": 8,    "description": "8-bit activations + 8-bit weights"},
    "a8w4":  {"weight_bits": 4,    "activation_bits": 8,    "description": "8-bit activations + 4-bit weights"},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecuTorchExportResult:
    pte_path: str
    backend: str
    precision: str
    weight_mismatches: list[dict[str, Any]]
    inference_stats: dict[str, float] | None = None


@dataclass
class ExecutorchExporterConfig(BackendExportConfig, ABC):
    """Abstract base config shared by all backend-specific exporters."""

    # "full" is the default — quantization is inferred from model metadata,
    # not forced from config.  Set a value only for legacy global-precision flows.
    precision: str = "full"
    dynamic_shapes: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class LLMWrapper(nn.Module):
    """Thin wrapper that presents the three model components under a fixed
    forward signature that torch.export can trace with a single token input."""

    def __init__(self, embedding: nn.Module, decoder: nn.Module, final_layer: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.final_layer = final_layer

    def forward(
        self,
        input_ids: torch.LongTensor,
        pos_id_tensor: torch.LongTensor,
    ) -> torch.Tensor:
        hidden = self.decoder(self.embedding(input_ids), pos_id=pos_id_tensor)
        return self.final_layer(hidden[-1, :])


def build_wrapper(components: Any, *, model_config: Any | None) -> nn.Module:
    if isinstance(components, dict):
        embedding   = components["embedding"]
        decoder     = components["decoder"]
        final_layer = components["final_layer"]
    elif isinstance(components, (tuple, list)) and len(components) == 3:
        embedding, decoder, final_layer = components
    else:
        raise TypeError(
            "Default wrapper builder expects dict {embedding, decoder, final_layer} "
            "or tuple/list (embedding, decoder, final_layer)."
        )
    return LLMWrapper(embedding, decoder, final_layer)


def build_example_inputs(model_config: Any | None, *, config: Any) -> tuple[Any, ...]:
    """Build a single-token example input for tracing / calibration."""
    vocab_size = int(getattr(model_config, "vocab_size", 100)) if model_config is not None else 100
    input_ids  = torch.randint(0, vocab_size, (1,), dtype=torch.long)
    pos_ids    = torch.tensor([1], dtype=torch.long)

    max_seq_len = getattr(config, "max_seq_len", None)
    if isinstance(max_seq_len, int) and max_seq_len > 0:
        if int(pos_ids[0].item()) > max_seq_len:
            raise ValueError(
                f"example position value ({int(pos_ids[0].item())}) exceeds max_seq_len ({max_seq_len})."
            )

    return (input_ids, pos_ids)


def resolve_components_and_wrapper(
    model_or_graph: Any,
    *,
    config: Any,
) -> tuple[nn.Module, Any | None, tuple[Any, ...]]:
    """Unpack the model input into (wrapper, model_config, example_inputs).

    Accepts the dict form produced by export_to_backend (after conversion)
    or a plain 3-tuple (embedding, decoder, final_layer).
    """
    if isinstance(model_or_graph, dict):
        components   = {k: model_or_graph[k] for k in ("embedding", "decoder", "final_layer")}
        model_config = model_or_graph.get("config")
    elif isinstance(model_or_graph, (tuple, list)) and len(model_or_graph) == 3:
        components   = model_or_graph
        model_config = None
    else:
        raise TypeError(
            "Unsupported model input at backend layer. "
            "Pass export-ready components {embedding, decoder, final_layer} "
            "or call export_to_backend for full-model conversion."
        )

    wrapper = build_wrapper(components, model_config=model_config)
    wrapper.eval()

    example_inputs = build_example_inputs(model_config, config=config)
    if config.dynamic_shapes is not None:
        warnings.warn(
            "dynamic_shapes is ignored; exporter uses a static seq_len=1 contract.",
            stacklevel=2,
        )

    return wrapper, model_config, example_inputs


# ---------------------------------------------------------------------------
# Per-layer quantization metadata extraction
# ---------------------------------------------------------------------------

def extract_layer_quant_info(wrapper: nn.Module) -> dict[str, dict[str, Any]]:
    """Scan the wrapper for compression-tagged linear layers.

    Returns a dict  {layer_name: info}  where info contains:

        scale        – per-channel weight scale tensor (None for soft-quant)
        precision    – integer bit-width (4, 8, …)
        hard         – True for torchao hard-quant, False for soft-quant
        per_channel  – True if scale is per output-channel, False for per-tensor
        act_scale    – activation scale (None if not calibrated)
        act_zero_point, act_precision, act_scheme  (only when act_scale is set)

    Side effects:
      - Hard-quantised weights are dequantised in-place so torch.export sees
        plain float tensors.
      - Activation fake-quant hooks are removed (PT2E handles quantisation
        from this point on).

    Layers without any compression tag are not included in the output.
    """
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:
        AffineQuantizedTensor = None

    layer_info: dict[str, dict[str, Any]] = {}

    for name, module in wrapper.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        info = _extract_weight_quant_info(module, AffineQuantizedTensor)
        if info is None:
            continue

        _extract_act_quant_info(module, info)  # adds act_* keys; removes hook
        layer_info[name] = info

    return layer_info


def _extract_weight_quant_info(
    module: nn.Linear,
    AffineQuantizedTensor: type | None,
) -> dict[str, Any] | None:
    """Return weight quantization info for one linear layer, or None if unquantised."""

    has_torchao_tag = hasattr(module, '_torchao_precision')
    is_live_aqt = (
        AffineQuantizedTensor is not None
        and isinstance(module.weight, AffineQuantizedTensor)
    )
    has_stashed_scale = hasattr(module, '_torchao_scale')

    if has_torchao_tag and (is_live_aqt or has_stashed_scale):
        # Hard-quantised layer.
        if is_live_aqt:
            # Weight is still a torchao AffineQuantizedTensor (direct model pass).
            # Dequantise in-place so torch.export sees a clean float parameter.
            w = module.weight
            scale       = w.tensor_impl.scale.detach().clone()
            per_channel = scale.numel() > 1
            module.weight = nn.Parameter(w.dequantize(), requires_grad=False)
        else:
            # Weight was dequantised earlier by _load_state_dict_dequantized
            # (convert_for_export path); scale was stashed as _torchao_scale.
            scale       = module._torchao_scale.detach().clone()
            per_channel = getattr(module, '_torchao_per_channel', scale.numel() > 1)

        return {
            "scale":       scale,
            "precision":   module._torchao_precision,
            "hard":        True,
            "per_channel": per_channel,
        }

    if hasattr(module, '_soft_quant_precision'):
        # Soft-quantised layer: weight is already fake-quantised float.
        # Scales must be re-derived from a calibration pass.
        warnings.warn(
            f"Layer '{module}' was soft-quantized; re-deriving scale via calibration pass.",
            stacklevel=3,
        )
        return {
            "scale":       None,
            "precision":   module._soft_quant_precision,
            "hard":        False,
            "per_channel": True,  # soft-quant always uses per-channel observers
        }

    return None  # unquantised — skip


def _extract_act_quant_info(module: nn.Linear, info: dict) -> None:
    """Populate activation quantization keys in *info* and remove the fake-quant hook.

    If the module has no stored activation calibration the key act_scale is set
    to None and no other act_* keys are added.
    """
    if hasattr(module, '_act_quant_scale'):
        info["act_scale"]      = module._act_quant_scale.detach().clone()
        info["act_zero_point"] = module._act_quant_zero_point.detach().clone()
        info["act_precision"]  = module._act_quant_precision
        info["act_scheme"]     = module._act_quant_scheme
        # PT2E inserts its own activation observers; the fake-quant hook is no longer needed.
        if hasattr(module, '_act_quant_hook_handle'):
            module._act_quant_hook_handle.remove()
            del module._act_quant_hook_handle
    else:
        info["act_scale"] = None


# ---------------------------------------------------------------------------
# PT2E quantizer construction
# ---------------------------------------------------------------------------

def build_quantizer(precision: str):
    """Build a global XNNPACKQuantizer for a single precision level.

    Used by backends that do not have per-layer compression metadata.
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    if precision == "w4":
        qconfig = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True, weight_qmin=-8, weight_qmax=7,
        )
    elif precision == "w8":
        qconfig = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    else:
        raise ValueError(f"Unsupported precision '{precision}' for XNNPACK. Use 'w4' or 'w8'.")

    return XNNPACKQuantizer().set_global(qconfig)


def build_quantizer_from_layer_info(layer_info: dict[str, dict[str, Any]]):
    """Build a per-module XNNPACKQuantizer driven by compression metadata.

    Each layer in layer_info gets its own qconfig; layers absent from
    layer_info are not quantised (they remain float).

    Layers with stored activation calibration data use static quantization
    (is_dynamic=False); others use dynamic weight-only (is_dynamic=True).
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = XNNPACKQuantizer()
    for layer_name, info in layer_info.items():
        qrange = _PRECISION_TO_QRANGE.get(info["precision"])
        if qrange is None:
            warnings.warn(
                f"Unsupported compression precision {info['precision']} for "
                f"layer '{layer_name}'; skipping — layer will remain float.",
                stacklevel=2,
            )
            continue

        is_dynamic = info["act_scale"] is None  # False → static (act + weight), True → weight-only
        qconfig = get_symmetric_quantization_config(
            is_per_channel=info["per_channel"],
            is_dynamic=is_dynamic,
            **qrange,
        )
        quantizer.set_module_name(layer_name, qconfig)

    return quantizer


# ---------------------------------------------------------------------------
# PT2E observer scale injection
# ---------------------------------------------------------------------------

def inject_scales_into_pt2e_observers(
    prepared_model: nn.Module,
    layer_info: dict[str, dict[str, Any]],
) -> int:
    """Bypass calibration for hard-quantised layers by injecting known scales.

    After prepare_pt2e + calibration forward pass, PT2E has populated every
    observer with statistics from the (dequantised) float weights.  For hard-
    quantised layers we override those statistics with the exact torchao scales
    so that convert_pt2e reproduces the original quantisation without error.
    Soft-quantised layers keep the calibration-derived values.

    Also injects stored activation calibration values into the paired
    activation observers for layers that have act_scale set.

    Returns the total number of observer modules overridden.
    """
    try:
        from torch.ao.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver
        _observer_types = (PerChannelMinMaxObserver, MinMaxObserver)
    except ImportError:
        warnings.warn("Could not import PT2E observer types; scale injection skipped.", stacklevel=2)
        return 0

    hard_layers = {k for k, v in layer_info.items() if v["hard"] and v["scale"] is not None}
    act_layers  = {k for k, v in layer_info.items() if v.get("act_scale") is not None}
    if not hard_layers and not act_layers:
        return 0

    overridden   = 0
    matched_hard = set()
    matched_act  = set()

    for node in prepared_model.graph.nodes:
        if node.op != 'call_module':
            continue

        # We only care about observer modules.
        try:
            submod = prepared_model.get_submodule(node.target)
        except AttributeError:
            continue
        if not isinstance(submod, _observer_types):
            continue

        # Weight observers have a get_attr node (the weight tensor) as their input.
        weight_node = node.args[0] if node.args else None
        if weight_node is None or weight_node.op != 'get_attr':
            continue

        # Strip ".weight" suffix to recover the layer path (e.g. "decoder.blocks.0.mlp.gate_proj").
        parts = weight_node.target.split('.')
        if parts[-1] != 'weight':
            continue
        layer_name = '.'.join(parts[:-1])

        info = layer_info.get(layer_name)
        if info is None:
            continue

        # Override weight observer min/max with exact torchao scales.
        if layer_name in hard_layers:
            overridden += _inject_weight_observer(submod, info)
            matched_hard.add(layer_name)

        # Find and override the paired activation observer for this layer.
        # The activation observer feeds into the same downstream op as the weight observer.
        if layer_name in act_layers and layer_name not in matched_act:
            if _inject_paired_act_observer(node, prepared_model, submod, info, _observer_types):
                overridden += 1
                matched_act.add(layer_name)

    _warn_unmatched(hard_layers - matched_hard, "weight")
    _warn_unmatched(act_layers  - matched_act,  "activation")

    return overridden


def _inject_weight_observer(observer: nn.Module, info: dict) -> int:
    """Set observer min/max from stored torchao scale. Returns 1 on success, 0 on skip."""
    denom = _PRECISION_TO_EFFECTIVE_DENOM.get(info["precision"])
    if denom is None:
        return 0

    # Symmetric quantization: min = -scale * denom, max = scale * denom.
    # Setting these values causes convert_pt2e to reproduce the exact same scale.
    max_abs = info["scale"].reshape(-1).float() * denom
    if not info["per_channel"]:
        max_abs = max_abs.squeeze()

    observer.min_vals = -max_abs
    observer.max_vals = max_abs
    return 1


def _inject_paired_act_observer(
    weight_obs_node: torch.fx.Node,
    prepared_model: nn.Module,
    weight_obs_submod: nn.Module,
    info: dict,
    observer_types: tuple,
) -> bool:
    """Find the activation observer paired to weight_obs_node and inject calibration values.

    The pairing heuristic: both the weight observer and the activation observer
    feed into the same downstream op (the quantised linear computation).

    Returns True if the activation observer was found and updated.
    """
    for downstream_op in weight_obs_node.users:
        for arg in downstream_op.args:
            if not isinstance(arg, torch.fx.Node) or arg is weight_obs_node:
                continue
            if arg.op != 'call_module':
                continue
            try:
                act_submod = prepared_model.get_submodule(arg.target)
            except AttributeError:
                continue
            if not isinstance(act_submod, observer_types):
                continue
            # Found the activation observer — inject stored calibration values.
            _inject_act_observer(act_submod, info)
            return True
    return False


def _inject_act_observer(observer: nn.Module, info: dict) -> None:
    """Set observer min/max from stored activation scale and zero_point.

    Recovers the original observed min/max range using:
        min_val = (qmin - zero_point) * scale
        max_val = (qmax - zero_point) * scale
    This causes convert_pt2e to reproduce the same scale and zero_point.
    """
    s  = info["act_scale"].float().squeeze()
    zp = info["act_zero_point"].float().squeeze()

    if info["act_scheme"] == "symmetric":
        qmax = float(2 ** (info["act_precision"] - 1) - 1)
        qmin = -qmax
    else:
        qmax = float(2 ** info["act_precision"] - 1)
        qmin = 0.0

    min_val = (qmin - zp) * s
    max_val = (qmax - zp) * s

    # PerChannelMinMaxObserver uses min_vals/max_vals; MinMaxObserver uses min_val/max_val.
    if hasattr(observer, 'min_vals'):
        observer.min_vals = min_val
        observer.max_vals = max_val
    else:
        observer.min_val = min_val
        observer.max_val = max_val


def _warn_unmatched(unmatched: set[str], kind: str) -> None:
    if unmatched:
        warnings.warn(
            f"Could not match PT2E {kind} observers for layers: {sorted(unmatched)}. "
            "Calibration-derived values will be used instead.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Post-export checks
# ---------------------------------------------------------------------------

def dequant_from_exported_state_dict(state_dict: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Reconstruct dequantized weight tensors from a PT2E exported state dict.

    PT2E stores quantised weights as _frozen_paramN (int8) alongside
    _scale_N and optionally _zero_point_N tensors.
    """
    dequantized = []
    for key, int_w in state_dict.items():
        if not key.startswith("_frozen_param") or int_w.ndim != 2:
            continue

        suffix    = key.replace("_frozen_param", "")
        scale_key = f"_scale_{suffix}"
        if scale_key not in state_dict:
            continue

        scale = state_dict[scale_key].float()
        zp    = state_dict.get(f"_zero_point_{suffix}")
        zp    = zp.float() if zp is not None else torch.zeros_like(scale)

        dequantized.append((int_w.float() - zp.unsqueeze(1)) * scale.unsqueeze(1))

    return dequantized


def find_weight_mismatches(
    wrapper: nn.Module,
    quantized_exported: Any,
    eps: float,
) -> list[dict[str, Any]]:
    """Best-effort match of exported dequantized weights against the wrapper's float weights.

    For each dequantized exported weight, finds the original layer whose shape
    matches and whose absolute error exceeds eps.  Shape-based matching is
    approximate but sufficient for a sanity check.
    """
    original_candidates = [
        (name, tensor.detach().float())
        for name, tensor in wrapper.state_dict().items()
        if tensor.ndim == 2 and tensor.is_floating_point()
    ]

    state_dict_attr = getattr(quantized_exported, "state_dict", None)
    exported_state_dict = state_dict_attr() if callable(state_dict_attr) else state_dict_attr
    if not isinstance(exported_state_dict, dict):
        raise TypeError("exported object does not expose a dict-like state_dict")

    mismatches = []
    for dq in dequant_from_exported_state_dict(exported_state_dict):
        same_shape = [(n, t) for n, t in original_candidates if list(t.shape) == list(dq.shape)]
        if not same_shape:
            continue

        # Pick the original weight with the smallest absolute error (best match by shape).
        best_name, best_max, best_mean = None, None, None
        for name, t in same_shape:
            err      = (t - dq).abs()
            max_err  = float(err.max().item())
            mean_err = float(err.mean().item())
            if best_max is None or max_err < best_max:
                best_name, best_max, best_mean = name, max_err, mean_err

        if best_max is not None and best_max > eps:
            mismatches.append({
                "matched_weight": best_name,
                "max_abs_err":    best_max,
                "mean_abs_err":   best_mean,
                "eps":            eps,
            })

    return mismatches


def run_simple_inference_stats(
    wrapper: nn.Module,
    pte_path: str,
    inference_inputs: tuple[Any, ...],
) -> dict[str, float] | None:
    """Run one forward pass through both the float wrapper and the exported program
    and return absolute/MSE error statistics.  Returns None if the ExecuTorch
    runtime is not installed.
    """
    try:
        from executorch.runtime import Runtime
    except (ImportError, ModuleNotFoundError) as e:
        warnings.warn(
            f"ExecuTorch runtime not available; skipping inference stats. ({e})",
            stacklevel=2,
        )
        return None

    wrapper.eval()
    with torch.no_grad():
        y_ref = wrapper(*inference_inputs)

    # Load and run the exported program.
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method_names = _get_method_names(program)
    if "forward" in method_names:
        method_name = "forward"
    elif method_names:
        method_name = method_names[0]
        warnings.warn(
            f"'forward' not found in exported program; using '{method_name}' instead.",
            stacklevel=2,
        )
    else:
        raise RuntimeError(f"Exported program has no runnable methods. pte_path='{pte_path}'")

    y_et = program.load_method(method_name).execute(list(inference_inputs))[0]
    if not isinstance(y_et, torch.Tensor):
        y_et = torch.tensor(y_et)

    err = (y_ref - y_et).abs()
    mse = ((y_ref - y_et) ** 2).mean()
    return {
        "max_abs_err":  float(err.max().item()),
        "mean_abs_err": float(err.mean().item()),
        "mse":          float(mse.item()),
        "rmse":         float(torch.sqrt(mse).item()),
    }


def _get_method_names(program: Any) -> list[str]:
    attr = getattr(program, "method_names", None)
    if callable(attr):
        return list(attr())
    return list(attr) if attr is not None else []


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
    """Run post-export sanity checks and assemble the final result object."""

    # --- Weight mismatch check ---
    mismatches: list[dict[str, Any]] = []
    if run_weight_mismatch_check and exported_for_mismatch is not None:
        try:
            mismatches = find_weight_mismatches(wrapper, exported_for_mismatch, eps=weight_mismatch_eps)
        except Exception as exc:
            warnings.warn(f"Weight mismatch check skipped: {exc}", stacklevel=2)
    elif run_weight_mismatch_check and precision == "full":
        warnings.warn(
            "Weight mismatch check skipped for precision='full' (no quantized frozen params).",
            stacklevel=2,
        )

    if mismatches:
        worst = max(mismatches, key=lambda x: x["max_abs_err"])
        warnings.warn(
            f"Weight mismatch above eps: count={len(mismatches)}, "
            f"worst={worst['matched_weight']} max_abs_err={worst['max_abs_err']:.6g} "
            f"(eps={weight_mismatch_eps:.6g})",
            stacklevel=2,
        )

    # --- Inference stats ---
    stats = None
    if verbose:
        stats = run_simple_inference_stats(wrapper, pte_path, example_inputs)
        if stats is not None:
            print("[ExecuTorch export] Inference error statistics")
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
