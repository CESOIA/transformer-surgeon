"""
Backend-agnostic export utilities shared by every backend exporter.

This module is the *parent* layer of the export package.  It holds everything
that does not depend on a concrete backend runtime (ExecuTorch, TensorRT, …):

  * model normalisation / tracing wrapper
  * per-layer compression-metadata extraction
  * the PT2E observer scale-injection + calibration machinery
  * generic post-export weight-mismatch checks and the result container

Concrete backends live in sibling sub-packages (``executorch_exporters``,
``tensorrt``) and import from here.  Anything backend-specific (a runtime
loader, a partitioner, a backend quantizer) stays in the backend package.

Export pipeline (followed by each backend exporter):

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

  3. <backend>.build_quantizer(...)   [quantised models only]
       Each backend builds its own PT2E quantizer from the per-layer info.

  4. prepare_pt2e / calibrate_pt2e_observers / inject_scales_into_pt2e_observers
       Standard PT2E flow.  For hard-quantised layers the observer min/max
       values are overridden with the exact torchao scales so no re-calibration
       is needed.  For soft-quantised layers the calibration-derived values are
       kept.  Stored activation calibration values are injected similarly.

  5. convert_pt2e / torch.export   [quantised models only]

  6. finalize_export_result()
       Optional weight-mismatch check and inference stats, then wraps
       everything into a BackendExportResult (or a backend subclass of it).
"""

import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.fx
import torch.nn as nn

from .config import BackendExportConfig


# ---------------------------------------------------------------------------
# Quantization range tables
# ---------------------------------------------------------------------------

# Symmetric per-channel weight quantization ranges by bit-width.  These match
# the ranges the PT2E observers use and are backend-agnostic.
_PRECISION_TO_QRANGE: dict[int, dict[str, int]] = {
    4: {"weight_qmin": -8,   "weight_qmax": 7},
    8: {"weight_qmin": -127, "weight_qmax": 127},
}

# Denominator for converting a per-channel torchao scale back to observer min/max,
# chosen to match the PT2E observer's own formula: (weight_qmax - weight_qmin) / 2.
# Injection sets max_abs = scale * denom so the observer reproduces pt2e_scale = scale exactly.
_PRECISION_TO_EFFECTIVE_DENOM: dict[int, float] = {
    4: 7.5,   # (7 - (-8)) / 2
    8: 127.0, # (127 - (-127)) / 2
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
class BackendExportResult:
    """Backend-agnostic export result.

    ``output_path`` is the on-disk artifact (an ExecuTorch ``.pte`` file, a
    TensorRT engine/program, …).  Backends may subclass this to add
    backend-flavoured aliases (see ExecuTorchExportResult.pte_path).
    """

    output_path: str
    backend: str
    precision: str
    weight_mismatches: list[dict[str, Any]]
    inference_stats: dict[str, float] | None = None


@dataclass
class ExporterConfig(BackendExportConfig, ABC):
    """Abstract base config shared by all torch.export-based backend exporters."""

    dynamic_shapes: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def build_zero_caches(decoder: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build per-block zero KV caches for the ``io_*`` cache implementations.

    Introspects each block's attention module for its (possibly per-layer)
    cache geometry, so heterogeneous shapes work without re-plumbing.
    Returns ``([], [])`` for the mutable path (no cache I/O).
    """
    if getattr(decoder, "cache_impl", "mutable") == "mutable":
        return [], []
    key_caches, value_caches = [], []
    for block in decoder.blocks:
        attn = block.attn
        shape = (attn.max_cache_length, attn.kv_num_heads, attn.head_dim)
        key_caches.append(torch.zeros(shape, dtype=attn.dtype))
        value_caches.append(torch.zeros(shape, dtype=attn.dtype))
    return key_caches, value_caches


class LLMWrapper(nn.Module):
    """Thin wrapper that presents the three model components under a fixed
    forward signature that torch.export can trace with a single token input.

    The signature depends on the decoder's ``cache_impl``:
      - ``mutable``: ``forward(input_ids, pos_id)`` -> ``logits`` (KV cache is
        internal module state; unchanged legacy contract).
      - ``io_*``: ``forward(input_ids, pos_id, key_caches, value_caches)`` ->
        ``(logits, new_key_caches, new_value_caches)`` (KV cache is graph I/O).
    """

    def __init__(self, embedding: nn.Module, decoder: nn.Module, final_layer: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.final_layer = final_layer
        self.cache_impl = getattr(decoder, "cache_impl", "mutable")

    def forward(
        self,
        input_ids: torch.LongTensor,
        pos_id_tensor: torch.LongTensor,
        key_caches: list[torch.Tensor] | None = None,
        value_caches: list[torch.Tensor] | None = None,
    ):
        if self.cache_impl == "mutable":
            hidden = self.decoder(self.embedding(input_ids), pos_id=pos_id_tensor)
            return self.final_layer(hidden[-1, :])

        hidden, new_key_caches, new_value_caches = self.decoder(
            self.embedding(input_ids),
            pos_id=pos_id_tensor,
            key_caches=key_caches,
            value_caches=value_caches,
        )
        logits = self.final_layer(hidden[-1, :])
        return logits, new_key_caches, new_value_caches


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
    # For io_* cache modes, the KV cache is explicit graph I/O: append per-block
    # zero caches to the example inputs so torch.export traces the full contract.
    if getattr(wrapper, "cache_impl", "mutable") != "mutable":
        key_caches, value_caches = build_zero_caches(wrapper.decoder)
        example_inputs = (*example_inputs, key_caches, value_caches)

    if getattr(config, "dynamic_shapes", None) is not None:
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
        #
        # If LRD was applied after quantization, the stashed scale belongs to
        # the original full weight W[out, in].  After LRD the layer holds two
        # factors: weight[out, rank] and linear_V.weight[rank, in].  PT2E's
        # quantizer would scope both under the same qconfig and try to apply
        # the out-channel scale to linear_V.weight, causing a shape mismatch.
        # Skip such layers — their weights are already dequantised floats and
        # the old scale is no longer valid for the decomposed factors.
        if getattr(module, 'linear_V', None) is not None:
            return None

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
    """Populate activation quantization keys in *info* and remove the fake-quant hooks.

    Extracts both input-side and output-side activation calibration.  If a side
    has no stored calibration its *_scale key is set to None.
    """
    # Input activation (pre-linear).
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

    # Output activation (post-linear).
    if hasattr(module, '_act_out_quant_scale'):
        info["act_out_scale"]      = module._act_out_quant_scale.detach().clone()
        info["act_out_zero_point"] = module._act_out_quant_zero_point.detach().clone()
        info["act_out_precision"]  = module._act_out_quant_precision
        info["act_out_scheme"]     = module._act_out_quant_scheme
        if hasattr(module, '_act_out_quant_hook_handle'):
            module._act_out_quant_hook_handle.remove()
            del module._act_out_quant_hook_handle
    else:
        info["act_out_scale"] = None


# ---------------------------------------------------------------------------
# PT2E observer scale injection
# ---------------------------------------------------------------------------

def _layer_names_from_observer(
    obs_node: torch.fx.Node,
    layer_info: dict[str, dict[str, Any]],
    stop_at_first: bool = False,
    _diagnosed: list | None = None,
) -> list[str]:
    """Walk downstream from an observer node and collect layer names from all reachable
    linear ops via nn_module_stack metadata.

    stop_at_first=True returns after the first linear found (weight observers always
    feed exactly one linear). stop_at_first=False collects all reachable linears,
    needed for activation observers shared by multiple projections (e.g. gate+up).
    """
    _linear_targets = {
        getattr(torch.ops.aten.linear, 'default', None),
        getattr(torch.ops.aten.mm, 'default', None),
        getattr(torch.ops.aten.addmm, 'default', None),
    } - {None}

    found: list[str] = []
    visited: set[torch.fx.Node] = {obs_node}
    frontier = list(obs_node.users)

    for depth in range(6):
        next_frontier: list[torch.fx.Node] = []
        for n in frontier:
            if n in visited:
                continue
            visited.add(n)
            if n.op == 'call_function' and n.target in _linear_targets:
                stack = n.meta.get('nn_module_stack', {})
                if _diagnosed is not None and not _diagnosed:
                    _diagnosed.append({
                        'obs_node': obs_node.name,
                        'linear_node': n.name,
                        'linear_target': str(n.target),
                        'depth': depth,
                        'stack_keys': list(stack.keys()),
                        'stack_values': [(str(v) if not isinstance(v, tuple) else v) for v in stack.values()],
                    })
                for _, (qname, _type) in reversed(list(stack.items())):
                    if qname in layer_info:
                        found.append(qname)
                        break
                if stop_at_first and found:
                    return found
                # Don't descend past the linear node.
            else:
                next_frontier.extend(n.users)
        frontier = next_frontier
        if not frontier:
            break

    return found


def _layer_names_from_output_observer(
    obs_node: torch.fx.Node,
    layer_info: dict[str, dict[str, Any]],
) -> list[str]:
    """Check if an observer's direct predecessor is a linear and return the layer name.

    Output observers sit immediately downstream of the linear op in the PT2E graph:
      linear → [output_obs] → ...
    We only check the direct predecessor (depth 0) to avoid false positives: a
    deeper upstream walk from an input observer could reach the *previous* layer's
    linear through layer-norm or element-wise ops.  Returns the matching layer name
    in a list, or an empty list if the direct predecessor is not a known linear.
    """
    _linear_targets = {
        getattr(torch.ops.aten.linear, 'default', None),
        getattr(torch.ops.aten.mm, 'default', None),
        getattr(torch.ops.aten.addmm, 'default', None),
    } - {None}

    input_node = obs_node.args[0] if obs_node.args else None
    if input_node is None or not isinstance(input_node, torch.fx.Node):
        return []

    if input_node.op == 'call_function' and input_node.target in _linear_targets:
        stack = input_node.meta.get('nn_module_stack', {})
        for _, (qname, _type) in reversed(list(stack.items())):
            if qname in layer_info:
                return [qname]

    return []  # direct predecessor is not a known linear → input observer


def inject_scales_into_pt2e_observers(
    prepared_model: nn.Module,
    layer_info: dict[str, dict[str, Any]],
) -> int:
    """Inject pre-computed scales into PT2E observer modules.

    Identifies each observer by tracing downstream (input observers) or upstream
    (output observers) to the nearest linear op and reading the layer name from
    nn_module_stack.  Weight observers (input is a get_attr) receive torchao
    scales for hard-quantised layers.  Input activation observers receive the
    input-side scales computed by the transformer-surgeon calibration manager.
    Output activation observers receive the output-side scales.

    Returns the total number of observer modules overridden.
    """
    # Duck-type observer detection: every PT2E observer implements calculate_qparams,
    # regardless of whether it comes from torchao or torch.ao.quantization.
    def _is_observer(m: nn.Module) -> bool:
        return callable(getattr(m, 'calculate_qparams', None))

    hard_layers    = {k for k, v in layer_info.items() if v["hard"] and v["scale"] is not None}
    act_layers     = {k for k, v in layer_info.items() if v.get("act_scale") is not None}
    act_out_layers = {k for k, v in layer_info.items() if v.get("act_out_scale") is not None}
    if not hard_layers and not act_layers and not act_out_layers:
        return 0

    overridden       = 0
    matched_hard     = set()
    matched_act      = set()
    matched_act_out  = set()
    _diag: list = []

    for node in prepared_model.graph.nodes:
        if node.op != 'call_module':
            continue

        try:
            submod = prepared_model.get_submodule(node.target)
        except AttributeError:
            continue
        if not _is_observer(submod):
            continue

        input_node = node.args[0] if node.args else None
        is_weight_obs = input_node is not None and input_node.op == 'get_attr'

        if is_weight_obs:
            names = _layer_names_from_observer(node, layer_info, stop_at_first=True, _diagnosed=_diag)
            if names and names[0] in hard_layers:
                _inject_weight_observer(submod, layer_info[names[0]])
                overridden += 1
                matched_hard.add(names[0])
        else:
            # Check if this is an output observer (linear is immediately upstream).
            out_names = _layer_names_from_output_observer(node, layer_info)
            if out_names:
                # Output activation observer: inject output-side scale.
                lname = out_names[0]
                if lname in act_out_layers and lname not in matched_act_out:
                    _inject_act_observer(submod, layer_info[lname], use_out=True)
                    overridden += 1
                    matched_act_out.add(lname)
            else:
                # Input activation observer: may be shared across multiple linears
                # (e.g. gate_proj and up_proj share the same input in gated MLP).
                names = _layer_names_from_observer(node, layer_info, stop_at_first=False, _diagnosed=_diag)
                injected = False
                for lname in names:
                    if lname in act_layers and lname not in matched_act:
                        if not injected:
                            _inject_act_observer(submod, layer_info[lname])
                            overridden += 1
                            injected = True
                        matched_act.add(lname)

    unmatched_hard = hard_layers - matched_hard
    if unmatched_hard:
        if _diag:
            import pprint
            print("[inject_scales diagnostic] first observer→linear sample:")
            pprint.pprint(_diag[0])
        raise RuntimeError(
            f"Could not inject weight scales into PT2E observers for layers: {sorted(unmatched_hard)}. "
            "The observer graph structure may have changed — check _layer_name_from_observer."
        )
    unmatched_act = act_layers - matched_act
    if unmatched_act:
        raise RuntimeError(
            f"Could not inject activation scales into PT2E observers for layers: {sorted(unmatched_act)}. "
            "The observer graph structure may have changed — check _layer_name_from_observer."
        )
    unmatched_act_out = act_out_layers - matched_act_out
    if unmatched_act_out:
        raise RuntimeError(
            f"Could not inject output activation scales into PT2E observers for layers: {sorted(unmatched_act_out)}. "
            "The observer graph structure may have changed — check _layer_names_from_output_observer."
        )

    return overridden


def _inject_weight_observer(observer: nn.Module, info: dict) -> None:
    """Patch observer.calculate_qparams to return the exact torchao scale.

    Setting min_val/max_val and relying on calculate_qparams to recover the
    scale via (max_val - min_val) / (qmax - qmin) loses precision for some
    float16 scales due to floating-point rounding in the multiply-divide
    round-trip.  Directly overriding calculate_qparams avoids this entirely.
    """
    if info["precision"] not in _PRECISION_TO_EFFECTIVE_DENOM:
        return
    qrange = _PRECISION_TO_QRANGE[info["precision"]]
    scale = info["scale"].reshape(-1).float()
    zero_point = torch.zeros(scale.shape, dtype=torch.int32)
    if not info["per_channel"]:
        scale = scale.squeeze()
        zero_point = zero_point.squeeze()

    _scale_ref = scale
    _zp_ref = zero_point

    def _patched_calculate_qparams():
        return _scale_ref, _zp_ref

    observer.calculate_qparams = _patched_calculate_qparams


def _inject_act_observer(observer: nn.Module, info: dict, use_out: bool = False) -> None:
    """Patch observer.calculate_qparams to return exact calibrated activation scale/zp.

    HistogramObserver derives its qparams from an internal histogram built during
    the PT2E calibration pass, and ignores min_val / max_val attributes entirely.
    We must override calculate_qparams directly — the same approach used for weight
    observers — to guarantee the exported scale matches transformer-surgeon calibration.

    The calibration stores scale/zp in the unsigned convention [0, qmax_unsigned].
    Backend activation observers use signed int8 [-128, 127].  For the 8-bit case
    both ranges span 255 levels so the scale is identical; only the zero_point
    reference frame shifts: zp_signed = zp_unsigned + obs_qmin.  For other widths
    we recompute both from the recovered float [act_min, act_max] range.

    use_out=True selects the output-side calibration (act_out_*) for output observers.
    """
    key = "act_out" if use_out else "act"
    s  = info[f"{key}_scale"].float().squeeze()
    zp = info[f"{key}_zero_point"].float().squeeze()

    scheme    = info[f"{key}_scheme"]
    precision = info[f"{key}_precision"]

    if scheme == "symmetric":
        our_qmax = float(2 ** (precision - 1) - 1)
        our_qmin = -our_qmax
    else:
        our_qmax = float(2 ** precision - 1)
        our_qmin = 0.0

    # Recover the calibrated float activation range.
    act_min_f = float((our_qmin - zp) * s)
    act_max_f = float((our_qmax - zp) * s)

    # Clip extreme asymmetric outliers. With per-tensor asymmetric INT8, a single
    # rare token can push act_max to 100-150× |act_min| (observed: act_max=2016 in
    # Qwen2-0.5B layer 2 down_proj with 32-sample WikiText calibration). This yields
    # scale≈8, leaving only 3-4 effective INT8 levels for typical ±20 activations.
    # Cap the positive side to 10× the negative side when asymmetry exceeds that ratio.
    _act_min_abs = abs(act_min_f)
    _MAX_ASYMMETRY = 10.0
    _clipped = _act_min_abs > 0 and act_max_f > _act_min_abs * _MAX_ASYMMETRY
    if _clipped:
        act_max_f = _act_min_abs * _MAX_ASYMMETRY

    # Read the observer's native quantization range (may differ from ours).
    obs_qmin = float(getattr(observer, 'quant_min', our_qmin))
    obs_qmax = float(getattr(observer, 'quant_max', our_qmax))
    our_range = our_qmax - our_qmin
    obs_range = obs_qmax - obs_qmin

    if _clipped:
        # Clipping changed the range: recompute scale from the new float bounds.
        float_range = act_max_f - act_min_f
        obs_scale = torch.tensor(max(float_range / obs_range, 1e-8), dtype=torch.float32)
        obs_zp = torch.clamp(
            torch.round(torch.tensor(obs_qmin - act_min_f / obs_scale.item())),
            obs_qmin, obs_qmax,
        ).to(torch.int32)
    else:
        # No clipping: derive obs_scale from surgeon scale directly to avoid a
        # float64 round-trip (act_min_f / act_max_f are Python floats) that could
        # shift obs_scale by ~1e-7, which cascades over 24 layers.
        if obs_range != our_range:
            obs_scale = (s * our_range / obs_range).clamp(min=1e-8)
        else:
            obs_scale = s.clamp(min=1e-8)
        obs_zp = torch.clamp(
            torch.round(obs_qmin - torch.tensor(act_min_f) / obs_scale),
            obs_qmin, obs_qmax,
        ).to(torch.int32)

    # Patch calculate_qparams directly — HistogramObserver ignores min_val/max_val.
    _s_ref  = obs_scale.reshape(1)
    _zp_ref = obs_zp.reshape(1)

    def _patched_calculate_qparams():
        return _s_ref, _zp_ref

    observer.calculate_qparams = _patched_calculate_qparams

    # Also set min_val/max_val for MinMaxObserver-style observers that use them.
    observer.min_val = torch.tensor(act_min_f)
    observer.max_val = torch.tensor(act_max_f)


# ---------------------------------------------------------------------------
# PT2E calibration
# ---------------------------------------------------------------------------

_CAL_SENTENCES = [
    "The Eiffel Tower is located in Paris, France, and was built in 1889.",
    "Machine learning models learn patterns from large datasets to make predictions.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "Python is widely used for data science, machine learning, and automation tasks.",
    "The capital of Japan is Tokyo, which is one of the most populous cities on Earth.",
    "Quantum computers use quantum mechanical phenomena to perform complex calculations.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
    "Climate change is caused by greenhouse gas emissions from fossil fuels and deforestation.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "Artificial intelligence is transforming industries from healthcare to finance.",
    "Water covers approximately 71 percent of the Earth's surface.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
    "The Amazon rainforest produces 20 percent of the world's oxygen supply.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules.",
]


def calibrate_pt2e_observers(
    prepared: nn.Module,
    model_config: Any | None,
    config: Any,
    example_inputs: tuple[Any, ...] | None = None,
) -> None:
    """Run real-text calibration passes through the prepared PT2E model.

    Random token calibration leaves output activation observers with degenerate
    histograms because the distribution of gate_proj / up_proj outputs with
    uniformly-random token IDs is very different from real text.  Tokenizing a
    small set of fixed sentences produces realistic activation distributions for
    all observers.  Falls back to random passes if the tokenizer cannot be loaded.

    For ``io_*`` cache modes the prepared graph also takes KV caches; we detect
    this from ``example_inputs`` (len > 2) and pass fresh zero caches per step.
    """
    model_name = getattr(model_config, '_name_or_path', None) if model_config else None
    vocab_size = int(getattr(model_config, 'vocab_size', 32000)) if model_config else 32000
    max_pos = max(1, min(512, getattr(config, 'max_seq_len', 512)))

    # io cache template (empty tuple for mutable / legacy contract).
    cache_template: tuple[Any, ...] = ()
    if example_inputs is not None and len(example_inputs) > 2:
        cache_template = tuple(example_inputs[2:])

    def _fresh_caches() -> tuple[Any, ...]:
        return tuple([t.clone() for t in group] for group in cache_template)

    token_batches: list[list[int]] = []
    if model_name:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name)
            for sent in _CAL_SENTENCES:
                ids = tok.encode(sent, add_special_tokens=True)
                token_batches.append(ids)
        except Exception:
            pass

    with torch.no_grad():
        if token_batches:
            for ids in token_batches:
                for token_id in ids:
                    inp = torch.tensor([token_id], dtype=torch.long)
                    pos = torch.tensor([1], dtype=torch.long)
                    prepared(inp, pos, *_fresh_caches())
        else:
            for _ in range(32):
                rand_ids = torch.randint(0, vocab_size, (1,), dtype=torch.long)
                rand_pos = torch.randint(1, max_pos + 1, (1,), dtype=torch.long)
                prepared(rand_ids, rand_pos, *_fresh_caches())


# ---------------------------------------------------------------------------
# Post-export checks
# ---------------------------------------------------------------------------

def dequant_from_exported_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
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
    # Detect the model's native dtype from regular (non-quantized) float parameters.
    # This determines the precision at which torchao dequantizes INT4 weights.
    compare_dtype = torch.float32
    for _, tensor in wrapper.state_dict().items():
        if tensor.ndim >= 2 and type(tensor) is torch.Tensor and tensor.is_floating_point():
            compare_dtype = tensor.dtype
            break

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

        # Compare at the model's native dtype so torchao float16/float32 dequantization
        # and our float32 reconstruction are evaluated at the same precision.
        dq_cmp = dq.to(compare_dtype).float()
        best_name, best_max, best_mean = None, None, None
        for name, t in same_shape:
            err      = (t.to(compare_dtype).float() - dq_cmp).abs()
            max_err  = float(err.max().item())
            mean_err = float(err.mean().item())
            if best_max is None or max_err < best_max:
                best_name, best_max, best_mean = name, max_err, mean_err

        # Use mean_abs_err to detect systematic scale injection failures.
        # max_abs_err catches at most a few boundary elements off by 1 quant step
        # (e.g. torchao INT8 uses qmin=-128 while backends use qmin=-127), which
        # results in mean≈0 even when max≈1 step.  A real scale error affects
        # most elements and produces a large mean.
        if best_mean is not None and best_mean > eps:
            mismatches.append({
                "matched_weight": best_name,
                "max_abs_err":    best_max,
                "mean_abs_err":   best_mean,
                "eps":            eps,
            })

    return mismatches


def finalize_export_result(
    *,
    output_path: str,
    backend: str,
    precision: str,
    wrapper: nn.Module,
    exported_for_mismatch: Any | None,
    run_weight_mismatch_check: bool,
    weight_mismatch_eps: float,
    verbose: bool,
    inference_stats_fn: Callable[[], dict[str, float] | None] | None = None,
    result_cls: type = BackendExportResult,
) -> BackendExportResult:
    """Run post-export sanity checks and assemble the final result object.

    This is backend-agnostic.  Backends supply an ``inference_stats_fn`` closure
    (capturing whatever runtime handle they need) to compute float-vs-exported
    error statistics, and may pass a ``result_cls`` subclass of
    BackendExportResult for backend-flavoured field aliases.
    """

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
        worst = max(mismatches, key=lambda x: x["mean_abs_err"])
        warnings.warn(
            f"Weight mismatch above eps: count={len(mismatches)}, "
            f"worst={worst['matched_weight']} mean_abs_err={worst['mean_abs_err']:.6g} "
            f"(eps={weight_mismatch_eps:.6g})",
            stacklevel=2,
        )

    # --- Inference stats ---
    stats = None
    if verbose and inference_stats_fn is not None:
        stats = inference_stats_fn()
        if stats is not None:
            print(f"[{backend} export] Inference error statistics")
            print(f"  max_abs_err : {stats['max_abs_err']:.8f}")
            print(f"  mean_abs_err: {stats['mean_abs_err']:.8f}")
            print(f"  mse         : {stats['mse']:.10f}")
            print(f"  rmse        : {stats['rmse']:.8f}")

    return result_cls(
        output_path=output_path,
        backend=backend,
        precision=precision,
        weight_mismatches=mismatches,
        inference_stats=stats,
    )
