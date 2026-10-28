"""PyTorch 2 export (PT2E) quantisation flow shared by the ExecuTorch backends.

The backends build a PT2E quantiser, insert observers, run a short calibration,
and then overwrite the observers' parameters with the exact scales decided by
the compression step (so the device uses the same numbers as the compressed
model).
"""

from typing import Any

import torch
import torch.fx
import torch.nn as nn

from .quant_info import _PRECISION_TO_EFFECTIVE_DENOM


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


_LINEAR_TARGETS = (torch.ops.aten.linear.default,)


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
        # A weight observer observes a constant AND feeds a linear's weight
        # argument directly. Other constants (e.g. RMSNorm weights, observed when
        # a backend quantizes every op) must not receive a linear's scales.
        is_weight_obs = (
            input_node is not None and input_node.op == 'get_attr'
            and any(u.op == 'call_function' and u.target in _LINEAR_TARGETS
                    and len(u.args) > 1 and u.args[1] is node for u in node.users)
        )

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
    if info.get("group_size") is not None:
        # Per-block observer (e.g. QNN LPBQ): keep the (out, in // group_size) layout.
        scale = info["scale"].float().reshape(info["scale"].shape[0], -1)
        zero_point = torch.zeros(scale.shape, dtype=torch.int32)
        observer.calculate_qparams = lambda: (scale, zero_point)
        return
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
    # Match input_ids' shape to whatever resolve_components_and_wrapper actually
    # built it as (see build_example_inputs' add_batch_dim) rather than
    # re-deciding independently here.
    add_batch_dim = bool(example_inputs is not None and example_inputs[0].dim() == 2)

    # io cache template (empty tuple for mutable / legacy contract).
    cache_template: tuple[Any, ...] = ()
    if example_inputs is not None and len(example_inputs) > 2:
        cache_template = tuple(example_inputs[2:])

    def _fresh_caches() -> tuple[Any, ...]:
        return tuple([t.clone() for t in group] for group in cache_template)

    token_batches: list[list[int]] = []
    user_ids = getattr(config, "calibration_token_ids", None)
    if user_ids is not None:
        token_batches = [[int(t) for t in seq][:max_pos] for seq in user_ids]
    elif model_name:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name)
            for sent in _CAL_SENTENCES:
                ids = tok.encode(sent, add_special_tokens=True)
                token_batches.append(ids[:max_pos])
        except Exception:
            pass

    def _reset_mutable_caches():
        for name, buf in prepared.named_buffers():
            if "cache" in name.rsplit(".", 1)[-1]:
                buf.zero_()

    with torch.no_grad():
        if token_batches:
            # Real teacher-forced decode: positions 0..n-1 with the KV cache filling up,
            # so the attention/cache observers see the ranges they see at run time.
            # (Every token at pos 1 with an empty cache left them uncalibrated:
            # Qwen2.5-1.5B 16a8w all-op, KL 0.96 -> 0.57 vs the unquantized-activation model.)
            for ids in token_batches:
                _reset_mutable_caches()
                for pos_i, token_id in enumerate(ids):
                    inp = torch.tensor([[token_id]] if add_batch_dim else [token_id], dtype=torch.long)
                    pos = torch.tensor([pos_i], dtype=torch.long)
                    prepared(inp, pos, *_fresh_caches())
            _reset_mutable_caches()
        else:
            for _ in range(32):
                rand_shape = (1, 1) if add_batch_dim else (1,)
                rand_ids = torch.randint(0, vocab_size, rand_shape, dtype=torch.long)
                # Cache slots are 0 .. max_seq_len - 1 (randint's high is exclusive).
                rand_pos = torch.randint(0, max_pos, (1,), dtype=torch.long)
                prepared(rand_ids, rand_pos, *_fresh_caches())


__all__ = ["inject_scales_into_pt2e_observers", "calibrate_pt2e_observers"]
