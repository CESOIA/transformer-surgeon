import warnings
from typing import Any

import torch
import torch.nn as nn

from .validators import boolean, one_of, positive_int, positive_number
from .abstract import Compressor
from .quantization_methods import METHOD_FUNCTIONS, ACT_METHOD_FUNCTIONS, activation_qrange
from .unstructured_pruning_methods import METHOD_FUNCTIONS as SPARSITY_FUNCTIONS
from .unstructured_pruning import validate_unstructured_pruning_ratio


VALID_METHODS = list(METHOD_FUNCTIONS)  # quantization_methods/ is the single list
# "dynamic_per_token": int8 asymmetric per-token scales computed at run time
# (no calibration) -- the activation scheme of ExecuTorch/XNNPACK 8daXw kernels.
VALID_ACT_METHODS = ["maxmin", "dynamic_per_token"]
VALID_SPARSE_METHODS = ["magnitude", "random"]
VALID_ACTIVATION_SCHEMES = ["symmetric", "asymmetric"]
VALID_GRANULARITIES = ["per_tensor", "per_channel", "per_group"]
CALIBRATED_METHOD_DICT = {
    "gptq": ("covariance",),
}



# Runtime state that Quantizer.apply() leaves on a quantized layer. Outside this
# module it is read only through export_quant_info / copy_quant_state (the
# converter also carries the buffers in the state dict); strip_runtime_state
# removes it before saving.
QUANT_RUNTIME_BUFFERS = (
    "_act_quant_scale", "_act_quant_zero_point",
    "_act_out_quant_scale", "_act_out_quant_zero_point",
    "_torchao_scale",
)
QUANT_RUNTIME_ATTRS = (
    "_torchao_precision", "_torchao_per_channel", "_torchao_group_size", "_torchao_group_scale_bits",
    "_torchao_act_dynamic",
    "_soft_quant_precision",
    "_act_quant_precision", "_act_quant_method", "_act_quant_scheme",
    "_act_out_quant_precision", "_act_out_quant_method", "_act_out_quant_scheme",
)


class Quantizer(Compressor):
    def __init__(self, config):
        self.config = config
        self.method = self.config["method"]
        self.precision = self.config["precision"]
        self.sparsity = self.config["sparsity"]
        self.sparse_method = self.config["sparse_method"]
        self.eps = self.config["eps"]
        self.granularity = self.config["granularity"]
        self.group_size = self.config.get("group_size", 128)
        # group_scale_bits: per_group scales restricted to channel_scale * m with an
        # integer m in [1, 2**bits] (4 = Qualcomm HTP LPBQ); None = free group scales.
        self.group_scale_bits = self.config.get("group_scale_bits", None)
        self.precision_activation = self.config["precision_activation"]
        self.method_activation = self.config["method_activation"]
        self.scheme_activation = self.config["scheme_activation"]
        self.quantize_output_activation = self.config.get("quantize_output_activation", True)
        self.calibration_store = None

    def set_calibration_store(self, calibration_data):
        self.calibration_store = calibration_data

    def needs_calibration(self):
        summaries = list(CALIBRATED_METHOD_DICT.get(self.method, ()))
        if self.precision_activation != "full" and self.method_activation != "dynamic_per_token":
            if "activation_range" not in summaries:
                summaries.append("activation_range")
            if self.quantize_output_activation and "output_activation_range" not in summaries:
                summaries.append("output_activation_range")
        return tuple(summaries)

    def apply(self, module, hard=False, soft_applied=False):
        precision_activation = self.precision_activation
        method_activation = self.method_activation
        scheme_activation = self.scheme_activation
        validate_precision(precision_activation)
        validate_activation_method(method_activation)
        validate_activation_scheme(scheme_activation)
        self.config["precision_activation"] = precision_activation
        self.config["method_activation"] = method_activation
        self.config["scheme_activation"] = scheme_activation

        if self._to_compress():
            method = self.method
            precision = self.precision
            sparsity = self.sparsity
            sparse_method = self.sparse_method
            eps = self.eps
            granularity = self.granularity

            validate_quantization_method(method)
            validate_precision(precision)
            validate_unstructured_pruning_ratio(sparsity)
            validate_sparse_method(sparse_method)
            validate_quantization_eps(eps)
            validate_quantization_granularity(granularity)
            validate_quantization_group_scale_bits(self.group_scale_bits)
            if self.group_scale_bits is not None and (granularity != "per_group" or method != "vanilla"):
                raise NotImplementedError("group_scale_bits (LPBQ-style group scales) needs "
                                          "granularity='per_group' and method='vanilla'.")
            self.config["group_scale_bits"] = self.group_scale_bits

            self.config["method"] = method
            self.config["precision"] = precision
            self.config["sparsity"] = sparsity
            self.config["sparse_method"] = sparse_method
            self.config["eps"] = eps
            self.config["granularity"] = granularity

            # Hard-path: use torchao's native quantize_() API, bypassing soft-quant.
            if hard:
                _apply_hard_quantization(
                    module, method, precision, granularity, self.group_size, eps,
                    self._calibration_kwargs(method),
                    group_scale_bits=self.group_scale_bits,
                    dynamic_activation=(precision_activation != "full"
                                        and method_activation == "dynamic_per_token"),
                )
            elif not soft_applied:
                # Soft-path: fake-quant via dequantize.
                for name, param in module.named_parameters():
                    if name not in ["weight", "linear_V.weight"]:
                        continue

                    calibration_kwargs = {}
                    if method in CALIBRATED_METHOD_DICT and self.calibration_store:
                        for key in CALIBRATED_METHOD_DICT[method]:
                            if key in self.calibration_store:
                                calibration_kwargs[key] = self.calibration_store[key]

                    # When quantizing the U factor (weight) of an LRD-decomposed layer,
                    # the effective input to U is h = V·x, not x itself.
                    # The correct Hessian is C_U = V @ E[x^Tx] @ V^T, derived from
                    # the single calibration pass covariance and the already-computed V.
                    if (
                        name == "weight"
                        and isinstance(getattr(module, "rank", "full"), int)
                        and "covariance" in calibration_kwargs
                    ):
                        V = module.linear_V.weight.data.float()
                        C = calibration_kwargs["covariance"].float()
                        calibration_kwargs = {**calibration_kwargs, "covariance": V @ C @ V.T}

                    if granularity == "per_group":
                        if method != "vanilla":
                            raise NotImplementedError("per_group granularity is only implemented for method='vanilla'.")
                        calibration_kwargs["group_size"] = self.group_size
                        if self.group_scale_bits is not None:
                            calibration_kwargs["group_scale_bits"] = self.group_scale_bits
                    qdata, scale, dequantize_fn = METHOD_FUNCTIONS[method](
                        param.data, precision, eps, granularity, **calibration_kwargs
                    )

                    qweight = dequantize_fn(qdata, scale)
                    if sparsity > 0.0:
                        if sparse_method == "magnitude":
                            mask = SPARSITY_FUNCTIONS[sparse_method](
                                param, pruning_ratio=sparsity
                            )
                        elif sparse_method == "random":
                            mask = SPARSITY_FUNCTIONS[sparse_method](
                                param, pruning_ratio=sparsity
                            )
                        else:
                            raise ValueError(f"Unsupported sparsity method '{sparse_method}'.")
                        param.data.copy_(qweight * mask + param.data * (~mask))
                    else:
                        param.data.copy_(qweight)
                module._soft_quant_precision = precision

        if precision_activation != "full" and method_activation == "dynamic_per_token":
            if precision_activation != 8:
                raise ValueError("dynamic_per_token activation quantization is int8 only.")
            if not hard:
                raise NotImplementedError("dynamic_per_token activations require hard=True.")
            return  # carried by the quantized weight tensor itself (no hooks, no calibration)

        # Input activation fake-quant emulation via forward pre-hook.
        if precision_activation != "full":
            if self.calibration_store is None or "activation_range" not in self.calibration_store:
                raise RuntimeError(
                    "Activation quantization requires calibration data. "
                    "Run run_compression_calibration() before calling apply()."
                )
            act_range = self.calibration_store["activation_range"]
            act_min = act_range["min"].float()
            act_max = act_range["max"].float()

            scale, zero_point = ACT_METHOD_FUNCTIONS[method_activation](
                act_min, act_max, precision_activation, scheme_activation
            )

            module.register_buffer("_act_quant_scale", scale)
            module.register_buffer("_act_quant_zero_point", zero_point)
            module._act_quant_precision = precision_activation
            module._act_quant_method = method_activation
            module._act_quant_scheme = scheme_activation

            # Remove any previously registered hook before adding a new one.
            if hasattr(module, "_act_quant_hook_handle"):
                module._act_quant_hook_handle.remove()

            module._act_quant_hook_handle = module.register_forward_pre_hook(
                _make_fake_quant_hook("_act_quant", precision_activation, scheme_activation)
            )

        # Output activation fake-quant emulation via forward post-hook.
        validate_quantize_output_activation(self.quantize_output_activation)
        self.config["quantize_output_activation"] = self.quantize_output_activation
        if precision_activation != "full" and self.quantize_output_activation:
            if self.calibration_store is None or "output_activation_range" not in self.calibration_store:
                raise RuntimeError(
                    "Output activation quantization requires calibration data. "
                    "Run run_compression_calibration() before calling apply()."
                )
            out_range = self.calibration_store["output_activation_range"]
            out_min = out_range["min"].float()
            out_max = out_range["max"].float()

            out_scale, out_zero_point = ACT_METHOD_FUNCTIONS[method_activation](
                out_min, out_max, precision_activation, scheme_activation
            )

            module.register_buffer("_act_out_quant_scale", out_scale)
            module.register_buffer("_act_out_quant_zero_point", out_zero_point)
            module._act_out_quant_precision = precision_activation
            module._act_out_quant_method = method_activation
            module._act_out_quant_scheme = scheme_activation

            # Remove any previously registered hook before adding a new one.
            if hasattr(module, "_act_out_quant_hook_handle"):
                module._act_out_quant_hook_handle.remove()

            module._act_out_quant_hook_handle = module.register_forward_hook(
                _make_fake_quant_hook("_act_out_quant", precision_activation, scheme_activation, output=True)
            )

    def _calibration_kwargs(self, method):
        kw = {}
        if method in CALIBRATED_METHOD_DICT and self.calibration_store:
            for key in CALIBRATED_METHOD_DICT[method]:
                if key in self.calibration_store:
                    kw[key] = self.calibration_store[key]
        return kw

    def restore(self, module):
        self.config["method"] = "vanilla"
        self.config["precision"] = "full"
        self.config["sparsity"] = 0.0
        self.config["sparse_method"] = "magnitude"
        self.config["granularity"] = "per_tensor"
        self.config["group_size"] = 128
        self.config["group_scale_bits"] = None
        self.config["precision_activation"] = "full"
        self.config["method_activation"] = "maxmin"
        self.config["scheme_activation"] = "asymmetric"
        self.config["quantize_output_activation"] = True
        self.strip_runtime_state(module)

    def strip_runtime_state(self, module):
        """Remove activation hooks and quantization buffers from *module*.

        Unlike restore(), this does NOT reset the compressor config — it only
        removes the runtime artifacts added by apply() so that the module's
        state_dict is clean for serialization.  Used by
        manager.prepare_for_save() before model.save_pretrained().
        """
        if hasattr(module, "_act_quant_hook_handle"):
            module._act_quant_hook_handle.remove()
            del module._act_quant_hook_handle
        if hasattr(module, "_act_out_quant_hook_handle"):
            module._act_out_quant_hook_handle.remove()
            del module._act_out_quant_hook_handle
        for attr in QUANT_RUNTIME_BUFFERS + QUANT_RUNTIME_ATTRS:
            if hasattr(module, attr):
                delattr(module, attr)

    def prepare_for_save(self, module):
        self.strip_runtime_state(module)

    def _to_compress(self):
        return self.precision != "full"

    def __repr__(self):
        return (
            f"Quantizer(method='{self.method}', precision={self.precision}, "
            f"sparsity={self.sparsity}, sparse_method='{self.sparse_method}', "
            f"method_activation='{self.method_activation}', precision_activation={self.precision_activation}, "
            f"scheme_activation='{self.scheme_activation}')"
        )


def _make_fake_quant_hook(prefix: str, precision: int, scheme: str, output: bool = False):
    """Hook that fake-quantises a module's input (forward pre-hook) or output (forward hook).

    Scale and zero point are read from the module buffers ``<prefix>_scale`` and
    ``<prefix>_zero_point`` at call time, so the hook follows the module across
    ``.to(device)`` and state-dict loads.
    """
    qmin, qmax = activation_qrange(precision, scheme)

    def fake_quant(module, x):
        s = getattr(module, f"{prefix}_scale").float()
        zp = getattr(module, f"{prefix}_zero_point").float()
        return ((torch.clamp(torch.round(x / s) + zp, qmin, qmax) - zp) * s).to(x.dtype)

    if output:
        return lambda module, inputs, out: fake_quant(module, out)
    return lambda module, inputs: (fake_quant(module, inputs[0]),) + inputs[1:]


### EXPORT CONTRACT ###
# The only readers of the runtime state above outside this module go through
# these two functions (export/convert.py and export/quant_info.py).

def export_quant_info(module: nn.Linear) -> dict | None:
    """Backend-independent description of the quantisation decided for one linear layer.

    Returns None for an unquantised layer, otherwise a dict with ``scale``
    (per-channel/per-group weight scale; None for soft quantisation),
    ``precision``, ``hard``, ``per_channel``, ``group_size``,
    ``group_scale_bits``, ``act_dynamic`` and the ``act_*`` / ``act_out_*``
    activation keys (``act_scale``/``act_out_scale`` None when not calibrated).

    Side effects: a hard-quantised weight is dequantised in place (so
    ``torch.export`` sees a plain float tensor) and the activation fake-quant
    hooks are removed (the backend quantiser takes over).
    """
    try:
        from torchao.dtypes import AffineQuantizedTensor
    except ImportError:
        AffineQuantizedTensor = None
    info = _extract_weight_quant_info(module, AffineQuantizedTensor)
    if info is not None:
        _extract_act_quant_info(module, info)
    return info


def copy_quant_state(old: nn.Module, new: nn.Module) -> None:
    """Copy the quantisation state of linear layer ``old`` onto its converted copy ``new``.

    Tensor state (``QUANT_RUNTIME_BUFFERS``) travels with the state dict; this
    copies the plain attributes and, if ``old`` still holds a live torchao
    weight, stashes its scale on ``new``.
    """
    for attr in QUANT_RUNTIME_ATTRS:
        if hasattr(old, attr):
            setattr(new, attr, getattr(old, attr))

    # Fallback: if the weight is still a live quantized tensor and the scale was
    # not already stashed into the state dict, extract and register it now.
    w_old = old.weight
    if "_torchao_scale" not in new._buffers and isinstance(w_old, torch.Tensor) and type(w_old) is not torch.Tensor:
        scale = extract_torchao_scale(w_old)
        if scale is not None:
            new.register_buffer("_torchao_scale", scale)

    # Derive _torchao_per_channel from the scale buffer (authoritative source).
    if "_torchao_scale" in new._buffers:
        new._torchao_per_channel = new._torchao_scale.numel() > 1 and not hasattr(new, "_torchao_group_size")


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
        # LRD factors are quantized separately (quantization runs after LRD):
        # this module's own weight is the U factor [out, rank] and its stashed
        # scale has one row per output row; linear_V is visited as its own
        # nn.Linear. A stash whose rows do not match the weight belongs to a
        # pre-LRD weight -- refuse instead of exporting the layer float.
        w_rows = module.weight.shape[0]
        if has_stashed_scale and module._torchao_scale.dim() == 2 and module._torchao_scale.shape[0] not in (1, w_rows):
            raise RuntimeError(
                f"{module}: stashed quantization scale {tuple(module._torchao_scale.shape)} does not match "
                f"weight {tuple(module.weight.shape)} (quantized before LRD?).")

        if has_stashed_scale and not is_live_aqt:
            w = module.weight
            if hasattr(w, "dequantize") and type(w) is not torch.Tensor and type(w) is not nn.Parameter:
                module.weight = nn.Parameter(w.dequantize(), requires_grad=False)
            elif isinstance(w, nn.Parameter) and hasattr(w.data, "dequantize") and type(w.data) is not torch.Tensor:
                module.weight = nn.Parameter(w.data.dequantize(), requires_grad=False)

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

        group_size = getattr(module, '_torchao_group_size', None)
        return {
            "scale":       scale,
            "precision":   module._torchao_precision,
            "hard":        True,
            "per_channel": per_channel and group_size is None,
            # per-group: scale is (out_features, in_features // group_size)
            "group_size":  group_size,
            # LPBQ-style group scales (channel scale x integer in [1, 2**bits]), or None
            "group_scale_bits": getattr(module, '_torchao_group_scale_bits', None),
            # int8 asymmetric per-token dynamic activations (XNNPACK 8daXw)
            "act_dynamic": bool(getattr(module, '_torchao_act_dynamic', False)),
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
            "group_size":  None,
            "act_dynamic": False,
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



def extract_torchao_scale(w: torch.Tensor) -> torch.Tensor | None:
    """Extract the quantization scale from any torchao quantized weight tensor.

    Tries multiple attribute paths to handle old AffineQuantizedTensor (tensor_impl.scale)
    and new-style subclasses (direct .scale attribute).
    """
    if hasattr(w, 'tensor_impl') and hasattr(w.tensor_impl, 'scale'):
        return w.tensor_impl.scale.detach().clone()
    scale = getattr(w, 'scale', None)
    if isinstance(scale, torch.Tensor):
        return scale.detach().clone()
    return None


def _apply_hard_quantization(module, method, precision, granularity, group_size, eps,
                             calibration_kwargs, dynamic_activation=False, group_scale_bits=None) -> None:
    """Hard quantization computed by tsurgeon's own method functions.

    The integer weights and scales come from METHOD_FUNCTIONS[method] (vanilla
    RTN, GPTQ, HQQ); torchao is only the *container*: each weight becomes an
    ``IntxUnpackedToInt8Tensor`` holding exactly those integers/scales (scale
    stored in the weight dtype), so the eager model and every exporter see the
    same numbers. For an LRD layer both factors are quantized (the U factor
    with the covariance projected through V, as in the soft path).
    ``dynamic_activation`` marks the tensor for int8 asymmetric per-token
    dynamic activation quantization (eager simulation == XNNPACK 8daXw).
    """
    from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
        IntxUnpackedToInt8Tensor,
        IntxUnpackedToInt8TensorActivationQuantization as _Act,
    )

    if not (isinstance(precision, int) and 2 <= precision <= 8):
        raise ValueError(f"Hard quantization needs an int precision in [2, 8]; got {precision!r}.")
    if method == "gptq" and granularity == "per_group":
        raise NotImplementedError("GPTQ with per_group granularity is not implemented.")

    targets = [(module, "weight", calibration_kwargs)]
    V = getattr(module, "linear_V", None)
    if V is not None:
        kw_u = dict(calibration_kwargs)
        if "covariance" in kw_u:
            v = V.weight.data.float()
            kw_u["covariance"] = v @ kw_u["covariance"].float().to(v.device) @ v.T
        targets = [(module, "weight", kw_u), (V, "weight", calibration_kwargs)]

    for mod, attr, kw in targets:
        w = getattr(mod, attr).data
        if type(w) is not torch.Tensor:
            raise RuntimeError(f"{type(mod).__name__}.{attr} is already quantized.")
        extra = dict(kw)
        if granularity == "per_group":
            extra["group_size"] = group_size
            if group_scale_bits is not None:
                extra["group_scale_bits"] = group_scale_bits
        qdata, scale, _ = METHOD_FUNCTIONS[method](w, precision, eps, granularity, **extra)
        n, k = w.shape
        if granularity == "per_group":
            scale2d, block = scale.reshape(n, k // group_size), (1, group_size)
        elif granularity == "per_channel":
            scale2d, block = scale.reshape(n, 1), (1, k)
        else:
            scale2d, block = scale.reshape(1, 1), (n, k)
        scale2d = scale2d.to(w.dtype).clamp_min(torch.finfo(w.dtype).tiny).contiguous()
        qw = IntxUnpackedToInt8Tensor(
            qdata=qdata.to(torch.int8).contiguous(),
            scale=scale2d,
            zero_point=torch.zeros_like(scale2d, dtype=torch.int8),
            target_dtype=getattr(torch, f"int{precision}"),
            block_size=block,
            dtype=w.dtype,
            activation_quantization=_Act.INT8_ASYM_PER_TOKEN if dynamic_activation else None,
        )
        setattr(mod, attr, nn.Parameter(qw, requires_grad=False))
        mod._torchao_precision = precision
        mod.register_buffer("_torchao_scale", scale2d.clone())
        mod._torchao_per_channel = granularity == "per_channel"
        if granularity == "per_group":
            mod._torchao_group_size = int(group_size)
            if group_scale_bits is not None:
                mod._torchao_group_scale_bits = int(group_scale_bits)
        mod._torchao_act_dynamic = bool(dynamic_activation)


validate_quantization_method = one_of("Quantization method", VALID_METHODS)
validate_activation_method = one_of("Activation quantization method", VALID_ACT_METHODS)
validate_precision = positive_int("Precision", also=["full", "binary"], low=2, high=16)
validate_sparse_method = one_of("Sparse method", VALID_SPARSE_METHODS)
validate_quantization_eps = positive_number("Quantization eps")
validate_quantization_granularity = one_of("Quantization granularity", VALID_GRANULARITIES)
validate_quantization_group_size = positive_int("Quantization group_size")
validate_quantization_group_scale_bits = positive_int("group_scale_bits", allow_none=True, high=8)
validate_quantize_output_activation = boolean("quantize_output_activation")
validate_activation_scheme = one_of("Activation scheme", VALID_ACTIVATION_SCHEMES)

__all__ = [
    "Quantizer",
    "export_quant_info",
    "copy_quant_state",
    "validate_quantization_method",
    "validate_activation_method",
    "validate_precision",
    "validate_sparse_method",
    "validate_quantization_granularity",
    "validate_quantization_group_size",
    "validate_quantization_group_scale_bits",
    "validate_activation_scheme",
    "validate_quantize_output_activation",
]
