import torch
import torch.nn as nn
from typing import Union

from .abstract import Compressor
from .quantization_methods import METHOD_FUNCTIONS, ACT_METHOD_FUNCTIONS
from .unstructured_pruning_methods import METHOD_FUNCTIONS as SPARSITY_FUNCTIONS
from .unstructured_pruning import validate_unstructured_pruning_ratio


VALID_METHODS = ["vanilla", "gptq", "hqq"]
# "dynamic_per_token": int8 asymmetric per-token scales computed at run time
# (no calibration) -- the activation scheme of ExecuTorch/XNNPACK 8daXw kernels.
VALID_ACT_METHODS = ["maxmin", "dynamic_per_token"]
VALID_SPARSE_METHODS = ["magnitude", "random"]
VALID_ACTIVATION_SCHEMES = ["symmetric", "asymmetric"]
VALID_GRANULARITIES = ["per_tensor", "per_channel", "per_group"]
CALIBRATED_METHOD_DICT = {
    "gptq": ("covariance",),
}


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
        self.precision_activation = self.config["precision_activation"]
        self.method_activation = self.config["method_activation"]
        self.scheme_activation = self.config["scheme_activation"]
        self.calibration_store = None

    def set_calibration_store(self, calibration_data):
        self.calibration_store = calibration_data

    def needs_calibration(self):
        summaries = list(CALIBRATED_METHOD_DICT.get(self.method, ()))
        if self.precision_activation != "full" and self.method_activation != "dynamic_per_token":
            if "activation_range" not in summaries:
                summaries.append("activation_range")
            if "output_activation_range" not in summaries:
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
                _make_activation_fake_quant_hook(scale, zero_point, precision_activation, scheme_activation)
            )

        # Output activation fake-quant emulation via forward post-hook.
        if precision_activation != "full":
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
                _make_output_activation_fake_quant_hook(out_scale, out_zero_point, precision_activation, scheme_activation)
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
        self.config["precision_activation"] = "full"
        self.config["method_activation"] = "maxmin"
        self.config["scheme_activation"] = "asymmetric"
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
        for attr in (
            "_act_quant_scale",
            "_act_quant_zero_point",
            "_act_quant_precision",
            "_act_quant_method",
            "_act_quant_scheme",
            "_act_out_quant_scale",
            "_act_out_quant_zero_point",
            "_act_out_quant_precision",
            "_act_out_quant_method",
            "_act_out_quant_scheme",
            "_soft_quant_precision",
            "_torchao_precision",
            "_torchao_per_channel",
            "_torchao_group_size",
            "_torchao_act_dynamic",
            "_torchao_scale",
        ):
            if hasattr(module, attr):
                delattr(module, attr)

    def _to_compress(self):
        return self.precision != "full"

    def __repr__(self):
        return (
            f"Quantizer(method='{self.method}', precision={self.precision}, "
            f"sparsity={self.sparsity}, sparse_method='{self.sparse_method}', "
            f"method_activation='{self.method_activation}', precision_activation={self.precision_activation}, "
            f"scheme_activation='{self.scheme_activation}')"
        )


def _make_activation_fake_quant_hook(
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    precision: int,
    scheme: str,
):
    """Return a forward pre-hook that fake-quantizes the input activation."""
    if scheme == "symmetric":
        qmax = 2 ** (precision - 1) - 1
        qmin = -qmax
    else:
        qmax = 2 ** precision - 1
        qmin = 0

    zp = zero_point.float()
    s = scale.float()

    def hook(_, inputs):
        x = inputs[0]
        xq = torch.clamp(torch.round(x / s) + zp, qmin, qmax)
        x_fq = (xq - zp) * s
        return (x_fq.to(x.dtype),) + inputs[1:]

    return hook


def _make_output_activation_fake_quant_hook(
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    precision: int,
    scheme: str,
):
    """Return a forward post-hook that fake-quantizes the output activation."""
    if scheme == "symmetric":
        qmax = 2 ** (precision - 1) - 1
        qmin = -qmax
    else:
        qmax = 2 ** precision - 1
        qmin = 0

    zp = zero_point.float()
    s = scale.float()

    def hook(_, inputs, output):
        xq = torch.clamp(torch.round(output / s) + zp, qmin, qmax)
        x_fq = (xq - zp) * s
        return x_fq.to(output.dtype)

    return hook


def _extract_aqt_scale(w: torch.Tensor) -> torch.Tensor | None:
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
                             calibration_kwargs, dynamic_activation=False) -> None:
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
        mod._torchao_act_dynamic = bool(dynamic_activation)


def _apply_torchao_hard_quantization(module, precision, granularity: str, group_size: int = 128) -> None:
    """Apply torchao native weight-only quantization to *module* in-place."""
    try:
        from torchao.quantization import (
            quantize_,
            Int8WeightOnlyConfig,
            IntxWeightOnlyConfig,
        )
        from torchao.quantization.granularity import PerRow, PerTensor, PerAxis, PerGroup
    except ImportError as exc:
        raise ImportError(
            "torchao is required for hard quantization. "
            "Install it with: pip install torchao"
        ) from exc

    ao_granularity_row = PerRow()        # per-channel (output dim)
    ao_granularity_tensor = PerTensor()
    ao_granularity_axis = PerAxis(axis=0)  # used by IntxWeightOnlyConfig

    per_channel = granularity == "per_channel"

    if precision == "binary":
        raise ValueError(
            "Hard quantization does not support 'binary' precision. "
            "Use soft quantization (hard=False) for binary."
        )

    if isinstance(precision, int) and precision > 8:
        raise ValueError(
            f"Hard quantization requires precision ≤ 8; got {precision}."
        )

    if granularity == "per_group":
        # Symmetric max-abs scale per (output row, group of `group_size` inputs).
        if not (isinstance(precision, int) and 2 <= precision <= 8):
            raise ValueError(f"per_group hard quantization needs an int precision in [2, 8]; got {precision!r}.")
        in_features = module.weight.shape[-1]
        if in_features % group_size:
            raise ValueError(f"per_group: in_features {in_features} is not divisible by group_size {group_size}.")
        config = IntxWeightOnlyConfig(
            weight_dtype=getattr(torch, f"int{precision}"),
            granularity=PerGroup(group_size),
        )
    elif precision == 8:
        config = Int8WeightOnlyConfig(
            granularity=ao_granularity_row if per_channel else ao_granularity_tensor
        )
    elif isinstance(precision, int) and 2 <= precision <= 7:
        # IntxWeightOnlyConfig handles any sub-byte width (1–7 bits) including int4,
        # using a generic AffineQuantizedTensor path that does not require mslk.
        torch_dtype = getattr(torch, f"int{precision}")
        config = IntxWeightOnlyConfig(
            weight_dtype=torch_dtype,
            granularity=ao_granularity_axis,
        )
    else:
        raise ValueError(f"Unsupported precision for hard quantization: {precision!r}.")

    quantize_(module, config)

    quantized_any = False
    for m in module.modules():
        if not isinstance(m, nn.Linear):
            continue
        w = m.weight
        # Detect any torchao quantized weight: old AffineQuantizedTensor or new-style
        # subclasses (e.g. IntxWeightOnly) which don't inherit from AffineQuantizedTensor.
        if type(w) is torch.Tensor:
            continue
        if not isinstance(w, torch.Tensor):
            continue
        quantized_any = True
        m._torchao_precision = precision
        scale = _extract_aqt_scale(w)
        if scale is not None:
            m.register_buffer("_torchao_scale", scale)
            m._torchao_per_channel = scale.numel() > 1
            if granularity == "per_group":
                # scale is (out_features, in_features // group_size)
                m._torchao_group_size = int(group_size)

    if not quantized_any and not isinstance(module, nn.Linear):
        # torchao's quantize_() only dispatches to nn.Linear submodules in this
        # version -- on anything else (Conv2dCompressed/Conv3dCompressed,
        # EmbeddingCompressed, ...) it silently leaves the weight untouched.
        # Fail loudly instead of returning a model that looks quantized but isn't.
        raise NotImplementedError(
            f"Hard (torchao) quantization is not supported for {type(module).__name__} "
            "in this torchao version (quantize_() only targets nn.Linear and made no "
            "change here). Use soft quantization (hard=False) instead."
        )


### CONFIGURATION VALIDATORS ###

def validate_quantization_method(method: str) -> None:
    if method not in VALID_METHODS:
        raise ValueError(f"Quantization method must be one of {VALID_METHODS}, but got '{method}'.")


def validate_activation_method(method: str) -> None:
    if method not in VALID_ACT_METHODS:
        raise ValueError(f"Activation quantization method must be one of {VALID_ACT_METHODS}, but got '{method}'.")


def validate_precision(precision: Union[str, int]) -> None:
    if isinstance(precision, str):
        if precision not in ["full", "binary"]:
            raise ValueError(
                f"Precision must be an integer, 'full' or 'binary', but got '{precision}'."
            )
    elif isinstance(precision, int):
        if precision < 2 or precision > 16:
            raise ValueError(
                f"Precision must be a positive integer between 2 and 16, but got {precision}."
            )
    else:
        raise ValueError(
            f"Precision must be an integer, 'full' or 'binary', but got type {type(precision)}."
        )

def validate_sparse_method(sparse_method: str) -> None:
    if sparse_method not in VALID_SPARSE_METHODS:
        raise ValueError(f"Sparse method must be one of {VALID_SPARSE_METHODS}, but got '{sparse_method}'.")

def validate_quantization_eps(eps: float) -> None:
    if not isinstance(eps, (int, float)):
        raise ValueError(f"Quantization eps must be numeric, but got type {type(eps)}.")
    if eps <= 0:
        raise ValueError(f"Quantization eps must be positive, but got {eps}.")

def validate_quantization_granularity(granularity: str) -> None:
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"Quantization granularity must be one of {VALID_GRANULARITIES}, but got '{granularity}'."
        )

def validate_quantization_group_size(group_size: int) -> None:
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError(f"Quantization group_size must be a positive int, but got {group_size!r}.")

def validate_activation_scheme(scheme: str) -> None:
    if scheme not in VALID_ACTIVATION_SCHEMES:
        raise ValueError(
            f"Activation scheme must be one of {VALID_ACTIVATION_SCHEMES}, but got '{scheme}'."
        )

__all__ = [
    "Quantizer",
    "validate_quantization_method",
    "validate_activation_method",
    "validate_precision",
    "validate_sparse_method",
    "validate_quantization_granularity",
    "validate_quantization_group_size",
    "validate_activation_scheme",
]
