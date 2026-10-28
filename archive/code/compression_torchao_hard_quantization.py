"""ARCHIVED: _apply_torchao_hard_quantization from compression/quantization.py.

Removed during the 2026-10-28 refactor: nothing called it. It was superseded by
``_apply_hard_quantization``, which builds the torchao tensor from tsurgeon's own
integer weights and scales (so every exporter sees the same numbers). Kept for
reference only.
"""
import torch

from transformersurgeon.compression.quantization import extract_torchao_scale as _extract_aqt_scale


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
        scale = extract_torchao_scale(w)
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
