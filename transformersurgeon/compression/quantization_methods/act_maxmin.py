import torch
from typing import Tuple, Union


def compute_activation_scale_zp_maxmin(
    act_min: torch.Tensor,
    act_max: torch.Tensor,
    precision: Union[str, int],
    scheme: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scale and zero_point from calibrated min/max activation range.

    Returns scale (float) and zero_point (int32) following the standard
    affine quantization convention:
        q = clamp(round(x / scale) + zero_point, qmin, qmax)
        x_fq = (q - zero_point) * scale
    """
    if scheme == "symmetric":
        qmax = 2 ** (precision - 1) - 1
        scale = torch.max(act_min.abs(), act_max.abs()) / qmax
        zero_point = torch.zeros(1, dtype=torch.int32, device=scale.device)
    else:  # asymmetric
        qmax = 2 ** precision - 1
        scale = (act_max - act_min) / qmax
        zero_point = torch.clamp(torch.round(-act_min / scale), 0, qmax).to(torch.int32)

    scale = scale.clamp(min=1e-8)
    return scale, zero_point


__all__ = ["compute_activation_scale_zp_maxmin"]
