import torch
from typing import Callable, Optional, Tuple, Union

from .vanilla import _dequantize_symmetric


def quantize_gptq(
    weight: torch.Tensor,
    precision: Union[str, int],
    eps: float = 1e-6,
    granularity: str = "per_channel",
    *,
    covariance: Optional[torch.Tensor] = None,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
    """GPTQ weight quantization (Frantar et al., 2022).

    Uses the inverse Hessian (H = 2 * X^T X) to spread quantization error
    across unquantized columns, minimizing the layer-wise reconstruction loss
    ||WX - W_q X||_F^2. Falls back to round-to-nearest when covariance is None.

    Args:
        weight: Float weight matrix of shape [out_features, in_features].
        precision: Bit-width (integer). Binary is not supported.
        eps: Scale clamp lower bound.
        granularity: ``"per_channel"`` (per output row) or ``"per_tensor"``.
        covariance: X^T X calibration matrix of shape [in_features, in_features].
                    If None, falls back to vanilla round-to-nearest.
        block_size: Number of columns processed per Cholesky block.
        percdamp: Hessian diagonal damping as a fraction of its mean.

    Returns:
        (qdata, scale, dequantize_fn) where qdata is int8, matching the
        signature of other quantization method functions.
    """
    if precision == "binary":
        raise ValueError("GPTQ does not support 'binary' precision.")

    W = weight.clone().float()
    out_features, in_features = W.shape
    qmax = 2 ** (precision - 1) - 1
    per_channel = granularity == "per_channel"

    # Compute a fixed scale from the original weight (before Hessian corrections).
    if per_channel:
        scale = W.abs().amax(dim=1, keepdim=True) / qmax  # [out, 1]
    else:
        scale = W.abs().max() / qmax  # scalar
    scale = scale.clamp(min=eps)

    if covariance is None:
        q = torch.clamp(torch.round(W / scale), -qmax, qmax).to(torch.int8)
        return q, scale, _dequantize_symmetric

    device = W.device
    H = 2.0 * covariance.float().to(device)  # Hessian of squared reconstruction loss

    # Diagonal damping for numerical stability on low-rank calibration data.
    damp = percdamp * H.diagonal().mean()
    H.diagonal().add_(damp)

    # H_inv = upper Cholesky factor of H^{-1}, as used in Algorithm 1 of the paper.
    # This factored form allows efficient blocked column updates.
    H_inv = _cholesky_inv_upper(H, damp)

    Q = torch.zeros_like(W)  # accumulates dequantized output

    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)

        W1 = W[:, block_start:block_end].clone()   # [out, block_len]
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        H1 = H_inv[block_start:block_end, block_start:block_end]  # [block_len, block_len]

        for j in range(block_end - block_start):
            w = W1[:, j]          # [out_features]
            d = H1[j, j]          # diagonal of upper Cholesky factor at column j

            # Round to nearest integer, clamp, and dequantize.
            if per_channel:
                q = torch.clamp(torch.round(w / scale[:, 0]), -qmax, qmax)
                q_float = q * scale[:, 0]
            else:
                q = torch.clamp(torch.round(w / scale), -qmax, qmax)
                q_float = q * scale

            Q1[:, j] = q_float

            # Normalize error by diagonal entry and propagate through remaining block columns.
            err = (w - q_float) / d
            Err1[:, j] = err
            W1[:, j + 1:] -= err.unsqueeze(1) * H1[j, j + 1:].unsqueeze(0)

        Q[:, block_start:block_end] = Q1

        # Propagate accumulated block error to all columns right of this block.
        W[:, block_end:] -= Err1 @ H_inv[block_start:block_end, block_end:]

    # Re-encode Q (dequantized) back to int8. Because Q was built from exact
    # quantized values, this round-trip is lossless.
    q_int = torch.clamp(torch.round(Q / scale), -qmax, qmax).to(torch.int8)
    return q_int, scale, _dequantize_symmetric


def _cholesky_inv_upper(H: torch.Tensor, damp: float) -> torch.Tensor:
    """Return upper Cholesky factor of H^{-1}, with a retry on LinAlgError."""
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        return torch.linalg.cholesky(H_inv, upper=True)
    except torch.linalg.LinAlgError:
        H.diagonal().add_(damp * 10.0)
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        return torch.linalg.cholesky(H_inv, upper=True)


__all__ = ["quantize_gptq"]
