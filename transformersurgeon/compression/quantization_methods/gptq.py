import torch
from typing import Callable, Optional, Tuple, Union

from .vanilla import _dequantize_symmetric, _dequantize_grouped


def quantize_gptq(
    weight: torch.Tensor,
    precision: Union[str, int],
    eps: float = 1e-6,
    granularity: str = "per_channel",
    *,
    covariance: Optional[torch.Tensor] = None,
    block_size: int = 128,
    percdamp: float = 0.01,
    group_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Callable]:
    """GPTQ weight quantization (Frantar et al., 2022).

    Uses the inverse Hessian (H = 2 * X^T X) to spread quantization error
    across unquantized columns, minimizing the layer-wise reconstruction loss
    ||WX - W_q X||_F^2. Falls back to round-to-nearest when covariance is None.

    Args:
        weight: Float weight matrix of shape [out_features, in_features].
        precision: Bit-width (integer). Binary is not supported.
        eps: Scale clamp lower bound.
        granularity: ``"per_channel"`` (per output row), ``"per_group"`` (per output row and
                     ``group_size`` input columns) or ``"per_tensor"``.
        covariance: X^T X calibration matrix of shape [in_features, in_features].
                    If None, falls back to vanilla round-to-nearest.
        block_size: Number of columns processed per Cholesky block.
        percdamp: Hessian diagonal damping as a fraction of its mean.
        group_size: per_group only (default 128). As in the reference implementation, a
                    group's scale is taken from the error-updated weights when the sweep
                    reaches its first column.

    Returns:
        (qdata, scale, dequantize_fn) where qdata is int8, matching the
        signature of other quantization method functions.
    """
    if precision == "binary":
        raise ValueError("GPTQ does not support 'binary' precision.")

    W = weight.clone().float()
    out_features, in_features = W.shape
    qmax = 2 ** (precision - 1) - 1
    per_group = granularity == "per_group"
    per_channel = granularity == "per_channel"

    if per_group:
        g = group_size or 128
        if in_features % g:
            raise ValueError(f"per_group: in_features {in_features} not divisible by group_size {g}.")
        if covariance is None:
            Wg = W.reshape(out_features, in_features // g, g)
            scale = (Wg.abs().amax(dim=2) / qmax).clamp(min=eps)  # [out, groups]
            q = torch.clamp(torch.round(Wg / scale.unsqueeze(-1)), -qmax, qmax)
            return q.reshape(out_features, in_features).to(torch.int8), scale, _dequantize_grouped
        # Blocks hold whole groups, so a group's scale sees fully updated columns.
        block_size = -(-block_size // g) * g
        scale = torch.empty(out_features, in_features // g, device=W.device)
    else:
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
            col = block_start + j
            if per_group:
                if col % g == 0:
                    scale[:, col // g] = (W1[:, j:j + g].abs().amax(dim=1) / qmax).clamp(min=eps)
                s_col = scale[:, col // g]
                q = torch.clamp(torch.round(w / s_col), -qmax, qmax)
                q_float = q * s_col
            elif per_channel:
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
    if per_group:
        q_int = torch.clamp(torch.round(Q / scale.repeat_interleave(g, dim=1)), -qmax, qmax).to(torch.int8)
        return q_int, scale, _dequantize_grouped
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
