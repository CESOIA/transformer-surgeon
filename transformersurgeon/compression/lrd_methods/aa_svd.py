import torch
from typing import Tuple


def low_rank_aa_svd(
    weight: torch.Tensor,
    rank: int,
    cross_covariance: torch.Tensor,
    shifted_covariance: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    in_features = weight.size(1)
    device = weight.device

    if cross_covariance.shape != (in_features, in_features):
        raise ValueError(
            f"Cross-covariance shape must be {(in_features, in_features)}, but got {tuple(cross_covariance.shape)}."
        )
    if shifted_covariance.shape != (in_features, in_features):
        raise ValueError(
            f"Shifted covariance shape must be {(in_features, in_features)}, but got {tuple(shifted_covariance.shape)}."
        )

    cross_covariance = cross_covariance.to(device=device, dtype=torch.float32)
    shifted_covariance = shifted_covariance.to(device=device, dtype=torch.float32)
    weight_f32 = weight.float()

    shifted_covariance = (shifted_covariance + shifted_covariance.transpose(0, 1)) * 0.5

    evals, evecs = torch.linalg.eigh(shifted_covariance)
    evals = torch.flip(evals, dims=(0,))
    evecs = torch.flip(evecs, dims=(1,))

    evals = torch.clamp_min(evals, eps)
    inv_sqrt_evals = torch.rsqrt(evals)

    # AA-SVD objective: min_rank<=k ||W X - W' X'||_F^2,
    # using C = X X'^T and S = X' X'^T with EVD-based whitening of S.
    M = weight_f32 @ cross_covariance @ (evecs * inv_sqrt_evals.unsqueeze(0))

    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    sqrt_S_r = torch.sqrt(S_r)

    L = U_r * sqrt_S_r.unsqueeze(0)
    R = sqrt_S_r.unsqueeze(1) * (
        Vh_r @ (inv_sqrt_evals.unsqueeze(1) * evecs.transpose(0, 1))
    )

    return L.to(weight.dtype).contiguous(), R.to(weight.dtype).contiguous()


__all__ = ["low_rank_aa_svd"]
