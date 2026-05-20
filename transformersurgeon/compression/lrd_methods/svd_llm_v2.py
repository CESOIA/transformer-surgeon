import torch
from typing import Tuple


def low_rank_svd_llm_v2(
    weight: torch.Tensor,
    rank: int,
    covariance: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    in_features = weight.size(1)
    device = weight.device

    if covariance.shape != (in_features, in_features):
        raise ValueError(
            f"Covariance shape must be {(in_features, in_features)}, but got {tuple(covariance.shape)}."
        )
    covariance = covariance.to(device=device, dtype=torch.float32)
    weight_f32 = weight.float()

    covariance = (covariance + covariance.transpose(0, 1)) * 0.5

    evals, evecs = torch.linalg.eigh(covariance)
    evals = torch.flip(evals, dims=(0,))
    evecs = torch.flip(evecs, dims=(1,))

    evals = torch.clamp_min(evals, eps)
    sqrt_evals = torch.sqrt(evals)
    inv_sqrt_evals = torch.rsqrt(evals)

    C_sqrt = evecs * sqrt_evals.unsqueeze(0)
    W_tilde = weight_f32 @ C_sqrt

    U, S, Vh = torch.linalg.svd(W_tilde, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    sqrt_S_r = torch.sqrt(S_r)

    L = U_r * sqrt_S_r.unsqueeze(0)
    R = sqrt_S_r.unsqueeze(1) * ((Vh_r * inv_sqrt_evals.unsqueeze(0)) @ evecs.transpose(0, 1))

    return L.to(weight.dtype).contiguous(), R.to(weight.dtype).contiguous()

__all__ = ["low_rank_svd_llm_v2"]