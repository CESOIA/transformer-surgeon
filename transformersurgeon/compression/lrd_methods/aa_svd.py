import warnings
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

    if not torch.all(torch.isfinite(cross_covariance)):
        raise RuntimeError("aa-svd: cross_covariance contains non-finite values.")
    if not torch.all(torch.isfinite(shifted_covariance)):
        raise RuntimeError("aa-svd: shifted_covariance contains non-finite values.")

    shifted_covariance = (shifted_covariance + shifted_covariance.transpose(0, 1)) * 0.5

    # Eigendecomposition-based whitening: more stable than Cholesky when S is
    # ill-conditioned because we clamp eigenvalues explicitly, bounding the whitening
    # amplification to 1/sqrt(eps * max_eval) without risking Cholesky failure.
    evals, evecs = torch.linalg.eigh(shifted_covariance)
    min_eval = evals[0].item()
    max_eval = evals[-1].clamp_min(0.0).item()

    floor = eps * max_eval + eps
    if min_eval < 0:
        warnings.warn(
            f"aa-svd: shifted_covariance has negative eigenvalue {min_eval:.3e}; "
            f"clamping to floor={floor:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif evals[0].item() < floor * 0.1:
        warnings.warn(
            f"aa-svd: shifted_covariance is ill-conditioned "
            f"(min_eval={min_eval:.3e}, max_eval={max_eval:.3e}, floor={floor:.3e}). "
            "Whitening will clamp small eigenvalues.",
            RuntimeWarning,
            stacklevel=2,
        )

    evals_safe = evals.clamp_min(floor)
    inv_sqrt_evals = evals_safe.rsqrt()  # 1 / sqrt(d_i)

    # W_tilde = W @ C @ S^{-1/2}  where  S^{-1/2} = V diag(inv_sqrt_evals) V^T
    WC = weight_f32 @ cross_covariance
    # (out, in) @ (in, in) -> project into eigenbasis, scale, project back
    WCV = WC @ evecs                                        # (out, in)
    W_tilde = (WCV * inv_sqrt_evals.unsqueeze(0)) @ evecs.T  # (out, in)

    if not torch.all(torch.isfinite(W_tilde)):
        raise RuntimeError("aa-svd: whitened matrix W_tilde contains non-finite values.")

    U, S, Vh = torch.linalg.svd(W_tilde, full_matrices=False)

    if not torch.all(torch.isfinite(S)):
        raise RuntimeError("aa-svd: SVD of W_tilde produced non-finite singular values.")

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    total_energy = S.pow(2).sum()
    captured_energy = S_r.pow(2).sum()
    if total_energy > 0:
        energy_ratio = (captured_energy / total_energy).item()
        if energy_ratio < 0.5:
            warnings.warn(
                f"aa-svd: rank={rank} captures only {energy_ratio*100:.1f}% of the "
                "whitened weight energy. Consider increasing rank.",
                RuntimeWarning,
                stacklevel=2,
            )

    sqrt_S_r = S_r.clamp_min(0.0).sqrt()

    L_factor = U_r * sqrt_S_r.unsqueeze(0)
    # R = sqrt(S_r) * Vh_r @ S^{-1/2} = sqrt(S_r) * (Vh_r @ evecs * inv_sqrt_evals) @ evecs^T
    R_factor = sqrt_S_r.unsqueeze(1) * ((Vh_r @ evecs) * inv_sqrt_evals.unsqueeze(0)) @ evecs.T

    return L_factor.to(weight.dtype).contiguous(), R_factor.to(weight.dtype).contiguous()


__all__ = ["low_rank_aa_svd"]
