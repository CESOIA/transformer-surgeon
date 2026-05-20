import torch
from typing import Tuple


def low_rank_svd(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    weight_f32 = weight.float()
    U, S, Vh = torch.linalg.svd(weight_f32, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    US_r_out = (U_r * S_r.unsqueeze(0)).to(weight.dtype)
    V_r_out = Vh_r.to(weight.dtype)
    return US_r_out.contiguous(), V_r_out.contiguous()

__all__ = ["low_rank_svd"]