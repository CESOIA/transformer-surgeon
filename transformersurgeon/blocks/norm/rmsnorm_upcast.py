"""Root-mean-square normalisation computed in float32."""

import torch

from .rmsnorm import RMSNorm, rms_normalize


class RMSNormUpcast(RMSNorm):
    """Normalises in float32, casts back to the input type, then applies ``weight``.

    This is the Hugging Face formulation. Float32 cannot overflow here, so no
    rescale is needed; on GPUs the two casts are nearly free.
    """

    def forward(self, x):
        input_dtype = x.dtype
        x = rms_normalize(x.to(torch.float32), self.eps)
        return self.weight * x.to(input_dtype)


__all__ = ["RMSNormUpcast"]
