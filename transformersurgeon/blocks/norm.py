import torch

class RMSNorm(torch.nn.Module):
    """RMS normalization, optionally preceded by a max-absolute-value rescale.

    ``prescale=True`` (the default) divides ``x`` by its per-row max |x| before
    computing the variance, which keeps ``sum(x**2)`` away from fp16 overflow on
    wide hidden sizes. It is not free: it adds an abs, a full last-axis max
    reduction, a clamp and a div in front of every norm, so a graph that
    ExecuTorch would otherwise pattern-match into a single ``aten.rms_norm``
    becomes six ops. Across a 24-layer decoder that is ~245 extra HTP kernel
    dispatches per decoded token.

    ``prescale=False`` matches Qualcomm's own LLM reference, which normalizes
    with a plain ``torch.nn.RMSNorm``
    (executorch/examples/qualcomm/oss_scripts/llama/model/layernorm.py). Prefer
    it when exporting in float32, or in float16 for a model whose activations
    are known to stay in range; keep the prescale when fp16 overflow is a real
    risk. Note the two are not bit-identical: rescaling changes how ``eps``
    enters the variance.
    """

    def __init__(self, hidden_size, dtype=None, prescale=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.prescale = prescale

    def forward(self, x):
        if self.prescale:
            # Normalize by max absolute value for stability
            max_x = torch.max(x.abs(), dim=-1, keepdim=True).values
            max_x = torch.clamp(max_x, min=1.0)  # Scaling is useless if max_x < 1
            x = x / max_x

        # Evaluate variance
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        # Normalize with variance
        x = x * torch.rsqrt(variance + 1e-5)

        # Multiply element-wise with the learned weights
        x = self.weight * x
        return x

__all__ = [
    "RMSNorm"
]