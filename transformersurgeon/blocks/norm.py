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
    enters the variance (eps weighs max|x|**2 more). ``prescale_threshold=t``
    rescales only rows with max|x| > t, down to max t: fp16-safe for t <= 128
    and equal to the unscaled norm to < 1e-7 for Qwen2.5 (decisions D-78 of
    compression-benchmark); eps stays a constant, so QNN still fuses the norm.

    ``upcast=True`` is the third option, and the Hugging Face / TensorRT
    Edge-LLM formulation: normalize in float32 and cast back, then apply the
    weight. It needs no prescale (float32 cannot overflow here) and is the
    numerically faithful choice for fp16 models on GPUs, where the two casts
    are nearly free; it takes precedence over ``prescale``.
    """

    def __init__(self, hidden_size, dtype=None, prescale=True, upcast=False, eps=1e-5, prescale_threshold=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.prescale = prescale
        self.upcast = upcast
        self.eps = eps
        self.prescale_threshold = prescale_threshold

    def forward(self, x):
        if self.upcast:
            input_dtype = x.dtype
            x = x.to(torch.float32)
            x = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
            return self.weight * x.to(input_dtype)

        if self.prescale:
            # Normalize by max absolute value for stability
            max_x = torch.max(x.abs(), dim=-1, keepdim=True).values
            if self.prescale_threshold is None:
                max_x = torch.clamp(max_x, min=1.0)  # Scaling is useless if max_x < 1
            else:
                # Rescale only rows whose max exceeds the threshold, down to it. Rows below are
                # untouched, so eps enters exactly as unscaled; rows above have a variance
                # >= threshold**2 / hidden_size, which makes the rescaled eps negligible
                # (128 on 1536 wide: < 1e-7 relative) while x*x stays <= threshold**2.
                max_x = torch.clamp(max_x / self.prescale_threshold, min=1.0)
            x = x / max_x

        # Evaluate variance. x * x, not x.pow(2): ExecuTorch's QNN RecomposeRmsNorm
        # fuses [mul(w, mul(x, rsqrt(mean(x*x) + eps)))] into one HTP RmsNorm op, and
        # the QNN exporter rewrites pow(x, 2) to pow.Tensor_Tensor, which it does not
        # match. Decomposed and quantized, x*x of Qwen2.5's ~5e3 activations spans ~2.5e7
        # in one 16-bit scale (all-op 16a8w: KL 0.25 vs 0.066 with the norm float).
        variance = (x * x).mean(dim=-1, keepdim=True)

        # Normalize with variance
        x = x * torch.rsqrt(variance + self.eps)

        # Multiply element-wise with the learned weights
        x = self.weight * x
        return x

__all__ = [
    "RMSNorm"
]