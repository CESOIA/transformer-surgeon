"""
Demonstrates manual int8 symmetric weight quantization with torchao.

Quantization scheme (naive):
  - Symmetric  : range is [-max_abs, +max_abs], zero_point = 0
  - Per-tensor : one single scale factor for the whole weight matrix
  - scale      = max(abs(W)) / 127
  - q          = round(W / scale).clamp(-128, 127).to(int8)
  - W_approx   = q * scale   (dequantize)
"""

import torch
import torch.nn as nn
from torchao.quantization import Int8Tensor, PerTensor, quantize_, Int8WeightOnlyConfig


# ---------------------------------------------------------------------------
# 1.  Tiny model
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4, bias=False)
        self.fc2 = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# 2.  Naive symmetric per-tensor int8 quantization
# ---------------------------------------------------------------------------

def naive_quantize(weight: torch.Tensor):
    """
    Returns qdata (int8, same shape) and scale (float32, shape [1, 1]).
    scale = max(|W|) / 127  so the full range maps to [-127, 127].
    """
    max_abs = weight.abs().max()
    scale   = max_abs / 127.0
    qdata   = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    # torchao expects scale shaped [1, 1] for PerTensor on a 2-D weight
    return qdata, scale.to(torch.float32).reshape(1, 1)


def dequantize(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return qdata.to(torch.float32) * scale


# ---------------------------------------------------------------------------
# 3.  Quantize with torchao  (version=2 returns Int8Tensor)
# ---------------------------------------------------------------------------

torch.manual_seed(0)
model_ao = TinyMLP()

print("=== Original weights ===")
for name, m in model_ao.named_modules():
    if isinstance(m, nn.Linear):
        print(f"{name}.weight:\n{m.weight.data}\n")

quantize_(model_ao, Int8WeightOnlyConfig(granularity=PerTensor(), version=2))

print("=== After torchao quantize_ ===")
for name, m in model_ao.named_modules():
    if isinstance(m, nn.Linear):
        w = m.weight
        print(f"{name}.weight  type : {type(w).__name__}")
        print(f"  block_size : {w.block_size}")
        print(f"  scale      : {w.scale.item():.6f}  shape={w.scale.shape}")
        print(f"  zero_point : {w.zero_point}")
        print(f"  qdata      :\n{w.qdata}\n")

# ---------------------------------------------------------------------------
# 4.  Build the same Int8Tensors manually and inject them
# ---------------------------------------------------------------------------

torch.manual_seed(0)
model_manual = TinyMLP()   # identical fp32 weights

print("=== Manual quantization ===")
for name, module in model_manual.named_modules():
    if not isinstance(module, nn.Linear):
        continue

    fp32_w = module.weight.data.clone()
    qdata, scale = naive_quantize(fp32_w)

    print(f"{name}.weight  shape={fp32_w.shape}")
    print(f"  scale  = {scale.item():.6f}")
    print(f"  qdata  =\n{qdata}")
    print(f"  deq    =\n{dequantize(qdata, scale)}\n")

    # block_size = full weight shape → one block covers the whole tensor (PerTensor)
    block_size  = list(fp32_w.shape)
    zero_point  = torch.zeros(1, 1, dtype=torch.int8)

    int8_weight = Int8Tensor(
        qdata      = qdata,
        scale      = scale,
        block_size = block_size,
        dtype      = fp32_w.dtype,
        zero_point = zero_point,
    )

    module.weight = nn.Parameter(int8_weight, requires_grad=False)

# ---------------------------------------------------------------------------
# 5.  Verify: both models should produce identical output
# ---------------------------------------------------------------------------

x = torch.randn(3, 8)
with torch.no_grad():
    out_ao     = model_ao(x)
    out_manual = model_manual(x)

print("=== Inference comparison ===")
print("torchao output:\n", out_ao)
print("manual  output:\n", out_manual)
print("max absolute diff:", (out_ao - out_manual).abs().max().item())
