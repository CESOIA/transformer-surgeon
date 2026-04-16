"""
Single-layer model ExecuTorch export with multiple quantization approaches.

Aim: reverse-engineer the specific quantization algorithms applied by each path.

Selectable options (via argparse):
  --quant-method   : gptq | awq           (fake-quantize before export)
  --backend        : xnnpack | qnn        (ExecuTorch partitioner)
  --precision      : int4 | int8          (bit width)
  --calibration    : with | without       (run calibration data through observers)
  --pipeline       : pt2e | torchao       (quantization pipeline)
  --skip-requant   : skip PT2E re-quantization (preserve custom-quantized weights)

Quantization algorithms applied per combination:
  ┌───────────┬──────────┬──────────────────────────────────────────────────────┐
  │ pipeline  │ precision│ algorithm / notes                                    │
  ├───────────┼──────────┼──────────────────────────────────────────────────────┤
  │ pt2e      │ int8     │ Symmetric per-channel weight, per-tensor activation  │
  │           │          │ dynamic quantization (MinMax observers).             │
  │           │          │ Calibration: recommended (populates activation       │
  │           │          │ observers); without calibration weight observers     │
  │           │          │ still fire but activations use default ranges.       │
  │ pt2e      │ int4     │ Same observer structure but weight range clamped to  │
  │           │          │ [-8, 7] (4-bit symmetric). Calibration: same as int8│
  │ torchao   │ int4     │ Weight-only int4 group-quantization (group_size=64). │
  │           │          │ Uses torchao.quantization.Int4WeightOnlyConfig.      │
  │           │          │ Calibration: NOT used (weight-only, no observers).   │
  │ torchao   │ int8     │ Weight-only int8 quantization.                       │
  │           │          │ Uses torchao.quantization.Int8WeightOnlyConfig.      │
  │           │          │ Calibration: NOT used (weight-only, no observers).   │
  └───────────┴──────────┴──────────────────────────────────────────────────────┘

  GPTQ fake-quant: simulates GPTQ by computing per-column Hessian-based
      quantization order and reconstructing weights to minimise squared error.
  AWQ fake-quant: simulates AWQ by computing per-channel activation-aware
      saliency scales before uniform quantization.
"""

import argparse
import copy
import math
import os
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Single-layer ExecuTorch quantization export test"
    )
    p.add_argument(
        "--quant-method",
        choices=["gptq", "awq"],
        default="gptq",
        help="Fake-quantization method applied before export (default: gptq)",
    )
    p.add_argument(
        "--backend",
        choices=["xnnpack", "qnn"],
        default="xnnpack",
        help="ExecuTorch backend partitioner (default: xnnpack)",
    )
    p.add_argument(
        "--precision",
        choices=["int4", "int8"],
        default="int8",
        help="Quantization bit-width (default: int8)",
    )
    p.add_argument(
        "--calibration",
        choices=["with", "without"],
        default="with",
        help="Run calibration data through observers (default: with). "
             "For torchao pipeline calibration is NOT used (weight-only); "
             "this flag is ignored in that case.",
    )
    p.add_argument(
        "--pipeline",
        choices=["pt2e", "torchao"],
        default="pt2e",
        help="Quantization pipeline (default: pt2e)",
    )
    p.add_argument(
        "--num-calibration",
        type=int,
        default=32,
        help="Number of random calibration samples (default: 32)",
    )
    p.add_argument(
        "--in-features",
        type=int,
        default=32,
        help="Input feature size of the linear layer (default: 64)",
    )
    p.add_argument(
        "--out-features",
        type=int,
        default=16,
        help="Output feature size of the linear layer (default: 32)",
    )
    p.add_argument(
        "--skip-requant",
        action="store_true",
        default=False,
        help="Inject custom quantization scale into PT2E observers instead of "
             "letting PT2E recompute its own. The exported .pte uses integer "
             "kernels with the exact same scale (and for GPTQ: exact same "
             "integers) as the custom fake-quantization.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Simple single-layer model
# ---------------------------------------------------------------------------

class SingleLinearModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Calibration data generation
# ---------------------------------------------------------------------------

def generate_calibration_data(
    num_samples: int, in_features: int, batch_size: int = 1
) -> list[torch.Tensor]:
    """Return a list of random tensors simulating calibration inputs."""
    return [
        torch.randn(batch_size, in_features, dtype=torch.float32)
        for _ in range(num_samples)
    ]


# ---------------------------------------------------------------------------
# Fake-quantization helpers (custom GPTQ / AWQ simulation)
# ---------------------------------------------------------------------------

def _pt2e_qrange(n_bits: int) -> tuple[int, int]:
    """Return (qmin, qmax) matching the XNNPACK PT2E quantizer conventions.
    int8 symmetric: (-127, 127).  int4: (-8, 7).
    The observer denominator is (qmax - qmin) / 2.
    """
    if n_bits == 8:
        return -127, 127   # XNNPACK symmetric int8
    elif n_bits == 4:
        return -8, 7        # XNNPACK int4 (weight_qmin/qmax)
    else:
        qmax = (1 << (n_bits - 1)) - 1
        return -qmax, qmax


def _round_clamp(x: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    return torch.clamp(torch.round(x), qmin, qmax)


def fake_quantize_uniform(
    weight: torch.Tensor, n_bits: int, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple symmetric uniform fake-quantization.
    Returns (dequantized_weight, scale [out_features], int_weight).
    Scale is 1D per-channel when per_channel=True, scalar otherwise.

    Uses the same qmin/qmax and denominator as PT2E observers so that
    scale injection produces exact integer round-trips.
    """
    qmin, qmax = _pt2e_qrange(n_bits)
    denom = (qmax - qmin) / 2.0  # observer denominator: 127 for int8, 7.5 for int4
    if per_channel:
        max_val = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        max_val = weight.abs().max().clamp(min=1e-8)
    scale = max_val / denom  # [out, 1] or scalar
    int_w = _round_clamp(weight / scale, qmin, qmax)
    scale_1d = scale.squeeze() if per_channel else scale  # [out_features]
    return int_w * scale, scale_1d, int_w.to(torch.int8 if n_bits <= 8 else torch.int32)


def fake_quantize_gptq(
    weight: torch.Tensor,
    calibration_data: list[torch.Tensor],
    n_bits: int,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Simplified GPTQ fake-quantization.

    Algorithm (Frantar et al., 2022):
      1. Accumulate H = X^T X from calibration data (Hessian of squared error).
      2. For each row, quantize columns in order of increasing diagonal(H),
         then update remaining columns to compensate for the quantization error
         using the inverse Hessian row.
    This is a *simplified* version that operates column-by-column on the full
    weight matrix (no lazy-batch / block updates) for clarity.
    """
    qmin, qmax = _pt2e_qrange(n_bits)
    W = weight.clone().float()
    out_features, in_features = W.shape

    # Step 1: compute Hessian H = (1/N) * sum X_i^T X_i  (shape [in, in])
    H = torch.zeros(in_features, in_features)
    for x in calibration_data:
        # x shape: [batch, in_features]
        xf = x.reshape(-1, in_features).float()
        H += xf.T @ xf
    H /= len(calibration_data)
    # Damp diagonal for numerical stability
    damp = 0.01 * torch.diag(H).mean()
    H += damp * torch.eye(in_features)

    # Step 2: Cholesky of H^{-1}
    try:
        Hinv = torch.linalg.cholesky(torch.linalg.inv(H))
    except torch.linalg.LinAlgError:
        # Fall back to pseudo-inverse if singular
        Hinv = torch.linalg.cholesky(
            torch.linalg.pinv(H) + 1e-6 * torch.eye(in_features)
        )

    Q = torch.zeros_like(W)
    Q_int = torch.zeros_like(W)

    # Per-channel scale: max(|row|) / denom — one value per output channel.
    # Uses (qmax - qmin) / 2 as denominator to match PT2E observer convention.
    # Computed from the full row before error compensation modifies W.
    denom = (qmax - qmin) / 2.0  # 127.5 for int8, 7.5 for int4
    channel_scale = weight.abs().amax(dim=1).clamp(min=1e-8).float() / denom

    # Process in blocks
    for col_start in range(0, in_features, block_size):
        col_end = min(col_start + block_size, in_features)
        W_block = W[:, col_start:col_end].clone()
        Hinv_block = Hinv[col_start:col_end, col_start:col_end]

        Err = torch.zeros_like(W_block)
        for j in range(col_end - col_start):
            w_col = W_block[:, j]
            d = Hinv_block[j, j].clamp(min=1e-8)

            # Use per-channel scale (same for all columns in this row)
            scale = channel_scale

            int_col = _round_clamp(w_col / scale, qmin, qmax)
            q_col = int_col * scale
            Q[:, col_start + j] = q_col
            Q_int[:, col_start + j] = int_col

            err = (w_col - q_col) / d
            Err[:, j] = err

            # Update remaining columns in block
            if j + 1 < col_end - col_start:
                W_block[:, j + 1:] -= err.unsqueeze(1) * Hinv_block[j, j + 1:].unsqueeze(0)

        # Propagate block error to remaining weights
        if col_end < in_features:
            W[:, col_end:] -= Err @ Hinv[col_start:col_end, col_end:]

    # channel_scale shape: [out_features] — one scale per output channel
    return Q.to(weight.dtype), channel_scale.to(weight.dtype), Q_int.to(torch.int8 if n_bits <= 8 else torch.int32)


def fake_quantize_awq(
    weight: torch.Tensor,
    calibration_data: list[torch.Tensor],
    n_bits: int,
) -> torch.Tensor:
    """
    Simplified AWQ (Activation-Aware Weight Quantization) fake-quantization.

    Algorithm (Lin et al., 2023):
      1. Compute per-channel activation magnitude from calibration data.
      2. Use activation magnitudes as saliency: channels with larger activation
         magnitudes get a *larger* scale (protecting salient weights).
      3. Scale weights → quantize → unscale.
    """
    qmin, qmax = _pt2e_qrange(n_bits)
    W = weight.clone().float()
    in_features = W.shape[1]

    # Step 1: per-input-channel activation magnitude
    act_mag = torch.zeros(in_features)
    for x in calibration_data:
        act_mag += x.reshape(-1, in_features).float().abs().mean(dim=0)
    act_mag /= len(calibration_data)

    # Step 2: compute saliency-based scale  s_j = (act_mag_j / act_mag.mean)^alpha
    alpha = 0.5  # AWQ paper default search range includes 0.5
    s = (act_mag / (act_mag.mean() + 1e-8)).pow(alpha).clamp(min=1e-4)
    # s shape: [in_features]

    # Step 3: scale weights, quantize, unscale
    W_scaled = W * s.unsqueeze(0)  # [out, in] * [1, in]
    W_q, quant_scale, int_w = fake_quantize_uniform(W_scaled, n_bits, per_channel=True)
    W_out = W_q / s.unsqueeze(0)

    # quant_scale: [out_features] — per-channel symmetric scale on the scaled weights
    # s: [in_features] — per-input-channel AWQ saliency scale
    # Dequant: W_out[i,j] = int_w[i,j] * quant_scale[i] / s[j]
    # Return quant_scale (per-channel) and saliency s separately
    return W_out.to(weight.dtype), quant_scale.to(weight.dtype), int_w, s


# ---------------------------------------------------------------------------
# Weight printing utility
# ---------------------------------------------------------------------------

def print_weight_stats(tag: str, w: torch.Tensor):
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    print(f"  shape  : {list(w.shape)}")
    print(f"  dtype  : {w.dtype}")
    print(f"  min    : {w.min().item():.6f}")
    print(f"  max    : {w.max().item():.6f}")
    print(f"  mean   : {w.mean().item():.6f}")
    print(f"  std    : {w.std().item():.6f}")
    print(f"  unique : {w.unique().numel()} values")
    # Print first 4 rows x 8 cols for inspection
    rows = min(4, w.shape[0])
    cols = min(8, w.shape[1]) if w.ndim > 1 else min(8, w.shape[0])
    if w.ndim == 2:
        print(f"  sample [{rows}x{cols}]:")
        for r in range(rows):
            vals = "  ".join(f"{w[r, c].item():+.4f}" for c in range(cols))
            print(f"    [{r}] {vals}")
    else:
        vals = "  ".join(f"{w[i].item():+.4f}" for i in range(cols))
        print(f"  sample: {vals}")


# ---------------------------------------------------------------------------
# ExecuTorch export pipeline
# ---------------------------------------------------------------------------

def get_partitioner(backend: str):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        return XnnpackPartitioner()
    elif backend == "qnn":
        try:
            from executorch.backends.qualcomm.partition.qnn_partitioner import (
                QnnPartitioner,
            )
            from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
                QcomChipset,
            )
            from executorch.backends.qualcomm.utils.utils import (
                generate_qnn_executorch_compiler_spec,
            )
            compiler_spec = generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SM8650,
                is_online_prepare=True,
            )
            return QnnPartitioner(compiler_spec)
        except ImportError:
            print("WARNING: QNN backend not available, falling back to XNNPACK")
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )
            return XnnpackPartitioner()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def export_pt2e(
    model: nn.Module,
    example_input: tuple[torch.Tensor, ...],
    calibration_data: list[torch.Tensor],
    precision: str,
    calibrate: bool,
    backend: str,
    custom_scale: torch.Tensor | None = None,
):
    """
    PT2E pipeline:
      torch.export → prepare_pt2e (insert observers) → calibrate → convert_pt2e
      → re-export → to_edge_transform_and_lower → ExecuTorch program

    If custom_scale is provided (1D tensor, one value per output channel), the
    weight observer's recorded min/max are overridden AFTER calibration but
    BEFORE convert_pt2e, so that the q/dq ops use the custom scale instead of
    the observer-computed one.  This makes PT2E produce the exact same integer
    weights as the custom quantization (GPTQ, QAT, etc.).

    Why it works: the model already holds dequantized weights W[i,j] = int[i,j]
    * scale[i].  When convert_pt2e re-quantizes with the *same* scale:
        int_new[i,j] = round(W[i,j] / scale[i]) = round(int[i,j]) = int[i,j]
    So the exported integers match the custom ones exactly.

    Note for AWQ: AWQ applies a per-input-channel saliency factor that cannot be
    expressed as per-output-channel quantization.  Injecting the AWQ quant_scale
    will be close but not bit-exact — a small residual error remains.

    Quantization algorithm (standard, when custom_scale is None):
      - XNNPACK PT2E int8: symmetric per-channel weight (MinMaxObserver),
        per-tensor dynamic activation quantization.
      - XNNPACK PT2E int4: same observers but weight range clamped to [-8,7].
    """
    from torch.export import export as torch_export
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    model.eval()

    n_bits = 4 if precision == "int4" else 8
    qmax = (1 << (n_bits - 1)) - 1

    # Configure quantizer
    if precision == "int8":
        qconfig = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
    else:  # int4
        qconfig = get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True,
            weight_qmin=-8, weight_qmax=7,
        )
    quantizer = XNNPACKQuantizer().set_global(qconfig)

    # Step 1: export to graph
    exported = torch_export(model, example_input)
    print(f"\n[PT2E] Exported graph:\n{exported}")

    # Step 2: insert observers
    prepared = prepare_pt2e(exported.module(), quantizer)
    print(f"[PT2E] Prepared module (observers inserted)")

    # Step 3: calibration (populates both weight and activation observers)
    if calibrate:
        print(f"[PT2E] Running calibration with {len(calibration_data)} samples...")
        with torch.no_grad():
            for sample in calibration_data:
                prepared(sample)
        print(f"[PT2E] Calibration complete")
    else:
        print(
            "[PT2E] Skipping calibration. NOTE: weight observers still fire on first "
            "forward; activation observers will use default ranges."
        )
        with torch.no_grad():
            prepared(example_input[0])

    # Step 3b: inject custom scale into weight observer (override observer state)
    if custom_scale is not None:
        # Note: torchao has its own PerChannelMinMaxObserver in
        # torchao.quantization.pt2e.observer (different class from
        # torch.ao.quantization.observer). Match by class name to handle both.
        #
        # The observer computes: scale = max(max_val / D, eps).
        # D depends on qmin/qmax; eps is an internal threshold (~2^-12).
        # We probe D with max_val=1.0 (safely above eps), then set
        # corrected_max = custom_scale * D so the observer returns custom_scale.
        injected = False
        for name, mod in prepared.named_modules():
            if type(mod).__name__ == "PerChannelMinMaxObserver" and hasattr(mod, "min_val"):
                obs_scale_before = mod.min_val.abs().max().item() / qmax if mod.min_val.numel() > 0 else 0
                # Step A: probe the observer's denominator D with a safe value
                probe_val = torch.ones_like(mod.min_val)
                mod.min_val.copy_(-probe_val)
                mod.max_val.copy_(probe_val)
                probe_scale, _ = mod.calculate_qparams()
                obs_denom = (1.0 / probe_scale.float())  # D per channel
                # Step B: set corrected max so observer returns custom_scale
                corrected_max = custom_scale.float() * obs_denom
                mod.min_val.copy_(-corrected_max)
                mod.max_val.copy_(corrected_max)
                # Verify the injection
                computed_scale, _ = mod.calculate_qparams()
                max_scale_err = (computed_scale.float() - custom_scale.float()).abs().max().item()
                print(f"[PT2E] Injected custom scale into weight observer '{name}'")
                print(f"  Observer scale before: max={obs_scale_before:.8f}")
                print(f"  Custom scale injected: max={custom_scale.max().item():.8f}")
                print(f"  Observer denominator  : {obs_denom[0].item():.2f}")
                print(f"  Scale verification err: {max_scale_err:.2e}")
                injected = True
        if not injected:
            print("[PT2E] WARNING: no PerChannelMinMaxObserver found — "
                  "custom scale could not be injected")

    # Step 4: convert (replace observers with quantized ops)
    converted = convert_pt2e(prepared)
    print(f"[PT2E] Converted module (fake-quant ops inserted)")

    # Step 5: re-export and lower
    quantized_exported = torch_export(converted, example_input)
    print(f"[PT2E] Re-exported quantized graph:\n{quantized_exported}")

    partitioner = get_partitioner(backend)
    edge = to_edge_transform_and_lower(
        quantized_exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=True),
        partitioner=[partitioner],
    )

    tag = "custom" if custom_scale is not None else precision
    graph_str = edge.exported_program().graph_module.print_readable(print_output=False)
    log_name = f"single_layer_pt2e_{tag}_{backend}.log"
    with open(log_name, "w") as f:
        f.write(graph_str)
    print(f"[PT2E] Graph IR saved to {log_name}")

    et_program = edge.to_executorch()
    return et_program, converted, quantized_exported


def export_torchao(
    model: nn.Module,
    example_input: tuple[torch.Tensor, ...],
    precision: str,
    backend: str,
):
    """
    TorchAO pipeline:
      torchao.quantize_(model, config) → torch.export → to_edge_transform_and_lower

    Quantization algorithm:
      - Int4WeightOnlyConfig (group_size=64): weight-only int4 asymmetric
        group quantization. No calibration needed — weights are quantized
        directly using min/max per group.
      - Int8WeightOnlyConfig: weight-only int8 per-channel symmetric
        quantization. No calibration needed.

    NOTE: torchao weight-only quantization does NOT use calibration data.
          The --calibration flag is ignored for this pipeline.
    """
    from torch.export import export as torch_export
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torchao.quantization import quantize_

    model.eval()

    if precision == "int4":
        from torchao.quantization import Int4WeightOnlyConfig
        config = Int4WeightOnlyConfig(group_size=64)
        print("[TorchAO] Applying Int4WeightOnlyConfig (group_size=64, weight-only)")
        print("[TorchAO] Algorithm: asymmetric int4 group quantization, NO calibration")
    else:  # int8
        from torchao.quantization import Int8WeightOnlyConfig
        config = Int8WeightOnlyConfig()
        print("[TorchAO] Applying Int8WeightOnlyConfig (weight-only)")
        print("[TorchAO] Algorithm: symmetric int8 per-channel quantization, NO calibration")

    quantize_(model, config)
    print("[TorchAO] Model quantized in-place")

    exported = torch_export(model, example_input)
    print(f"[TorchAO] Exported graph:\n{exported}")

    partitioner = get_partitioner(backend)
    edge = to_edge_transform_and_lower(
        exported,
        compile_config=EdgeCompileConfig(_check_ir_validity=True),
        partitioner=[partitioner],
    )

    graph_str = edge.exported_program().graph_module.print_readable(print_output=False)
    log_name = f"single_layer_torchao_{precision}_{backend}.log"
    with open(log_name, "w") as f:
        f.write(graph_str)
    print(f"[TorchAO] Graph IR saved to {log_name}")

    et_program = edge.to_executorch()
    return et_program, model, exported


# ---------------------------------------------------------------------------
# Compute error metrics (no printing — results printed later)
# ---------------------------------------------------------------------------

def _compute_output_errors(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_inputs: list[torch.Tensor],
) -> dict:
    """Compute error metrics between original and quantized model outputs."""
    original_model.eval()
    quantized_model.eval()
    max_abs_errors = []
    mse_errors = []
    with torch.no_grad():
        for x in test_inputs:
            y_orig = original_model(x)
            y_quant = quantized_model(x)
            abs_err = (y_orig - y_quant).abs()
            max_abs_errors.append(abs_err.max().item())
            mse_errors.append(((y_orig - y_quant) ** 2).mean().item())
    return {
        "avg_max_err": sum(max_abs_errors) / len(max_abs_errors),
        "worst_max_err": max(max_abs_errors),
        "avg_mse": sum(mse_errors) / len(mse_errors),
    }


def _compute_executorch_errors(
    original_model: nn.Module,
    pte_path: str,
    test_inputs: list[torch.Tensor],
) -> dict:
    """Load .pte and compute error metrics against original model."""
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    max_abs_errors = []
    mse_errors = []
    original_model.eval()
    with torch.no_grad():
        for x in test_inputs:
            y_orig = original_model(x)
            outputs = method.execute([x])
            y_et = outputs[0]
            if not isinstance(y_et, torch.Tensor):
                y_et = torch.tensor(y_et)
            abs_err = (y_orig - y_et).abs()
            max_abs_errors.append(abs_err.max().item())
            mse_errors.append(((y_orig - y_et) ** 2).mean().item())
    return {
        "avg_max_err": sum(max_abs_errors) / len(max_abs_errors),
        "worst_max_err": max(max_abs_errors),
        "avg_mse": sum(mse_errors) / len(mse_errors),
    }


# ---------------------------------------------------------------------------
# Result printing helpers
# ---------------------------------------------------------------------------

def _print_pt2e_int_weights(int_w, scale, zp, precision: str):
    """Print PT2E exported integer weights, scale, and zero-point."""
    n_bits = 4 if precision == "int4" else 8
    qmin, qmax = _pt2e_qrange(n_bits)

    print(f"\n{'='*60}")
    print(f"  PT2E exported integer weights ({n_bits}-bit)")
    print(f"{'='*60}")

    # Integer weights
    rows = min(4, int_w.shape[0])
    cols = min(8, int_w.shape[1])
    print(f"  Integer weight ({int_w.dtype}, {list(int_w.shape)}):")
    for r in range(rows):
        vals = "  ".join(f"{int_w[r, c].item():+4d}" for c in range(cols))
        print(f"    [{r}] {vals}")
    print(f"  Unique int values : {int_w.unique().numel()}")
    print(f"  Integer range     : [{int_w.min().item()}, {int_w.max().item()}]")
    print(f"  Expected range    : [{qmin}, {qmax}]")

    # Scale
    n_show = min(16, scale.numel())
    sv = ", ".join(f"{scale[i].item():.6f}" for i in range(n_show))
    print(f"  Scale ({scale.dtype}, {list(scale.shape)}): [{sv}{'...' if scale.numel() > n_show else ''}]")

    # Zero-point
    if zp is not None:
        zv = ", ".join(f"{zp[i].item()}" for i in range(min(16, zp.numel())))
        print(f"  Zero-point ({zp.dtype}, {list(zp.shape)}): [{zv}{'...' if zp.numel() > 16 else ''}]")

    print(f"  Dequant formula: W[i,j] = (int_w[i,j] - zp[i]) * scale[i]")


def _print_fakequant_details(
    method: str, n_bits: int, scale: torch.Tensor, int_w: torch.Tensor,
    saliency: torch.Tensor | None = None,
):
    """Print scales and integer weights produced by custom GPTQ/AWQ fake-quantization."""
    qmin, qmax = _pt2e_qrange(n_bits)
    label = method.upper()

    print(f"\n{'='*60}")
    print(f"  {label} fake-quant internals ({n_bits}-bit)")
    print(f"{'='*60}")

    # --- Per-channel scale ---
    # GPTQ: scale[i] = max(|W[i,:]|) / qmax  — one per output channel
    # AWQ:  scale[i] = max(|W_scaled[i,:]|) / qmax  — one per output channel
    print(f"  Per-channel scale shape : {list(scale.shape)}")
    print(f"  Per-channel scale dtype : {scale.dtype}")
    print(f"  Per-channel scale range : [{scale.min().item():.8f}, {scale.max().item():.8f}]")
    n_show = min(16, scale.numel())
    vals = ", ".join(f"{scale[i].item():.6f}" for i in range(n_show))
    print(f"  Per-channel scale       : [{vals}{'...' if scale.numel() > n_show else ''}]")

    # --- AWQ saliency scale ---
    if saliency is not None:
        print(f"\n  AWQ saliency scale shape : {list(saliency.shape)}")
        print(f"  AWQ saliency scale range : [{saliency.min().item():.6f}, {saliency.max().item():.6f}]")
        n_show_s = min(16, saliency.numel())
        sv = ", ".join(f"{saliency[i].item():.6f}" for i in range(n_show_s))
        print(f"  AWQ saliency scale       : [{sv}{'...' if saliency.numel() > n_show_s else ''}]")
        print(f"  Dequant formula: W[i,j] = int_w[i,j] * scale[i] / saliency[j]")
    else:
        print(f"  Dequant formula: W[i,j] = int_w[i,j] * scale[i]")

    # --- Integer weights ---
    print(f"\n  Integer weight shape : {list(int_w.shape)}")
    print(f"  Integer weight dtype : {int_w.dtype}")
    print(f"  Integer range        : [{int_w.min().item()}, {int_w.max().item()}]")
    print(f"  Expected range       : [{qmin}, {qmax}]")
    print(f"  Unique int values    : {int_w.unique().numel()}")

    rows = min(4, int_w.shape[0])
    cols = min(8, int_w.shape[1]) if int_w.ndim > 1 else min(8, int_w.shape[0])
    if int_w.ndim == 2:
        print(f"  Integer sample [{rows}x{cols}]:")
        for r in range(rows):
            iv = "  ".join(f"{int_w[r, c].item():+4d}" for c in range(cols))
            print(f"    [{r}] {iv}")

    # Verify round-trip
    if saliency is not None:
        # AWQ: dequant = int_w * scale / saliency
        reconstructed = int_w.float() * scale.float().unsqueeze(1) / saliency.float().unsqueeze(0)
    else:
        # GPTQ: dequant = int_w * scale
        reconstructed = int_w.float() * scale.float().unsqueeze(1)
    print(f"\n  Round-trip check (dequantized from int + scale):")
    if reconstructed.ndim == 2:
        for r in range(min(2, reconstructed.shape[0])):
            rv = "  ".join(f"{reconstructed[r, c].item():+.4f}" for c in range(cols))
            print(f"    [{r}] {rv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    n_bits = 4 if args.precision == "int4" else 8
    calibrate = args.calibration == "with"

    print("=" * 60)
    print("  Single-Layer ExecuTorch Quantization Export Test")
    print("=" * 60)
    print(f"  quant-method  : {args.quant_method}")
    print(f"  backend       : {args.backend}")
    print(f"  precision     : {args.precision} ({n_bits}-bit)")
    print(f"  calibration   : {args.calibration}")
    print(f"  pipeline      : {args.pipeline}")
    print(f"  skip-requant  : {args.skip_requant}")
    print(f"  in_features   : {args.in_features}")
    print(f"  out_features  : {args.out_features}")
    print(f"  num_calib     : {args.num_calibration}")
    print("=" * 60)

    # --- 1. Create model ---
    model = SingleLinearModel(args.in_features, args.out_features)
    model.eval()

    # Keep a pristine copy for comparison
    original_model = copy.deepcopy(model)
    original_w = model.linear.weight.data.clone()

    # --- 2. Generate calibration & test data ---
    calib_data = generate_calibration_data(args.num_calibration, args.in_features)
    test_data = generate_calibration_data(16, args.in_features)
    example_input = (torch.randn(1, args.in_features, dtype=torch.float32),)

    # --- 3. Apply fake quantization (GPTQ or AWQ) ---
    print(f"\n>>> Applying {args.quant_method.upper()} fake-quantization ({n_bits}-bit)...")
    fq_saliency = None
    with torch.no_grad():
        if args.quant_method == "gptq":
            q_weight, fq_scale, fq_int = fake_quantize_gptq(
                model.linear.weight.data, calib_data, n_bits
            )
        else:  # awq
            q_weight, fq_scale, fq_int, fq_saliency = fake_quantize_awq(
                model.linear.weight.data, calib_data, n_bits
            )
        model.linear.weight.data.copy_(q_weight)
    fakequant_w = model.linear.weight.data.clone()

    # --- 4. Validate fake-quantized model vs original ---
    fq_validation = _compute_output_errors(original_model, model, test_data)

    # --- 5. Export to ExecuTorch ---
    export_model = copy.deepcopy(model)

    requant_tag = "norequant" if args.skip_requant else ("calib" if calibrate else "nocal")
    pte_filename = (
        f"single_layer_{args.quant_method}_{args.pipeline}_{args.precision}"
        f"_{requant_tag}_{args.backend}.pte"
    )

    if args.pipeline == "pt2e":
        custom_scale = fq_scale if args.skip_requant else None
        print(f">>> Exporting via PT2E pipeline...")
        if custom_scale is not None:
            print(f">>> --skip-requant: injecting custom {args.quant_method.upper()} scale into PT2E observers")
        et_program, converted_model, quantized_exported = export_pt2e(
            export_model, example_input, calib_data,
            args.precision, calibrate, args.backend,
            custom_scale=custom_scale,
        )
    else:  # torchao
        if calibrate:
            print(
                "[INFO] torchao pipeline uses weight-only quantization — "
                "calibration data is NOT used regardless of --calibration flag."
            )
        print(f">>> Exporting via TorchAO pipeline...")
        et_program, converted_model, quantized_exported = export_torchao(
            export_model, example_input, args.precision, args.backend,
        )

    # Save .pte
    pte_bytes = et_program.buffer
    with open(pte_filename, "wb") as f:
        f.write(pte_bytes)
    print(f"Saved ExecuTorch program to {pte_filename} ({len(pte_bytes)} bytes)")

    # --- 6. Gather exported weights ---
    exported_w = None
    pt2e_int_w = None
    pt2e_scale = None
    pt2e_zp = None
    if args.pipeline == "pt2e":
        sd = quantized_exported.state_dict
        pt2e_int_w = sd.get("_frozen_param0")
        pt2e_scale = sd.get("_scale_0")
        pt2e_zp = sd.get("_zero_point_0")
        if pt2e_int_w is not None and pt2e_scale is not None:
            zp_f = pt2e_zp.float() if pt2e_zp is not None else torch.zeros(pt2e_scale.shape)
            exported_w = (pt2e_int_w.float() - zp_f.unsqueeze(1)) * pt2e_scale.float().unsqueeze(1)
    else:  # torchao
        if hasattr(converted_model, "linear") and hasattr(converted_model.linear, "weight"):
            exported_w = converted_model.linear.weight.data.float()

    # --- 7. Validate ExecuTorch output vs original ---
    et_validation = None
    try:
        et_validation = _compute_executorch_errors(original_model, pte_filename, test_data)
    except Exception as e:
        print(f"  ExecuTorch runtime validation failed: {e}")

    # ===================================================================
    #  ALL PROCESSING DONE — NOW PRINT RESULTS
    # ===================================================================

    print(f"\n\n{'#'*60}")
    print(f"{'#':>2}  RESULTS")
    print(f"{'#'*60}")

    # --- A. Original weights ---
    print_weight_stats("ORIGINAL weights", original_w)

    # --- B. Fake-quant weights + internals ---
    print_weight_stats(
        f"AFTER {args.quant_method.upper()} fake-quant ({n_bits}-bit)", fakequant_w
    )
    _print_fakequant_details(args.quant_method, n_bits, fq_scale, fq_int, fq_saliency)

    # --- C. Exported weights ---
    if args.pipeline == "pt2e":
        if pt2e_int_w is not None:
            _print_pt2e_int_weights(pt2e_int_w, pt2e_scale, pt2e_zp, args.precision)
        if exported_w is not None:
            print_weight_stats("PT2E exported weight (dequantized)", exported_w)
    else:  # torchao
        if exported_w is not None:
            print_weight_stats(
                f"AFTER {args.pipeline.upper()} export quantization", exported_w
            )
        else:
            print(f"\n[INFO] Weight tensor not directly accessible after "
                  f"{args.pipeline} quantization (packed/fused format)")
            print("  Parameters in converted model:")
            for name, param in converted_model.named_parameters():
                print(f"    {name}: {list(param.shape)} {param.dtype}")
            for name, buf in converted_model.named_buffers():
                print(f"    (buffer) {name}: {list(buf.shape)} {buf.dtype}")

    # --- D. Weight error summary ---
    print(f"\n{'='*60}")
    print(f"  Weight Error Summary")
    print(f"{'='*60}")
    err_fq = (original_w.float() - fakequant_w.float()).abs()
    print(f"  Original vs {args.quant_method.upper()} fake-quant:")
    print(f"    max |err| : {err_fq.max().item():.6f}")
    print(f"    mean |err|: {err_fq.mean().item():.6f}")
    print(f"    MSE       : {(err_fq**2).mean().item():.8f}")
    if exported_w is not None:
        err_ex = (original_w.float() - exported_w.float()).abs()
        err_fq_ex = (fakequant_w.float() - exported_w.float()).abs()
        print(f"  Original vs exported ({args.pipeline.upper()}):")
        print(f"    max |err| : {err_ex.max().item():.6f}")
        print(f"    mean |err|: {err_ex.mean().item():.6f}")
        print(f"    MSE       : {(err_ex**2).mean().item():.8f}")
        print(f"  Fake-quant vs exported ({args.pipeline.upper()}):")
        print(f"    max |err| : {err_fq_ex.max().item():.6f}")
        print(f"    mean |err|: {err_fq_ex.mean().item():.6f}")
        print(f"    MSE       : {(err_fq_ex**2).mean().item():.8f}")
    else:
        print(f"  Original vs exported: (weight not directly accessible)")

    # --- E. Output error summary ---
    print(f"\n{'='*60}")
    print(f"  Output Error Summary")
    print(f"{'='*60}")
    print(f"  {args.quant_method.upper()} fake-quant vs original:")
    print(f"    Avg max |err| : {fq_validation['avg_max_err']:.6f}")
    print(f"    Worst max |err|: {fq_validation['worst_max_err']:.6f}")
    print(f"    Avg MSE       : {fq_validation['avg_mse']:.8f}")
    print(f"    Avg RMSE      : {math.sqrt(fq_validation['avg_mse']):.6f}")
    if et_validation is not None:
        print(f"  ExecuTorch ({pte_filename}) vs original:")
        print(f"    Avg max |err| : {et_validation['avg_max_err']:.6f}")
        print(f"    Worst max |err|: {et_validation['worst_max_err']:.6f}")
        print(f"    Avg MSE       : {et_validation['avg_mse']:.8f}")
        print(f"    Avg RMSE      : {math.sqrt(et_validation['avg_mse']):.6f}")
    else:
        print(f"  ExecuTorch vs original: (runtime not available)")

    print(f"\nDone. Artifacts: {pte_filename}")


if __name__ == "__main__":
    main()
