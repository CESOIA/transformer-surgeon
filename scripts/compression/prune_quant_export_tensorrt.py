#!/usr/bin/env python3
"""
Structured pruning + quantization -> TensorRT export for Qwen2-0.5B.

Demonstrates the full deploy path for a *pruned* model:

  1.  Load Qwen2-0.5B (float16).
  2.  auto_groups(): build the coupled-mask groups from the model's indexing
      (q/k, gate/up per block, plus the cross-block residual group). Enable
      share_mask on the MLP gate/up groups so gate_proj and up_proj prune the
      SAME neurons (coherent grouped pruning).
  3.  Randomly assign a structured-pruning ratio per block's MLP group (random
      method, no calibration) and randomly quantize each linear to int8
      (weight-only, per-channel). No LRD.
  4.  manager.apply(hard=True): removes MLP intermediate neurons, resizes the
      matrices, cascades down_proj's input, and hard-quantizes the chosen layers.
  5.  convert_for_export(): builds the custom decoder graph. The MLP blocks are
      pruning-aware, so the converted layers match the pruned weight shapes.
  6.  export_to_backend(TensorRTExportConfig): compiles to a TensorRT engine and
      compares float vs TensorRT outputs.

Attention (q/k/v/o) is left unpruned: hard-pruning it changes head_dim (Qwen is
GQA), which needs attention-forward/config surgery not yet wired. See prune_qwen.py.

Requires a CUDA device and torch_tensorrt.

Usage:
    cd transformer-surgeon
    python scripts/compression/prune_quant_export_tensorrt.py
    python scripts/compression/prune_quant_export_tensorrt.py --seed 7 --output /tmp/qwen_pruned.pt2
"""

import argparse
import os
import random
import tempfile

import torch
import torch.nn as nn

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.tensorrt import TensorRTExportConfig


MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
PRUNE_RATIOS = [0.0, 0.125, 0.25]


def parse_args():
    p = argparse.ArgumentParser(description="Prune + quantize + TensorRT export for Qwen2")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant-prob", type=float, default=0.5,
                   help="Probability each linear is int8-quantized (weight-only)")
    p.add_argument("--output", default=None, help="Engine output path (.pt2)")
    p.add_argument("--max-seq-len", type=int, default=512)
    return p.parse_args()


def apply_prune_and_quant(model, seed: int, quant_prob: float):
    """Grouped random structured pruning (MLP) + random int8 quant, hard, in-place."""
    rng = random.Random(seed)
    manager = Qwen2CompressionSchemesManager(model)

    # 1. Coupled-mask groups from indexing. Enable share_mask on the MLP gate/up
    #    groups so each block's gate_proj and up_proj prune the same neurons.
    groups = manager.auto_groups()
    mlp_groups = {g: paths for g, paths in groups.items()
                  if any("gate_proj" in p for p in paths)}
    for g in mlp_groups:
        manager.set("structured_pruning", "share_mask", True, group=g)

    # 2. Structured pruning config (random method needs no calibration).
    manager.set("structured_pruning", "method", "random", criteria=None)
    manager.set("structured_pruning", "reduce_op", "add", criteria=None)

    # Random pruning ratio per block's MLP group (coherent: gate/up share it).
    ratios = {}
    for g in mlp_groups:
        ratio = rng.choice(PRUNE_RATIOS)
        ratios[g] = ratio
        manager.set("structured_pruning", "ratio", ratio, group=g)
    pruned = sum(1 for r in ratios.values() if r > 0)
    print(f"  MLP groups: {len(mlp_groups)} ({pruned} pruned) | ratios sample: "
          f"{list(ratios.values())[:6]}")

    # 3. Random int8 weight-only quantization per linear (no LRD).
    quantized = 0
    for scheme in manager:
        module = scheme.get_module()
        if not isinstance(module, nn.Linear):
            continue
        if rng.random() < quant_prob:
            manager.set("quantization", "precision", 8, criteria=scheme.path)
            manager.set("quantization", "granularity", "per_channel", criteria=scheme.path)
            quantized += 1
    print(f"  int8-quantized linears: {quantized}")

    # 4. Hard apply: prune (remove neurons + cascade) then quantize.
    manager.apply(hard=True, criteria=None)
    return manager


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("This script requires a CUDA device for TensorRT export.")

    print(f"Loading {args.model_name} (float16) ...")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).eval()

    print("\nApplying grouped structured pruning + int8 quantization (hard) ...")
    apply_prune_and_quant(model, args.seed, args.quant_prob)

    b0 = model.model.layers[0].mlp
    print(f"  block 0 MLP: gate.out={b0.gate_proj.out_features} "
          f"down.in={b0.down_proj.in_features} (orig intermediate={model.config.intermediate_size})")

    output = args.output or os.path.join(tempfile.mkdtemp(), "qwen2_pruned_quant_trt.pt2")
    print(f"\nExporting to TensorRT -> {output}")
    export_cfg = TensorRTExportConfig(
        output_path=output,
        backend="tensorrt",
        max_seq_len=args.max_seq_len,
        convert_options={"use_sdpa": False},
        run_weight_mismatch_check=False,
        device="cuda:0",
        verbose=True,
    )
    result = export_to_backend(model, config=export_cfg)

    print("\nExport complete.")
    print(f"  backend: {result.backend}")
    print(f"  engine_path: {result.engine_path}")
    print(f"  file exists: {os.path.isfile(result.engine_path)} "
          f"({os.path.getsize(result.engine_path)} bytes)")
    if getattr(result, "inference_stats", None):
        s = result.inference_stats
        print(f"  float-vs-TensorRT: mean_abs_err={s['mean_abs_err']:.4f} "
              f"max_abs_err={s['max_abs_err']:.4f}")


if __name__ == "__main__":
    main()
