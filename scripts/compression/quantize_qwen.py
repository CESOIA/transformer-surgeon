#!/usr/bin/env python3
"""
HF-native generation quality test for a8w8 quantization of Qwen2-0.5B-Instruct.

Tests whether transformer-surgeon's calibrated MLP activation scales are
sufficient for coherent text generation when used as fake-quantization inside
HF's framework (no ExecuTorch export involved).

Calibration → apply path:
  1. WikiText-2 calibration: ActivationRangeSummary tracks running min/max of
     INPUT activations for each MLP linear (gate_proj, up_proj, down_proj).
  2. manager.apply(hard=False) converts min/max → scale/zp and installs a
     forward pre-hook on each linear that fake-quantizes the input tensor:
       x_fq = (round(x / scale) + zp).clamp(0, 255) - zp) * scale
     This emulates per-tensor asymmetric INT8 quantization on the input side.
  3. Weights are soft-quantized (dequantize(quantize(w)) in float).
  4. HF's model.generate() runs the modified model.

Interpretation of results:
  - Coherent output → surgeon input activation scales are correct; the a8w8
    ExecuTorch quality issue comes from PT2E's OUTPUT activation observers
    (gate_proj/up_proj/down_proj outputs), which are calibrated separately by
    _calibrate_pt2e_observers and can have degenerate scales (zp=127).
  - Garbled output → calibration data itself is insufficient for per-tensor
    asymmetric INT8 activation quantization of Qwen2-0.5B.

Usage:
    cd transformer-surgeon
    python scripts/compression/quantize_qwen.py
    python scripts/compression/quantize_qwen.py --num-cal-samples 4 --prompt "What is the capital of Japan?"
"""

import argparse
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager


MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a8w8 quantized Qwen2 generation quality inside HF"
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--num-cal-samples",
        type=int,
        default=4,
        help="Number of WikiText-2 token chunks for activation calibration",
    )
    parser.add_argument("--cal-seq-len", type=int, default=512,
                        help="Token length of each calibration chunk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Tell me one short fact about France.")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--skip-quant",
        action="store_true",
        help="Run unquantized baseline for comparison",
    )
    return parser.parse_args()


def build_wikitext_loader(tokenizer, num_samples: int, seq_len: int, seed: int) -> DataLoader:
    raw = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        split="train",
    )
    texts = [t for t in raw["page"] if isinstance(t, str) and len(t) > 0]
    corpus = "\n\n".join(texts)
    token_ids = tokenizer(corpus, truncation=False, padding=False,
                          return_attention_mask=False)["input_ids"]
    actual = min(num_samples, len(token_ids) // seq_len)
    rng = random.Random(seed)
    examples = []
    for _ in range(actual):
        start = rng.randint(0, len(token_ids) - seq_len - 1)
        examples.append({
            "input_ids": torch.tensor(token_ids[start:start + seq_len], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })
    return DataLoader(examples, batch_size=1, shuffle=False)


def apply_a8w8(model, tokenizer, num_cal_samples: int, cal_seq_len: int, seed: int) -> None:
    """Calibrate and apply MLP a8w8 fake-quantization in-place."""
    loader = build_wikitext_loader(tokenizer, num_cal_samples, cal_seq_len, seed)

    manager = Qwen2CompressionSchemesManager(model)
    criteria = ["mlp"]

    manager.set("quantization", "precision", 8, criteria=criteria)
    manager.set("quantization", "method", "vanilla", criteria=criteria)
    manager.set("quantization", "granularity", "per_channel", criteria=criteria)
    manager.set("quantization", "sparsity", 0.0, criteria=criteria)
    manager.set("quantization", "sparse_method", "magnitude", criteria=criteria)
    manager.set("quantization", "eps", 1e-6, criteria=criteria)
    # Activation: per-tensor asymmetric INT8 (matches ExecuTorch XNNPACK config)
    manager.set("quantization", "precision_activation", 8, criteria=criteria)
    # method_activation defaults to "maxmin"; scheme_activation defaults to "asymmetric"

    manager.set_calibration_data(loader)

    # hard=False: weight fake-quant (dequantize(quantize(w))) + activation pre-hook
    # This keeps the model fully in float so HF generate() works unmodified.
    manager.apply(hard=False, criteria=criteria, verbose=False)


def print_act_scales(model, n: int = 9) -> None:
    """Print the first n calibrated activation scales for sanity-check."""
    shown = 0
    for name, module in model.named_modules():
        if not hasattr(module, "_act_quant_scale"):
            continue
        scale = module._act_quant_scale.item()
        zp = module._act_quant_zero_point.item()
        act_min = (0 - zp) * scale
        act_max = (255 - zp) * scale
        print(f"  {name:55s}  scale={scale:.5f}  zp={zp:4.0f}  "
              f"range=[{act_min:.2f}, {act_max:.2f}]")
        shown += 1
        if shown >= n:
            print(f"  ... ({sum(1 for _, m in model.named_modules() if hasattr(m, '_act_quant_scale')) - n} more)")
            break


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    full_prompt = CHAT_TEMPLATE.format(instruction=prompt)
    input_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True), len(new_ids)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading {args.model_name} ...")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name, torch_dtype=torch.float32
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not args.skip_quant:
        print(f"\nCalibrating a8w8 on {args.num_cal_samples} WikiText-2 samples "
              f"(seq_len={args.cal_seq_len}) ...")
        apply_a8w8(model, tokenizer, args.num_cal_samples, args.cal_seq_len, args.seed)

        print("\nCalibrated input activation scales (first 9 MLP layers):")
        print_act_scales(model, n=9)
    else:
        print("\nSkipping quantization — running float32 baseline.")

    print(f"\nGenerating (max_new_tokens={args.max_new_tokens}):")
    print(f"  prompt: '{args.prompt}'")

    text, n_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens)

    print(f"\nGenerated ({n_tokens} tokens):")
    print(f"  '{text}'")


if __name__ == "__main__":
    main()
