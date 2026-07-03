"""
Basic low-rank decomposition (LRD) on a causal LM.

Loads Qwen2-0.5B-Instruct, applies reversible LRD to two specific projection
layers using the HuggingFace manager, runs a generation, then restores the
model to its original weights.

Usage:
    python examples/01_basic_lrd.py
    python examples/01_basic_lrd.py --model-name Qwen/Qwen2-1.5B-Instruct --lrd-rank 320

Requirements:
    pip install transformer-surgeon transformers torch
"""

import argparse

import torch
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress


def parse_args():
    parser = argparse.ArgumentParser(description="Basic LRD example")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--prompt", type=str, default="Write one short sentence about model compression.")
    parser.add_argument("--lrd-rank", type=int, default=640,
                        help="Target rank for low-rank decomposition")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model_name}")

    # --- Load model and tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name, torch_dtype="auto"
    ).to(device).eval()

    params_before = param_count(model)
    print(f"\nParameters before compression: {params_before / 1e6:.2f} M")

    # --- Configure and apply LRD ---
    # Criteria select layers by block index + layer name (AND logic inside a list).
    # See docs/concepts.md or README for the full criteria mini-language.
    manager = Qwen2CompressionSchemesManager(model)
    manager.set("lrd", "rank", args.lrd_rank, criteria=[[0, "self_attn.q_proj"]])
    manager.set("lrd", "rank", args.lrd_rank, criteria=[[1, "mlp.up_proj"]])
    manager.apply(hard=False)  # hard=False keeps compression reversible

    params_after = param_count(model)
    print(f"Parameters after  compression: {params_after / 1e6:.2f} M "
          f"({100 * (1 - params_after / params_before):.1f}% reduction)")

    # --- Generate with compressed model ---
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"\nPrompt:    {args.prompt}")
    print(f"Generated: {generated}")

    # --- Restore original weights ---
    manager.restore()
    params_restored = param_count(model)
    print(f"\nParameters after restore: {params_restored / 1e6:.2f} M "
          f"({'OK' if params_restored == params_before else 'MISMATCH'})")


if __name__ == "__main__":
    main()
