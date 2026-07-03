"""
Calibration-aware LRD using SVD-LLM-v2 on a causal LM.

Loads Qwen2-0.5B, builds a small WikiText-2 calibration DataLoader, and
applies SVD-LLM-v2 compression (which uses activation covariance statistics
to find a better low-rank factorization than plain SVD). Confirms the
compressed model still produces finite logits.

Usage:
    python examples/02_calibrated_lrd_svd_llm_v2.py
    python examples/02_calibrated_lrd_svd_llm_v2.py --lrd-rank 128 --num-calibration-samples 4

Requirements:
    pip install transformer-surgeon transformers torch datasets
"""

import argparse
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress


RANDOM_SEED = 42

MLP_RANK_OVERRIDES = {
    "mlp.up_proj": 497,
    "mlp.gate_proj": 497,
    "mlp.down_proj": 497,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrated LRD (SVD-LLM-v2) example")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--lrd-rank", type=int, default=64,
                        help="Target rank for attention projection layers")
    parser.add_argument("--num-calibration-samples", type=int, default=8,
                        help="Number of WikiText-2 sequences used for calibration")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Token sequence length per calibration sample")
    return parser.parse_args()


def _build_wikitext_calibration_loader(tokenizer, num_samples, seq_len):
    """Return a DataLoader of fixed-length token chunks from WikiText-2."""
    raw = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        split="train",
    )
    texts = [t for t in raw["page"] if isinstance(t, str) and len(t) > 0]
    corpus = "\n\n".join(texts)

    token_ids = tokenizer(
        corpus, truncation=False, padding=False, return_attention_mask=False
    )["input_ids"]

    max_samples = len(token_ids) // seq_len
    actual = min(num_samples, max_samples)

    rng = random.Random(RANDOM_SEED)
    examples = []
    for _ in range(actual):
        start = rng.randint(0, len(token_ids) - seq_len - 1)
        examples.append({
            "input_ids": torch.tensor(token_ids[start : start + seq_len], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })

    return DataLoader(examples, batch_size=1, shuffle=False)


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

    # --- Build calibration data ---
    print(f"Building WikiText-2 calibration loader "
          f"({args.num_calibration_samples} samples × {args.seq_len} tokens)...")
    calibration_loader = _build_wikitext_calibration_loader(
        tokenizer, args.num_calibration_samples, args.seq_len
    )

    # --- Configure SVD-LLM-v2 ---
    # Set method="svd-llm-v2" on all layers first, then override ranks per layer type.
    manager = Qwen2CompressionSchemesManager(model)
    manager.set("lrd", "method", "svd-llm-v2", criteria="all")

    # Attention layers get the user-specified rank.
    for scheme in manager.iter_filtered(criteria="all"):
        if hasattr(scheme.get_module(), "weight"):
            manager.set("lrd", "rank", args.lrd_rank, criteria=scheme.path)

    # MLP layers use dedicated rank values (larger to preserve more capacity).
    for layer_key, rank in MLP_RANK_OVERRIDES.items():
        manager.set("lrd", "rank", rank, criteria=layer_key)

    # --- Attach calibration data and apply ---
    # The manager will run a calibration pass (collecting activation covariance)
    # before applying SVD-LLM-v2 to each layer.
    manager.set_calibration_data(calibration_loader)
    print("Running calibration + compression (this may take a few minutes)...")
    manager.apply(hard=False, criteria="all", device=device, offload_to_cpu=True)

    params_after = param_count(model)
    print(f"\nParameters after  compression: {params_after / 1e6:.2f} M "
          f"({100 * (1 - params_after / params_before):.1f}% reduction)")

    # --- Verify the compressed model still works ---
    inputs = tokenizer("The quick brown fox", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)

    logits_ok = torch.isfinite(out.logits).all().item()
    print(f"\nForward pass logits finite: {logits_ok}")
    if not logits_ok:
        raise RuntimeError("Compressed model produced NaN or Inf logits.")
    print("Done.")


if __name__ == "__main__":
    main()
