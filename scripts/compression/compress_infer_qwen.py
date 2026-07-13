#!/usr/bin/env python3
"""
Compress Qwen2-0.5B-Instruct (LRD + structured pruning + quantization) and run
HF generate() directly on the compressed model — no export/backend involved.

Everything happens in plain HF/torch:
  1. Load Qwen2-0.5B-Instruct.
  2. Print the model structure (`print(model)`) so block/layer names are handy
     when writing `criteria=` strings below.
  3. Build a WikiText-2 calibration DataLoader (used by calibration-dependent
     methods, e.g. `lrd.method="svd-llm-v2"` or `structured_pruning.method in
     {"magnitude", "gradient"}`; ignored by calibration-free methods like
     `lrd.method="svd"` or `structured_pruning.method="random"`).
  4. `configure_compression()` — a single block where you dial in LRD / pruning
     / quantization settings per layer or block. Edit this to experiment.
  5. `manager.apply(hard=...)` compresses in place; `model.generate()` runs as
     normal HF inference.

Usage:
    cd transformer-surgeon
    python scripts/compression/compress_infer_qwen.py
    python scripts/compression/compress_infer_qwen.py --hard --prompt "What is the capital of Japan?"
    python scripts/compression/compress_infer_qwen.py --skip-compress   # float baseline
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
        description="Compress Qwen2-0.5B and run HF generate() (no export)"
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-cal-samples", type=int, default=4,
        help="Number of WikiText-2 token chunks for calibration",
    )
    parser.add_argument("--cal-seq-len", type=int, default=512,
                         help="Token length of each calibration chunk")
    parser.add_argument(
        "--hard", action="store_true",
        help="Hard-apply (irreversible, resizes weights). Default is soft "
             "(reversible, same-shape fake-compression) — safer for mixing "
             "LRD/pruning/quantization on the same layers.",
    )
    parser.add_argument("--prompt", default="Tell me one short fact about France.")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--skip-compress", action="store_true",
        help="Run the unmodified float baseline for comparison",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Calibration data
# --------------------------------------------------------------------------

def build_wikitext_loader(tokenizer, num_samples: int, seq_len: int, seed: int) -> DataLoader:
    """Random fixed-length token chunks from WikiText-2, wrapped as a DataLoader.

    Each example carries `labels = input_ids.clone()` so calibration methods
    that need a loss (e.g. gradient-based pruning) can use the model's own
    causal-LM loss via `manager.set_calibration_loss(lambda out, _: out.loss)`.
    """
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
        chunk = torch.tensor(token_ids[start:start + seq_len], dtype=torch.long)
        examples.append({
            "input_ids": chunk,
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "labels": chunk.clone(),
        })
    return DataLoader(examples, batch_size=1, shuffle=False)


# --------------------------------------------------------------------------
# Model inspection
# --------------------------------------------------------------------------

def print_model_structure(model) -> None:
    """Dump the module tree so block/layer names are easy to copy into
    `criteria=` below (e.g. "mlp.gate_proj", "self_attn.q_proj", block index 3)."""
    print("=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    print(model)
    print(f"\nnum_hidden_layers = {model.config.num_hidden_layers}")
    print(f"hidden_size       = {model.config.hidden_size}")
    print(f"intermediate_size = {model.config.intermediate_size}")
    print(f"num_attention_heads / num_key_value_heads = "
          f"{model.config.num_attention_heads} / {model.config.num_key_value_heads}")
    print("=" * 70)


# --------------------------------------------------------------------------
# ============================================================
# COMPRESSION SETTINGS — edit this block to try different setups
# ============================================================
#
# manager.set(<compression_type>, <param>, <value>, criteria=<criteria>)
#
# compression_type: "lrd" | "structured_pruning" | "quantization"
# criteria:
#   None / "all"        -> every linear layer
#   int                  -> all layers in that block index, e.g. criteria=3
#   str                  -> layers whose path contains the substring,
#                            e.g. "mlp", "self_attn", "down_proj", "q_proj"
#   [[str, int, ...]]    -> AND within the inner list, e.g. [["mlp", 0]]
#   [str, int, ...]      -> OR across items, e.g. ["q_proj", 3]
#
# Layer names inside each block (see printed model structure above):
#   self_attn.q_proj / k_proj / v_proj / o_proj
#   mlp.gate_proj / up_proj / down_proj
#
# Full parameter reference: transformer-surgeon/AGENTS.md
# --------------------------------------------------------------------------

def configure_compression(manager: Qwen2CompressionSchemesManager) -> None:
    # --- 1) Low-Rank Decomposition (LRD) ------------------------------------
    # rank="full" disables LRD. "svd" needs no calibration; "svd-llm-v2" needs
    # the "covariance" calibration summary (handled automatically once you
    # call manager.set_calibration_data(...) below).
    manager.set("lrd", "rank", "full", criteria=None)
    manager.set("lrd", "method", "svd", criteria=None)
    # Example: low-rank the down_proj of every block.
    # manager.set("lrd", "rank", 128, criteria="mlp.down_proj")

    # --- 2) Structured pruning (removes output neurons) ---------------------
    # Coupled masks so gate_proj/up_proj within a block prune the SAME
    # neurons (required for the down_proj cascade to be well-defined).
    groups = manager.auto_groups()
    for g in groups:
        print(f"Auto-group {g}: {groups[g]}")
    for g in groups:
        manager.set("structured_pruning", "share_mask", True, group=g)
        manager.set("structured_pruning", "reduce_op", "add", group=g)

    # Only MLP pruning is wired end-to-end for hard=True (attention GQA
    # head_dim resizing is not implemented). "magnitude"/"gradient" need
    # calibration; "random" doesn't.
    manager.set("structured_pruning", "method", "magnitude", group="group49")
    # manager.set("structured_pruning", "granularity", 128, group="group49")
    # manager.set("structured_pruning", "repeated_pattern", True, group="group49")
    manager.set("structured_pruning", "ratio", 0.1, group="group49")
    # Example: prune 12.5% of MLP intermediate neurons in every block.
    # manager.set("structured_pruning", "ratio", 0.125, criteria="mlp")
    # Example: prune only block 0's MLP harder.
    # manager.set("structured_pruning", "ratio", 0.25, criteria=[["mlp", 0]])

    # --- 3) Quantization ------------------------------------------------------
    # precision="full" disables quantization. precision is an int bit-width
    # (8, 4, 2, ...), NOT a string like "int8".
    manager.set("quantization", "precision", "full", criteria=None)
    manager.set("quantization", "granularity", "per_channel", criteria=None)
    # Example: int8 weight-only quantization on attention projections.
    # manager.set("quantization", "precision", 8, criteria="self_attn")
    # Example: int8 weight + int8 activation (needs calibration) on MLP.
    # manager.set("quantization", "precision", 8, criteria="mlp")
    # manager.set("quantization", "precision_activation", 8, criteria="mlp")


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, max_new_tokens: int):
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

    print_model_structure(model)

    if not args.skip_compress:
        print(f"\nBuilding WikiText-2 calibration set "
              f"({args.num_cal_samples} chunks x {args.cal_seq_len} tokens) ...")
        loader = build_wikitext_loader(tokenizer, args.num_cal_samples, args.cal_seq_len, args.seed)

        manager = Qwen2CompressionSchemesManager(model)
        manager.set_calibration_data(loader)
        # Needed only if configure_compression() enables "gradient" pruning.
        manager.set_calibration_loss(lambda model_output, target: model_output.loss)

        print("\nApplying compression settings from configure_compression() ...")
        configure_compression(manager)
        manager.apply(hard=args.hard, criteria=None, verbose=True)
        print(f"\nCompression applied ({'hard' if args.hard else 'soft'}).")
    else:
        print("\nSkipping compression — running float baseline.")

    print(f"\nGenerating (max_new_tokens={args.max_new_tokens}):")
    print(f"  prompt: '{args.prompt}'")

    text, n_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"\nGenerated ({n_tokens} tokens):")
    print(f"  '{text}'")


if __name__ == "__main__":
    main()
