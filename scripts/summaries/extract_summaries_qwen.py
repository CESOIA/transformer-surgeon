#!/usr/bin/env python3
"""
Load Qwen2-0.5B-Instruct and pull calibration summaries out of it with
`manager.extract_summaries(...)` — no compression applied, just inspection.

Everything happens in plain HF/torch:
  1. Load Qwen2-0.5B-Instruct.
  2. Print the model structure (`print(model)`) so block/layer names are handy
     when writing `criteria=` below.
  3. Build a tiny WikiText-2 calibration DataLoader (reused from
     scripts/compression/compress_infer_qwen.py) — `extract_summaries()` always
     runs at least one calibration forward pass, even for summaries that don't
     depend on activations (e.g. "weight"/"bias"), since it routes every
     summary through the same generic calibration pipeline.
  4. `collect_summaries()` — a single block where you dial in which layers
     (`criteria=`) and which summaries (`summaries=`) to extract. Edit this to
     experiment.
  5. Save the resulting nested dict to disk with `torch.save` (same format
     `manager.save_state()` uses elsewhere in this repo) so it can be reloaded
     with `torch.load(path, weights_only=False)` without rerunning calibration.

Usage:
    cd transformer-surgeon
    python scripts/summaries/extract_summaries_qwen.py
    python scripts/summaries/extract_summaries_qwen.py --num-cal-samples 8 --cal-seq-len 256
    python scripts/summaries/extract_summaries_qwen.py --output-path scripts/summaries/output/my_summaries.pt
"""

import argparse
import random
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager


MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_OUTPUT_PATH = "./output/qwen_summaries.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract calibration summaries from Qwen2-0.5B (no compression)"
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
        "--output-path", default=DEFAULT_OUTPUT_PATH,
        help="Where to torch.save() the extracted summaries dict. "
             "Pass an empty string to skip saving.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Calibration data
# --------------------------------------------------------------------------

def build_wikitext_loader(tokenizer, num_samples: int, seq_len: int, seed: int) -> DataLoader:
    """Random fixed-length token chunks from WikiText-2, wrapped as a DataLoader."""
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
    print("=" * 70)


# --------------------------------------------------------------------------
# ============================================================
# SUMMARY COLLECTION SETTINGS — edit this block to try different setups
# ============================================================
#
# manager.extract_summaries(criteria=<criteria>, summaries=<summaries>)
#   -> {layer_path: {summary_name: value}}
#
# criteria (same filter language as manager.set()/manager.apply()):
#   None / "all"        -> every linear layer
#   int                  -> all layers in that block index, e.g. criteria=3
#   str                  -> layers whose path contains the substring,
#                            e.g. "mlp", "self_attn", "down_proj", "q_proj"
#   [[str, int, ...]]    -> AND within the inner list, e.g. [["mlp.gate_proj", 0]]
#   [str, int, ...]      -> OR across items, e.g. ["q_proj", 3]
#
# summaries: any of SUPPORTED_SUMMARIES, e.g.
#   "weight", "bias"                       -> raw module parameters
#   "input_activation", "output_activation" -> concatenated raw activations
#   "activation_range", "output_activation_range" -> running min/max
#   "covariance", "shifted_covariance", "cross_covariance"
#   "weight_grad"                          -> needs manager.set_calibration_loss(...) first
#
# Layer names inside each block (see printed model structure above):
#   self_attn.q_proj / k_proj / v_proj / o_proj
#   mlp.gate_proj / up_proj / down_proj   (gate_proj is the "first" MLP layer)
#
# Full parameter reference: transformer-surgeon/AGENTS.md
# --------------------------------------------------------------------------

def collect_summaries(manager: Qwen2CompressionSchemesManager) -> dict:
    # Block 0's first MLP layer (mlp.gate_proj): weight, bias, and raw
    # input/output activations collected over the calibration set.
    criteria = [["mlp.gate_proj", 0], ["mlp.up_proj", 0], ["mlp.down_proj", 0]]
    requested = ["input_activation", "output_activation", "weight", "bias"]

    # Qwen2's MLP projections (gate_proj/up_proj/down_proj) have no bias --
    # only self_attn.q_proj/k_proj/v_proj do -- so drop "bias" here rather than
    # let extract_summaries raise. Change `criteria` above to a self_attn layer
    # to see "bias" populated instead.
    no_bias_layers = [
        scheme.path for scheme in manager.iter_filtered(criteria=criteria)
        if scheme.get_compression_module().bias is None
    ]
    if no_bias_layers and "bias" in requested:
        print(f"Note: dropping 'bias' -- no bias parameter on: {no_bias_layers}")
        requested = [name for name in requested if name != "bias"]

    return manager.extract_summaries(
        criteria=criteria,
        summaries=requested,
        show_progress=True,
    )

    # --- Try instead: covariance over every MLP down_proj ------------------
    # return manager.extract_summaries(
    #     criteria="mlp.down_proj",
    #     summaries=["covariance"],
    # )

    # --- Try instead: activation ranges for every self-attention layer ------
    # return manager.extract_summaries(
    #     criteria="self_attn",
    #     summaries=["activation_range", "output_activation_range"],
    # )


def print_summaries(summaries: dict) -> None:
    print("=" * 70)
    print("EXTRACTED SUMMARIES")
    print("=" * 70)
    for layer_path, layer_summaries in summaries.items():
        print(f"\n{layer_path}")
        for name, value in layer_summaries.items():
            if isinstance(value, torch.Tensor):
                print(f"  {name}: shape={tuple(value.shape)} dtype={value.dtype}")
            elif isinstance(value, dict):
                # e.g. activation_range -> {"min": tensor, "max": tensor}
                shown = {k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v)
                         for k, v in value.items()}
                print(f"  {name}: {shown}")
            else:
                print(f"  {name}: {value}")
    print("=" * 70)


def save_summaries(summaries: dict, output_path: str) -> None:
    """Persist the extracted summaries dict with torch.save.

    torch.save is the natural format here (same one manager.save_state() uses
    for calibration data elsewhere in this repo): the dict is nested
    ({layer_path: {summary_name: value}}), values are torch.Tensors of varying
    shape/dtype and occasional non-tensor structures (e.g. activation_range's
    {"min": tensor, "max": tensor}) -- all of which torch.save/torch.load
    round-trip natively. Reload with:

        summaries = torch.load(output_path, weights_only=False)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(summaries, path)
    print(f"\nSaved summaries to {path}")


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

    print(f"\nBuilding WikiText-2 calibration set "
          f"({args.num_cal_samples} chunks x {args.cal_seq_len} tokens) ...")
    loader = build_wikitext_loader(tokenizer, args.num_cal_samples, args.cal_seq_len, args.seed)

    manager = Qwen2CompressionSchemesManager(model)
    manager.set_calibration_data(loader)

    print("\nExtracting summaries from collect_summaries() ...")
    summaries = collect_summaries(manager)
    print_summaries(summaries)

    if args.output_path:
        save_summaries(summaries, args.output_path)


if __name__ == "__main__":
    main()
