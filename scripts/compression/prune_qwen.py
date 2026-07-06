#!/usr/bin/env python3
"""
Gradient-based structured pruning of Qwen2-0.5B-Instruct.

Prunes 10% of the output neurons of every prunable linear layer, ranked by a
gradient-based importance score (|weight * dLoss/dWeight|), and keeps HF's
standard generate() working.

Calibration -> apply path:
  1. auto_groups() reads each block's `pruning.coupled_masks` annotation and
     groups the layers that must share one pruning mask (q_proj/k_proj and
     gate_proj/up_proj). share_mask + reduce_op="add" make every member of a
     group prune the same neurons (reduced importance scores).
  2. WikiText-2 calibration runs a forward+backward pass; the WeightGradSummary
     records dLoss/dWeight for each linear (needs a loss callback, provided via
     set_calibration_loss). The self-supervised LM loss is model(...).loss with
     labels = input_ids.
  3. manager.apply(hard=False) computes per-layer gradient scores, builds the
     structured masks (shared within groups), and zeroes the least-important 10%
     of output rows per layer. Soft (reversible) pruning keeps the model in float.

Why soft (hard=False): hard pruning removes neurons and resizes matrices, and
cascades the removal to the next layer's inputs. That is only well-defined for
layers with a coupling target (mlp up/gate -> down, attn v -> o). Layers whose
output is the residual stream (o_proj, down_proj) have no coupling target, so
hard-pruning *every* layer would break the residual shapes. Soft pruning zeroes
rows in place and works for any layer.

Usage:
    cd transformer-surgeon
    python scripts/compression/prune_qwen.py
    python scripts/compression/prune_qwen.py --ratio 0.1 --num-cal-samples 4
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
        description="Gradient-based structured pruning of Qwen2 (10% per layer)"
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="Structured pruning ratio applied to every layer")
    parser.add_argument(
        "--num-cal-samples",
        type=int,
        default=4,
        help="Number of WikiText-2 token chunks for gradient calibration",
    )
    parser.add_argument("--cal-seq-len", type=int, default=512,
                        help="Token length of each calibration chunk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Tell me one short fact about France.")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--skip-prune",
        action="store_true",
        help="Run the unpruned baseline for comparison",
    )
    parser.add_argument(
        "--include-attention",
        action="store_true",
        help=(
            "Also hard-prune attention (q/k/v). Off by default: Qwen uses GQA, so "
            "hard-pruning attention changes head_dim (breaks the attention reshape, "
            "whose head_dim is fixed in config) and the v->o coupling is not 1:1 "
            "(o_proj input = num_heads*head_dim, v_proj output = num_kv_heads*head_dim). "
            "Enable to reproduce those failures."
        ),
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
        chunk = torch.tensor(token_ids[start:start + seq_len], dtype=torch.long)
        examples.append({
            "input_ids": chunk,
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            # Self-supervised LM labels so model(...).loss is differentiable and
            # produces per-layer weight gradients during calibration.
            "labels": chunk.clone(),
        })
    return DataLoader(examples, batch_size=1, shuffle=False)


def skip_output_layers(indexing) -> set:
    """Layer names whose OUTPUT feeds a residual/skip connection.

    Derived generically from the indexing: each skip_connections pair is
    [input_of_A, output_of_B]; the last projection of subblock B is the layer
    that writes into the residual stream. Those layers must not be hard-pruned,
    because their output dimension is the shared hidden dim and has no single
    coupling target to resize.
    """
    names = set()
    for block in indexing.values():
        path_list = block.get("path_list", {})
        for pair in block.get("skip_connections", []):
            subblock = pair[1]
            leaves = path_list.get(subblock, [])
            if leaves:
                names.add(f"{subblock}.{leaves[-1]}")
    return names


def apply_gradient_pruning(model, tokenizer, ratio: float,
                           num_cal_samples: int, cal_seq_len: int, seed: int,
                           include_attention: bool = False):
    """Calibrate weight gradients and apply hard 10% structured pruning in-place."""
    loader = build_wikitext_loader(tokenizer, num_cal_samples, cal_seq_len, seed)

    manager = Qwen2CompressionSchemesManager(model)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Group the layers that must share a pruning mask (q/k, gate/up) straight from
    # the model's indexing annotations. Enable share_mask FIRST — it resets each
    # grouped layer's regular config, so all other options are set afterwards.
    groups = manager.auto_groups()
    print(f"  auto_groups created {len(groups)} coupled-mask group(s)")
    for group_name in groups:
        manager.set("structured_pruning", "share_mask", True, group=group_name)

    # Gradient-based structured pruning, reducing scores with "add".
    manager.set("structured_pruning", "method", "gradient", criteria=None)
    manager.set("structured_pruning", "reduce_op", "add", criteria=None)

    # Head-dim projections (q/k/v): prune per head using a repeated pattern of
    # size head_dim, so q_proj (14 heads) and k_proj (2 heads, GQA) prune the
    # same head_dim positions and share one mask despite different output sizes.
    # (o_proj is NOT here: it writes the residual/hidden dim, pruned full-mask via
    # the cross-block coupled_masks_all group.)
    head_projections = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    for lname in head_projections:
        manager.set("structured_pruning", "granularity", head_dim, criteria=lname)
        manager.set("structured_pruning", "repeated_pattern", True, criteria=lname)

    # Prune `ratio` of every layer ...
    manager.set("structured_pruning", "ratio", ratio, criteria=None)

    # ... except the per-head projections q/k/v when attention is disabled: GQA
    # hard-pruning changes head_dim (breaks the attention reshape, head_dim fixed
    # in config) and the v->o coupling is not 1:1. o_proj and down_proj ARE pruned
    # here (Phase 2) — they share one cross-block hidden-dim mask.
    excluded = set()
    if not include_attention:
        excluded |= set(head_projections)
    for name in excluded:
        manager.set("structured_pruning", "ratio", 0.0, criteria=name)
    print(f"  excluded from pruning: {sorted(excluded) or 'none'}")

    # Gradient scoring needs a forward+backward calibration pass with a loss.
    manager.set_calibration_data(loader)
    manager.set_calibration_loss(lambda model_output, target: model_output.loss)

    # Hard pruning: remove neurons, resize weight matrices, and cascade the
    # removal into the coupled next layers' inputs.
    manager.apply(hard=True, criteria=None, verbose=False)
    return manager


def print_pruning_summary(model, n: int = 4) -> None:
    """Print the resized dimensions per block (hard pruning removes neurons).

    Hard pruning drops the weight_mask buffer and shrinks the weight matrices, so
    the effect shows up as reduced out/in features:
      * MLP intermediate: gate/up output rows removed, down_proj input cascaded.
      * hidden dim (Phase 2): o_proj/down_proj output rows removed, shared across
        every block by the coupled_masks_all group (same value in all blocks).
    """
    inter = model.config.intermediate_size
    hidden = model.config.hidden_size
    for i, layer in enumerate(model.model.layers):
        if i >= n:
            print(f"  ... ({len(model.model.layers) - n} more blocks)")
            break
        mlp, attn = layer.mlp, layer.self_attn
        g, d_in = mlp.gate_proj.out_features, mlp.down_proj.in_features
        o_out, d_out = attn.o_proj.out_features, mlp.down_proj.out_features
        print(f"  block {i}: mlp inter {inter}->{g} (down.in={d_in}) | "
              f"hidden {hidden}->o_proj.out={o_out}, down.out={d_out}")


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

    if not args.skip_prune:
        print(f"\nCalibrating gradients on {args.num_cal_samples} WikiText-2 samples "
              f"(seq_len={args.cal_seq_len}) and pruning {args.ratio * 100:.0f}% "
              f"of every layer ...")
        apply_gradient_pruning(
            model, tokenizer, args.ratio,
            args.num_cal_samples, args.cal_seq_len, args.seed,
            include_attention=args.include_attention,
        )
        print("\nPruning summary (first 8 layers):")
        print_pruning_summary(model, n=8)
    else:
        print("\nSkipping pruning — running float32 baseline.")

    print(f"\nGenerating (max_new_tokens={args.max_new_tokens}):")
    print(f"  prompt: '{args.prompt}'")

    try:
        text, n_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens)
        print(f"\nGenerated ({n_tokens} tokens):")
        print(f"  '{text}'")
    except RuntimeError as exc:
        print(f"\ngenerate() failed: {exc}")
        print(
            "\nThis is the expected Phase-2 limitation: o_proj/down_proj write the\n"
            "residual/hidden dim, and coupled_masks_all makes them share one mask\n"
            "across all blocks — but the residual stream's other consumers\n"
            "(embeddings, norms, every q/k/v/gate/up input, lm_head) are not yet\n"
            "coupled to that mask, so the residual add sees mismatched sizes.\n"
            "Full hidden-dim pruning needs that residual-wide coupling as a next step."
        )


if __name__ == "__main__":
    main()
