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

    # Convert the compressed model to a transformer-surgeon block model and
    # run generation on that instead of HF generate(). The block model has no
    # batched-prefill kernel, so the prompt is fed one token at a time through
    # the decode step to populate the KV cache before sampling continues.
    python scripts/compression/compress_infer_qwen.py --convert
    python scripts/compression/compress_infer_qwen.py --convert --cache-impl io_scatter
"""

import argparse
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager
from transformersurgeon.utils import convert_for_export
from transformersurgeon.export.common import LLMWrapper, build_zero_caches


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
    parser.add_argument(
        "--convert", action="store_true",
        help="Convert the (compressed) model to a transformer-surgeon block "
             "model (transformersurgeon.utils.convert_for_export) and run "
             "generation on that instead of the HF model. There is no "
             "prefill kernel for the converted block model, so the prompt "
             "is fed one token at a time through the same decode step used "
             "for generation, populating the KV cache before sampling starts.",
    )
    parser.add_argument(
        "--cache-impl", default="mutable",
        choices=["mutable", "io_scatter", "io_concat"],
        help="KV-cache strategy for the converted block model (--convert "
             "only). 'mutable' keeps the cache as internal module state "
             "(module-owned buffers, default). 'io_scatter'/'io_concat' "
             "make the cache explicit graph I/O that the caller threads "
             "through every step (needed for export backends that can't "
             "hold mutable state, e.g. TensorRT) — numerically identical "
             "to 'mutable', just a different calling convention.",
    )
    parser.add_argument(
        "--max-cache-len", type=int, default=None,
        help="Fixed KV-cache length for the converted block model "
             "(--convert only). Must cover the whole prompt + generation "
             "(writes past this length wrap around and silently corrupt "
             "the cache). Defaults to prompt length + max-new-tokens + 32.",
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

    # GQA-coupled per-kv-group pruning of the q/k group (group1). granularity=64
    # is Qwen2-0.5B's head_dim; repeated_pattern="auto" derives one distinct
    # pattern per kv-head from the group's shapes (q_proj 14 heads, k_proj 2 ->
    # 2 patterns) and tiles it group_size=7x over q, 1x over k. Because q/k are
    # position_linked, the mask ties each rotary frequency's real/imag channels,
    # so hard pruning stays RoPE-valid. "magnitude"/"gradient" need calibration;
    # "random" doesn't.
    manager.set("structured_pruning", "method", "magnitude", group="group1")
    manager.set("structured_pruning", "granularity", 64, group="group1")
    manager.set("structured_pruning", "repeated_pattern", "auto", group="group1")
    manager.set("structured_pruning", "ratio", 0.1, group="group1")
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


# --------------------------------------------------------------------------
# Inference on the converted block model (no prefill kernel: prompt tokens
# are fed one at a time through the same decode step, just to populate the
# KV cache, before greedy generation continues from the last prompt token).
# --------------------------------------------------------------------------

def build_converted_wrapper(model, cache_impl: str, max_cache_len: int) -> LLMWrapper:
    """Convert the (compressed) HF model's decoder stack to a transformer-surgeon
    TransformerDecoder block model and wrap it with the source embedding/lm_head.
    """
    converted = convert_for_export(
        model,
        options={
            "use_sdpa": False,
            "cache_impl": cache_impl,
            "max_cache_len": max_cache_len,
        },
    )
    wrapper = LLMWrapper(model.get_input_embeddings(), converted["text"], model.lm_head)
    wrapper.eval()
    return wrapper


@torch.no_grad()
def generate_converted(wrapper: LLMWrapper, tokenizer, prompt: str, max_new_tokens: int):
    """Greedy decode on the converted block model, one token at a time.

    The block model has no batch-prefill path (`in_seq_len` is always 1), so
    the prompt is "prefilled" by iteratively calling the same single-token
    decode step at each prompt position — this fills the KV cache exactly
    like a real prefill would, just without the parallel matmul.
    """
    full_prompt = CHAT_TEMPLATE.format(instruction=prompt)
    prompt_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"][0].long()

    is_io = wrapper.cache_impl != "mutable"
    key_caches, value_caches = build_zero_caches(wrapper.decoder)  # ([], []) for "mutable"

    def step(token_id: torch.Tensor, pos: int) -> torch.Tensor:
        nonlocal key_caches, value_caches
        pos_id = torch.tensor([pos], dtype=torch.long)
        if not is_io:
            return wrapper(token_id, pos_id)
        logits, key_caches, value_caches = wrapper(token_id, pos_id, key_caches, value_caches)
        return logits

    # Prefill (iterative decode): feed each prompt token at its own position.
    logits = None
    for pos in range(prompt_ids.size(0)):
        logits = step(prompt_ids[pos:pos + 1], pos)

    # Decode: append argmax, feed it back at the next position.
    eos_id = tokenizer.eos_token_id
    new_ids = []
    pos = prompt_ids.size(0)
    for _ in range(max_new_tokens):
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        next_id_val = int(next_id.item())
        if eos_id is not None and next_id_val == eos_id:
            break
        new_ids.append(next_id_val)
        logits = step(next_id, pos)
        pos += 1

    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, len(new_ids)


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

    if args.convert:
        prompt_len = len(tokenizer(CHAT_TEMPLATE.format(instruction=args.prompt))["input_ids"])
        max_cache_len = args.max_cache_len or (prompt_len + args.max_new_tokens + 32)
        print(f"\nConverting to transformer-surgeon block model "
              f"(cache_impl={args.cache_impl!r}, max_cache_len={max_cache_len}) ...")
        wrapper = build_converted_wrapper(model, args.cache_impl, max_cache_len)
        text, n_tokens = generate_converted(wrapper, tokenizer, args.prompt, args.max_new_tokens)
    else:
        text, n_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"\nGenerated ({n_tokens} tokens):")
    print(f"  '{text}'")


if __name__ == "__main__":
    main()
