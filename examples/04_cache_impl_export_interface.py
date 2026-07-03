"""
Understand the exported-model interface for each KV-cache implementation.

When you convert a causal LM with transformer-surgeon and export it, the KV
cache is exposed to the runtime in one of two ways, selected by the
``cache_impl`` option:

  - "mutable"    : the cache lives *inside* the graph as module state. The
                   exported program takes (input_ids, pos_id) and returns logits.
                   Fastest on QNN (in-place ScatterNd), but not portable to
                   functional runtimes like TensorRT.

  - "io_scatter" : the cache is *explicit graph I/O*. The program takes
    "io_concat"    (input_ids, pos_id, key_caches, value_caches) and returns
                   (logits, new_key_caches, new_value_caches). The caller owns
                   the cache and feeds the returned tensors back each step.
                   Portable to QNN (ScatterNd) and TensorRT (ScatterND).
                   io_concat expresses the same write without index_put (a
                   positional-mask/where write) for backends lacking a scatter
                   converter.

All three are numerically equivalent — they differ only in the forward-call
interface. This example converts Qwen2-0.5B under each mode, prints the exact
input/output contract a runtime will see, and runs an identical greedy
generation loop so you can see how to drive each interface.

Usage:
    python examples/04_cache_impl_export_interface.py
    python examples/04_cache_impl_export_interface.py --cache-impl io_scatter
    python examples/04_cache_impl_export_interface.py --model-name Qwen/Qwen2-0.5B-Instruct --max-new-tokens 24

Requirements:
    pip install transformer-surgeon transformers torch
"""

import argparse

import torch
from transformers import AutoTokenizer

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.utils import convert_for_export
from transformersurgeon.export.common import LLMWrapper, build_zero_caches


def parse_args():
    parser = argparse.ArgumentParser(description="Exported-model interface per cache_impl")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--prompt", type=str,
                        default="Give one short fact about the ocean.")
    parser.add_argument("--cache-impl", type=str, default="all",
                        choices=["all", "mutable", "io_scatter", "io_concat"],
                        help="Which cache implementation(s) to demonstrate.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-cache-len", type=int, default=256)
    return parser.parse_args()


def build_wrapper(model, cache_impl: str, max_cache_len: int) -> LLMWrapper:
    """Convert an HF model to export-ready components and wrap them.

    ``convert_for_export`` returns {"text": decoder, ...}; the embedding and LM
    head come straight from the source model. ``cache_impl`` is threaded through
    ``options`` into the converted decoder's config.
    """
    converted = convert_for_export(
        model,
        options={
            "use_sdpa": False,             # explicit attention (export-friendly)
            "cache_impl": cache_impl,      # <-- selects the exported interface
            "max_cache_len": max_cache_len,
        },
    )
    decoder = converted["text"]
    wrapper = LLMWrapper(
        model.get_input_embeddings(),
        decoder,
        model.lm_head,
    )
    wrapper.eval()
    return wrapper


def describe_interface(wrapper: LLMWrapper) -> None:
    """Print the exact input/output contract the runtime (ExecuTorch/TRT) sees."""
    n_layers = len(wrapper.decoder.blocks)
    print(f"  cache_impl = {wrapper.cache_impl!r}")
    if wrapper.cache_impl == "mutable":
        print("    inputs : [input_ids(1,), pos_id(1,)]")
        print("    outputs: [logits(vocab,)]")
        print("    cache  : internal module state (not a graph tensor)")
    else:
        print(f"    inputs : [input_ids(1,), pos_id(1,), "
              f"*key_caches({n_layers}), *value_caches({n_layers})]")
        print(f"    outputs: [logits(vocab,), "
              f"*new_key_caches({n_layers}), *new_value_caches({n_layers})]")
        attn0 = wrapper.decoder.blocks[0].attn
        print(f"    each cache tensor: "
              f"(max_cache_len={attn0.max_cache_length}, "
              f"kv_heads={attn0.kv_num_heads}, head_dim={attn0.head_dim})")


@torch.no_grad()
def generate(wrapper: LLMWrapper, prompt_ids: torch.Tensor, max_new_tokens: int,
             eos_id: int | None) -> torch.Tensor:
    """Greedy decode, one token at a time. Shows how to drive each interface.

    The wrapper always consumes a single token (in_seq_len == 1). ``pos_id`` is
    the absolute position of the current token, which is also where the new K/V
    is written into the fixed-size cache.
    """
    output_ids = prompt_ids.clone()
    is_io = wrapper.cache_impl != "mutable"

    # io modes: the caller owns the caches and threads them through every call.
    key_caches, value_caches = build_zero_caches(wrapper.decoder)  # ([],[]) for mutable

    def step(token_id: torch.Tensor, pos: int) -> torch.Tensor:
        nonlocal key_caches, value_caches
        pos_id = torch.tensor([pos], dtype=torch.long)
        if not is_io:
            return wrapper(token_id, pos_id)                       # -> logits
        logits, key_caches, value_caches = wrapper(               # cache in -> cache out
            token_id, pos_id, key_caches, value_caches
        )
        return logits

    # Prefill: feed each prompt token at its own position.
    logits = None
    for pos in range(output_ids.size(0)):
        logits = step(output_ids[pos:pos + 1], pos)

    # Decode: append argmax, feed it back at the next position.
    for _ in range(max_new_tokens):
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_id], dim=0)
        if eos_id is not None and int(next_id.item()) == eos_id:
            break
        logits = step(next_id, output_ids.size(0) - 1)

    return output_ids


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # A transformer-surgeon compress model carries the ``indexing`` metadata that
    # convert_for_export uses to map HF submodules onto TransformerDecoder blocks.
    # (You could also apply LRD/quantization here first — see examples 01/02.)
    model = Qwen2ForCausalLMCompress.from_pretrained(args.model_name, torch_dtype="auto")
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].long()

    impls = ["mutable", "io_scatter", "io_concat"] if args.cache_impl == "all" else [args.cache_impl]

    print(f"\nModel : {args.model_name}")
    print(f"Prompt: {args.prompt}\n")

    generated_texts = {}
    for impl in impls:
        print(f"=== cache_impl = {impl} ===")
        wrapper = build_wrapper(model, impl, args.max_cache_len)
        describe_interface(wrapper)

        output_ids = generate(wrapper, prompt_ids, args.max_new_tokens, tokenizer.eos_token_id)
        text = tokenizer.decode(output_ids[prompt_ids.size(0):], skip_special_tokens=True)
        generated_texts[impl] = text
        print(f"    generated: {text!r}\n")

    # All implementations are numerically equivalent → identical greedy output.
    if len(generated_texts) > 1:
        unique = set(generated_texts.values())
        status = "IDENTICAL ✓" if len(unique) == 1 else "DIFFERENT ✗ (unexpected!)"
        print(f"Cross-impl greedy output: {status}")


if __name__ == "__main__":
    main()
