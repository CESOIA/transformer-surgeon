import argparse
import os
import time

import torch
from executorch.runtime import Runtime
from transformers import Qwen2TokenizerFast

from transformersurgeon.export.common import load_llm_manifest, zero_caches_from_manifest

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text generation with a .pte exported by exporter_function_test.py"
    )
    parser.add_argument(
        "--pte-path",
        type=str,
        default="artifacts/export_hf_xnnpack_auto.pte",
        help="Path to exported .pte file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HF tokenizer identifier used during export",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tell me one short fact about France.",
        help="Prompt text",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. <= 0 uses greedy decoding.",
    )
    return parser.parse_args()


def logits_to_next_id(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / float(temperature)
    probs = torch.nn.functional.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def main():
    args = parse_args()

    if not os.path.exists(args.pte_path):
        raise FileNotFoundError(
            f"PTE file not found at '{args.pte_path}'. Run exporter_function_test.py first."
        )

    # Cache implementation and per-layer cache shapes/dtype come from the
    # manifest the exporter writes next to the .pte.
    manifest = load_llm_manifest(args.pte_path)
    if manifest.get("attn_impl") == "custom_sdpa":
        # Registers the llama::custom_sdpa/update_cache runtime kernels
        # (torch.ops.llama.* meta registration alone, done at export time in
        # a different process, isn't enough -- this process's operator
        # registry needs the same import before loading the .pte's method).
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

    runtime = Runtime.get()
    program = runtime.load_program(args.pte_path)
    method = program.load_method("forward")

    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_name)

    template = (
        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
        "<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    input_ids = tokenizer(
        template.format(instruction=args.prompt),
        return_tensors="pt",
    )["input_ids"].long()

    output_ids = input_ids[0].clone()
    generated_tokens = 0

    # For io_* modes, the KV cache is explicit graph I/O: host holds it and feeds
    # the returned cache back each step.
    io_mode = manifest["cache_io"]
    num_layers = manifest["num_layers"]
    kv_state = zero_caches_from_manifest(manifest) if io_mode else None

    t_start = time.perf_counter()

    def _execute_static(next_input_ids: torch.Tensor, effective_len: int) -> torch.Tensor:
        nonlocal kv_state
        # Cache slot (0-based position) of the token being fed.
        effective_len_tensor = torch.tensor([effective_len], dtype=torch.long)
        if not io_mode:
            out = method.execute([next_input_ids, effective_len_tensor])[0]
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out)
            return out

        # io mode: flatten caches into the argument list, read updated caches back.
        key_caches, value_caches = kv_state
        outputs = method.execute(
            [next_input_ids, effective_len_tensor, *key_caches, *value_caches]
        )
        logits = outputs[0]
        new_key_caches = list(outputs[1:1 + num_layers])
        new_value_caches = list(outputs[1 + num_layers:1 + 2 * num_layers])
        kv_state = (new_key_caches, new_value_caches)
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        return logits

    logits = None
    # Prefill by decode-iteration: feed each prompt token with its position.
    for effective_len in range(output_ids.size(0)):
        logits = _execute_static(output_ids[effective_len : effective_len+1], effective_len)

    for _ in range(args.max_new_tokens):
        if logits is None:
            raise RuntimeError("No logits produced before generation loop")

        next_id = logits_to_next_id(logits, args.temperature)
        output_ids = torch.cat([output_ids, next_id], dim=0)
        generated_tokens += 1

        if next_id.item() == tokenizer.eos_token_id:
            break

        # next_id is the last element of output_ids: its cache slot is size - 1.
        logits = _execute_static(next_id, output_ids.size(0) - 1)

    total_time_s = time.perf_counter() - t_start
    tokens_per_s = generated_tokens / max(total_time_s, 1e-12)
    avg_token_time_ms = (total_time_s / max(generated_tokens, 1)) * 1000.0

    # output_ids is 1-D: decode it as one sequence (batch_decode would split it per token).
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\nGeneration result")
    print(f"  pte_path            : {args.pte_path}")
    print(f"  cache_impl          : {manifest['cache_impl']}")
    print(f"  model_name          : {args.model_name}")
    print(f"  prompt              : {args.prompt}")
    print(f"  generated_tokens    : {generated_tokens}")
    print(f"  total_inference_s   : {total_time_s:.6f}")
    print(f"  tokens_per_s        : {tokens_per_s:.2f}")
    print(f"  avg_token_time_ms   : {avg_token_time_ms:.3f}")
    print(f"  output_text         : {generated_text}")


if __name__ == "__main__":
    main()
