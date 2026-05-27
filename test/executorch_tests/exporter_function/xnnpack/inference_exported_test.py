import argparse
import os
import time

import torch
from executorch.runtime import Runtime
from transformers import Qwen2TokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text generation with a .pte exported by exporter_function_test.py"
    )
    parser.add_argument(
        "--pte-path",
        type=str,
        default="artifacts/export_hf_xnnpack_int8.pte",
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
    parser.add_argument(
        "--static-seq-len-1",
        action="store_true",
        help=(
            "Use static seq_len=1 invocation contract: prefill is performed "
            "by iterating decode one token at a time."
        ),
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

    output_ids = input_ids.clone()
    generated_tokens = 0

    t_start = time.perf_counter()

    def _execute_dynamic(cur_ids: torch.Tensor) -> torch.Tensor:
        # Dynamic wrapper uses cache_len_tensor.size(0) as cache length.
        cache_len_tensor = torch.ones(cur_ids.size(1), dtype=torch.float32)
        out = method.execute([cur_ids, cache_len_tensor])[0]
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out)
        return out

    def _execute_static(next_input_ids: torch.Tensor, effective_len: int) -> torch.Tensor:
        # Static wrapper expects a 1-based effective KV length for the current token.
        effective_len_tensor = torch.tensor([effective_len], dtype=torch.long)
        out = method.execute([next_input_ids, effective_len_tensor])[0]
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out)
        return out

    logits = None
    if args.static_seq_len_1:
        # Prefill by decode-iteration: feed each prompt token with its position.
        for effective_len in range(1, output_ids.size(1) + 1):
            logits = _execute_static(output_ids[:, effective_len - 1 : effective_len], effective_len)
    else:
        logits = _execute_dynamic(output_ids)

    for _ in range(args.max_new_tokens):
        if logits is None:
            raise RuntimeError("No logits produced before generation loop")

        next_id = logits_to_next_id(logits, args.temperature)
        output_ids = torch.cat([output_ids, next_id], dim=1)
        generated_tokens += 1

        if next_id.item() == tokenizer.eos_token_id:
            break

        if args.static_seq_len_1:
            logits = _execute_static(next_id, output_ids.size(1))
        else:
            logits = _execute_dynamic(output_ids)

    total_time_s = time.perf_counter() - t_start
    tokens_per_s = generated_tokens / max(total_time_s, 1e-12)
    avg_token_time_ms = (total_time_s / max(generated_tokens, 1)) * 1000.0

    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    print("\nGeneration result")
    print(f"  pte_path            : {args.pte_path}")
    print(f"  model_name          : {args.model_name}")
    print(f"  prompt              : {args.prompt}")
    print(f"  generated_tokens    : {generated_tokens}")
    print(f"  total_inference_s   : {total_time_s:.6f}")
    print(f"  tokens_per_s        : {tokens_per_s:.2f}")
    print(f"  avg_token_time_ms   : {avg_token_time_ms:.3f}")
    print(f"  output_text         : {generated_text}")


if __name__ == "__main__":
    main()
