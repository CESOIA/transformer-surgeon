import argparse
import os
import time

import torch

try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None

from transformers import Qwen2TokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text generation with a TensorRT engine exported by exporter_function_test.py"
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default="artifacts/qwen2_tensorrt_full_seed42.pt2",
        help="Path to the exported TensorRT engine (ExportedProgram)",
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


def load_engine_module(engine_path: str):
    """Reload a saved TensorRT engine, tolerating both save formats/APIs."""
    if torch_tensorrt is not None and hasattr(torch_tensorrt, "load"):
        try:
            return torch_tensorrt.load(engine_path).module()
        except Exception:
            pass
    return torch.export.load(engine_path).module()


def main():
    args = parse_args()

    if not os.path.exists(args.engine_path):
        raise FileNotFoundError(
            f"Engine file not found at '{args.engine_path}'. Run exporter_function_test.py first."
        )

    module = load_engine_module(args.engine_path)

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

    t_start = time.perf_counter()

    def _execute_static(next_input_ids: torch.Tensor, effective_len: int) -> torch.Tensor:
        # Static wrapper expects a 1-based effective KV length for the current token.
        #
        # Inputs stay on CPU: the reloaded graph mixes plain torch ops (KV-cache /
        # position-id bookkeeping, whose buffers were traced on CPU) with the
        # TensorRT-accelerated linear subgraphs. torch_tensorrt's runtime wrapper
        # auto-moves tensors to the engine's device only at that subgraph's
        # boundary; pre-moving the whole input to `device` here breaks the
        # surrounding CPU-resident ops with a "Tensor device mismatch" error.
        effective_len_tensor = torch.tensor([effective_len], dtype=torch.long)
        with torch.no_grad():
            out = module(next_input_ids, effective_len_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out)
        return out.cpu()

    logits = None
    # Prefill by decode-iteration: feed each prompt token with its position.
    for effective_len in range(output_ids.size(0)):
        logits = _execute_static(output_ids[effective_len : effective_len + 1], effective_len)

    for _ in range(args.max_new_tokens):
        if logits is None:
            raise RuntimeError("No logits produced before generation loop")

        next_id = logits_to_next_id(logits, args.temperature)
        output_ids = torch.cat([output_ids, next_id], dim=0)
        generated_tokens += 1

        if next_id.item() == tokenizer.eos_token_id:
            break

        logits = _execute_static(next_id, output_ids.size(0))

    total_time_s = time.perf_counter() - t_start
    tokens_per_s = generated_tokens / max(total_time_s, 1e-12)
    avg_token_time_ms = (total_time_s / max(generated_tokens, 1)) * 1000.0

    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    print("\nGeneration result")
    print(f"  engine_path         : {args.engine_path}")
    print(f"  model_name          : {args.model_name}")
    print(f"  prompt              : {args.prompt}")
    print(f"  generated_tokens    : {generated_tokens}")
    print(f"  total_inference_s   : {total_time_s:.6f}")
    print(f"  tokens_per_s        : {tokens_per_s:.2f}")
    print(f"  avg_token_time_ms   : {avg_token_time_ms:.3f}")
    print(f"  output_text         : {generated_text}")


if __name__ == "__main__":
    main()
