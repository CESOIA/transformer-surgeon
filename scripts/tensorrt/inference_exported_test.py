import argparse
import json
import os
import time

import torch

try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None

from transformers import Qwen2TokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


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
    parser.add_argument(
        "--cache-impl",
        type=str,
        default=None,
        choices=["mutable", "io_scatter", "io_concat"],
        help="KV-cache implementation the engine was exported with. 'mutable' keeps the "
             "legacy (input_ids, pos_id) -> logits contract; 'io_scatter'/'io_concat' "
             "expose the cache as explicit inout graph buffers, so the geometry args "
             "below are then required. Defaults to the value in <engine-path "
             "stem>.cache_meta.json if that sidecar (written by "
             "exporter_function_test.py) exists next to --engine-path, else 'mutable'.",
    )
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Decoder layers (required for io_* cache modes; falls back to cache_meta.json).")
    parser.add_argument("--kv-num-heads", type=int, default=None,
                        help="KV heads per layer (required for io_* cache modes; falls back to cache_meta.json).")
    parser.add_argument("--head-dim", type=int, default=None,
                        help="Head dimension (required for io_* cache modes; falls back to cache_meta.json).")
    parser.add_argument("--max-cache-len", type=int, default=None,
                        help="Fixed cache length (required for io_* cache modes; falls back to cache_meta.json).")
    parser.add_argument(
        "--cache-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for the zero-initialized inout KV-cache buffers. Must match the "
             "dtype the engine was exported/compiled with (see --float-type on "
             "exporter_function_test.py).",
    )
    return parser.parse_args()


def _load_cache_metadata(engine_path: str) -> dict | None:
    """Load the geometry sidecar written by exporter_function_test.py, if present."""
    meta_path = os.path.splitext(engine_path)[0] + ".cache_meta.json"
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


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


def _module_device(module) -> torch.device | None:
    return next((p.device for p in getattr(module, "parameters", lambda: [])()), None)


def _to_device(x, device):
    # Caches are lists of tensors; torch_tensorrt's runtime wrapper auto-moves
    # top-level Tensor args to the engine's device but does not recurse into
    # list/tuple args, so inout cache buffers must be moved explicitly.
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(e, device) for e in x)
    return x


def main():
    args = parse_args()

    if not os.path.exists(args.engine_path):
        raise FileNotFoundError(
            f"Engine file not found at '{args.engine_path}'. Run exporter_function_test.py first."
        )

    meta = _load_cache_metadata(args.engine_path)
    if meta is not None:
        meta_path = os.path.splitext(args.engine_path)[0] + ".cache_meta.json"
        print(f"Loaded KV-cache geometry from {meta_path}")
        if args.cache_impl is None:
            args.cache_impl = meta.get("cache_impl", "mutable")
        if args.num_layers is None:
            args.num_layers = meta.get("num_layers")
        if args.kv_num_heads is None:
            args.kv_num_heads = meta.get("kv_num_heads")
        if args.head_dim is None:
            args.head_dim = meta.get("head_dim")
        if args.max_cache_len is None:
            args.max_cache_len = meta.get("max_cache_len")
    if args.cache_impl is None:
        args.cache_impl = "mutable"

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

    # For io_* modes, the KV cache is explicit graph I/O: host holds it and feeds
    # the returned cache back each step. Initialize zero caches from geometry.
    io_mode = args.cache_impl != "mutable"
    kv_state = None
    if io_mode:
        missing = [n for n, v in (
            ("--num-layers", args.num_layers),
            ("--kv-num-heads", args.kv_num_heads),
            ("--head-dim", args.head_dim),
            ("--max-cache-len", args.max_cache_len),
        ) if v is None]
        if missing:
            raise ValueError(f"cache_impl={args.cache_impl} requires geometry args: {', '.join(missing)}")
        cache_dtype = _DTYPE_MAP[args.cache_dtype]
        shape = (args.max_cache_len, args.kv_num_heads, args.head_dim)
        key_caches = [torch.zeros(shape, dtype=cache_dtype) for _ in range(args.num_layers)]
        value_caches = [torch.zeros(shape, dtype=cache_dtype) for _ in range(args.num_layers)]
        kv_state = (key_caches, value_caches)

    module_device = _module_device(module)

    t_start = time.perf_counter()

    def _execute_static(next_input_ids: torch.Tensor, effective_len: int) -> torch.Tensor:
        nonlocal kv_state
        # Static wrapper expects a 1-based effective KV length for the current token.
        #
        # Inputs stay on CPU: the reloaded graph mixes plain torch ops (KV-cache /
        # position-id bookkeeping, whose buffers were traced on CPU) with the
        # TensorRT-accelerated linear subgraphs. torch_tensorrt's runtime wrapper
        # auto-moves tensors to the engine's device only at that subgraph's
        # boundary; pre-moving the whole input to `device` here breaks the
        # surrounding CPU-resident ops with a "Tensor device mismatch" error.
        effective_len_tensor = torch.tensor([effective_len], dtype=torch.long)

        if not io_mode:
            with torch.no_grad():
                out = module(next_input_ids, effective_len_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out)
            return out.cpu()

        # io_* modes: the cache lists are a nested arg, so (unlike the plain
        # tensor args above) they are not auto-moved by the runtime wrapper.
        key_caches, value_caches = kv_state
        if module_device is not None and module_device.type == "cuda":
            key_caches = _to_device(key_caches, module_device)
            value_caches = _to_device(value_caches, module_device)

        with torch.no_grad():
            out = module(next_input_ids, effective_len_tensor, key_caches, value_caches)
        logits, new_key_caches, new_value_caches = out
        kv_state = (list(new_key_caches), list(new_value_caches))
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        return logits.cpu()

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
    print(f"  cache_impl          : {args.cache_impl}")
    print(f"  model_name          : {args.model_name}")
    print(f"  prompt              : {args.prompt}")
    print(f"  generated_tokens    : {generated_tokens}")
    print(f"  total_inference_s   : {total_time_s:.6f}")
    print(f"  tokens_per_s        : {tokens_per_s:.2f}")
    print(f"  avg_token_time_ms   : {avg_token_time_ms:.3f}")
    print(f"  output_text         : {generated_text}")


if __name__ == "__main__":
    main()
