import argparse
import time

import torch
from transformers import Qwen2TokenizerFast

from transformersurgeon import (
    Qwen2CompressionSchemesManager,
    Qwen2ForCausalLMCompress,
    convert_for_export,
)
from transformersurgeon.utils import CompressionSchemesManager


def parse_args():
    parser = argparse.ArgumentParser(description="Compression tests in pure PyTorch")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--prompt", type=str, default="Write one short sentence about compression.")
    parser.add_argument("--max-input-len", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-len", type=int, default=128)
    parser.add_argument("--lrd-rank", type=int, default=640)
    parser.add_argument(
        "--mode",
        type=str,
        default="hf_manager",
        choices=["hf_manager", "convert_then_compress"],
        help="hf_manager: compress HF model with manager and use model.generate. "
        "convert_then_compress: convert first, compress converted graph, and custom-generate in torch.",
    )
    return parser.parse_args()


def _tokenize_prompt(tokenizer, prompt, max_input_len, device):
    tokens = tokenizer(prompt, return_tensors="pt")
    return tokens["input_ids"][:, :max_input_len].to(device)


def _reset_decoder_kv_cache(decoder: torch.nn.Module):
    for module in decoder.modules():
        if hasattr(module, "key_cache") and isinstance(module.key_cache, torch.Tensor):
            module.key_cache.zero_()
        if hasattr(module, "value_cache") and isinstance(module.value_cache, torch.Tensor):
            module.value_cache.zero_()


def _custom_generate_greedy(embedding, decoder, lm_head, input_ids, max_new_tokens, max_cache_len):
    """Greedy generation with converted decoder only (no HF generate)."""
    _reset_decoder_kv_cache(decoder)

    generated = input_ids.clone()
    cur_cache_len = generated.shape[1]

    with torch.no_grad():
        x = embedding(generated)
        hidden = decoder(x, cur_cache_len)
        hidden = hidden.to(dtype=lm_head.weight.dtype)
        logits = lm_head(hidden[:, -1, :])

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if generated.shape[1] >= max_cache_len:
            break

        cur_cache_len += 1
        with torch.no_grad():
            x = embedding(next_token)
            hidden = decoder(x, cur_cache_len)
            hidden = hidden.to(dtype=lm_head.weight.dtype)
            logits = lm_head(hidden[:, -1, :])

    return generated


def test_hf_compressed_with_manager(args):
    """Path 1: compress HF model through HF manager and run generate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_name)
    model = Qwen2ForCausalLMCompress.from_pretrained(args.model_name, torch_dtype="auto").to(device)
    model.eval()

    manager = Qwen2CompressionSchemesManager(model)
    manager.set("lrd", "rank", args.lrd_rank, criteria=[[0, "self_attn.q_proj"]])
    manager.set("lrd", "rank", args.lrd_rank, criteria=[[1, "mlp.up_proj"]])
    manager.apply(hard=False)

    input_ids = _tokenize_prompt(tokenizer, args.prompt, args.max_input_len, device)

    tic = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    toc = time.perf_counter()

    new_tokens = int(generated.shape[1] - input_ids.shape[1])
    text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)

    print("[hf_manager] output:", text)
    print(f"[hf_manager] generated tokens: {new_tokens}")
    print(f"[hf_manager] speed: {new_tokens / max(toc - tic, 1e-6):.4f} tokens/s")


def test_convert_then_compress(args):
    """
    Path 2:
      1) convert HF model to custom graph,
      2) compress converted graph using inherited config/indexing from converter,
      3) run custom generation loop in torch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_name)

    model = Qwen2ForCausalLMCompress.from_pretrained(args.model_name, torch_dtype="auto").to(device)
    model.eval()

    converted = convert_for_export(model, options={"use_sdpa": False}, verbose=False)
    decoder = converted["text"].to(device).eval()
    embedding = model.get_input_embeddings().to(device).eval()
    lm_head = model.lm_head.to(device).eval()

    print(decoder.config)

    # Converter should provide manager-ready config/indexing on converted model.
    manager = CompressionSchemesManager(decoder, decoder.indexing)
    manager.set("lrd", "rank", args.lrd_rank, criteria=[[0, "attn.q_proj"]])
    manager.set("lrd", "rank", args.lrd_rank, criteria=[[1, "mlp.up_proj"]])
    manager.apply(hard=False)

    input_ids = _tokenize_prompt(tokenizer, args.prompt, args.max_input_len, device)

    tic = time.perf_counter()
    generated = _custom_generate_greedy(
        embedding,
        decoder,
        lm_head,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.cache_len,
    )
    if torch.cuda.is_available() and next(lm_head.parameters()).is_cuda:
        torch.cuda.synchronize()
    toc = time.perf_counter()

    new_tokens = int(generated.shape[1] - input_ids.shape[1])
    text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)

    print("[convert_then_compress] output:", text)
    print(f"[convert_then_compress] generated tokens: {new_tokens}")
    print(f"[convert_then_compress] speed: {new_tokens / max(toc - tic, 1e-6):.4f} tokens/s")


def main():
    args = parse_args()
    print(f"Mode: {args.mode}")

    if args.mode == "hf_manager":
        test_hf_compressed_with_manager(args)
    else:
        test_convert_then_compress(args)


if __name__ == "__main__":
    main()
