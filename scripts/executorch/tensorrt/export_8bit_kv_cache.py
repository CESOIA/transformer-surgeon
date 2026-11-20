"""Export Qwen2.5-0.5B with 8-bit MLP weights and an 8-bit KV cache to TensorRT.

Run the engine, compare its logits with the original fp16 model, and print greedy
completions from both. Use --no-build to export ONNX and its manifest on a CPU host.

    python scripts/executorch/tensorrt/export_8bit_kv_cache.py [output.onnx]
"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress
from transformersurgeon.export import export_to_backend, load_llm_manifest
from transformersurgeon.export.backends import TensorRTExportConfig
from transformersurgeon.export.backends.tensorrt.session import TensorRTLLMSession


MODEL = "Qwen/Qwen2.5-0.5B"
CACHE_LEN = 128
NEW_TOKENS = 20
MAX_RELATIVE_LOGIT_ERROR = 0.4


def run_engine(session, tokens):
    """Feed the reference tokens through the engine's prefill and decode profiles."""
    session.prefill(tokens[:1])
    logits = [session.logits.float().cpu().reshape(-1)]
    for token in tokens[1:].tolist():
        session.input_ids.fill_(token)
        session.step()
        logits.append(session.logits.float().cpu().reshape(-1))
    return torch.stack(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_path", nargs="?", default="artifacts/qwen25_tensorrt_w8_kv8.onnx")
    parser.add_argument("--no-build", action="store_true", help="export ONNX and its manifest without building an engine")
    args = parser.parse_args()
    if not args.no_build and not torch.cuda.is_available():
        parser.error("building and running TensorRT requires a CUDA GPU; use --no-build for ONNX export")

    model = Qwen2ForCausalLMCompress.from_pretrained(MODEL, dtype=torch.float16).eval()
    if not args.no_build:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokens = torch.tensor(tokenizer.encode("The Eiffel Tower is in Paris, the capital of France. "
                                               "It was built for the World's Fair of 1889."))
        prompt = torch.tensor(tokenizer.encode("The capital of France is"))
        model.cuda()
        with torch.no_grad():
            expected = model(tokens.unsqueeze(0).cuda()).logits[0].float().cpu()
            original_completion = model.generate(
                prompt.unsqueeze(0).cuda(), max_new_tokens=NEW_TOKENS, do_sample=False
            )[0, prompt.numel():].cpu()
        model.cpu()
        torch.cuda.empty_cache()

    manager = Qwen2CompressionSchemesManager(model)
    criteria = ["mlp"]
    for name, value in {"precision": 8, "method": "vanilla", "granularity": "per_channel"}.items():
        manager.set("quantization", name, value, criteria=criteria)
    manager.apply(hard=True, criteria=criteria)

    config = TensorRTExportConfig(
        output_path=args.output_path,
        max_seq_len=CACHE_LEN,
        max_input_len=32,
        build_engine=not args.no_build,
        # The engine returns uint8 rows and the session writes them into the cache.
        convert_options={"cache_impl": "io_rows", "kv_precision": "int8", "rmsnorm_impl": "rmsnorm_upcast"},
    )
    start = time.perf_counter()
    result = export_to_backend(model, config)
    print(f"Exported {result.onnx_path} in {time.perf_counter() - start:.1f} s")
    print(f"Manifest: {result.manifest_path}")
    del manager, model

    manifest = load_llm_manifest(result.manifest_path)
    assert manifest["kv_precision"] == "int8", manifest["kv_precision"]
    assert manifest["cache_impl"] == "io_rows" and manifest["cache_rows"], manifest["cache_impl"]
    assert all(layer[kind]["dtype"] == "uint8" for layer in manifest["caches"] for kind in ("key", "value"))
    print(f"Manifest confirms uint8 key and value storage for {manifest['num_layers']} layers")
    if args.no_build:
        return

    print(f"Engine: {result.engine_path}")
    session = TensorRTLLMSession(result.manifest_path, result.engine_path)
    assert {cache.dtype for cache in session.caches.values()} == {torch.uint8}
    got = run_engine(session, tokens)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    error = ((got - expected).norm() / expected.norm()).item()
    same = (got.argmax(-1) == expected.argmax(-1)).float().mean().item()
    print(f"Relative logit error {error:.3f}, same next-token prediction at {same:.0%} of {tokens.numel()} positions")
    assert error < MAX_RELATIVE_LOGIT_ERROR, f"Relative logit error exceeds {MAX_RELATIVE_LOGIT_ERROR}: {error}"

    completion = session.generate(prompt, NEW_TOKENS)
    print(f"TensorRT, 8-bit cache: {tokenizer.decode(completion, skip_special_tokens=True)!r}")
    print(f"Original fp16 model: {tokenizer.decode(original_completion, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
