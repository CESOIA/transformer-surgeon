"""Export Qwen2.5-0.5B with 8-bit MLP weights and an 8-bit KV cache to XNNPACK.

Run the exported program on the host, compare its logits with the original model,
and print greedy completions from both. Run in the activated cesoia environment:

    python scripts/executorch/xnnpack/export_8bit_kv_cache.py [output.pte]
"""

import argparse
from pathlib import Path
import time

import torch
from executorch.runtime import Runtime
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress
from transformersurgeon.export import export_to_backend, load_llm_manifest
from transformersurgeon.export.backends import XNNPACKExportConfig


MODEL = "Qwen/Qwen2.5-0.5B"
CACHE_LEN = 128
NEW_TOKENS = 20
MAX_RELATIVE_LOGIT_ERROR = 0.4


def run_exported(pte_path, tokens, prompt, eos_token_ids):
    program = Runtime.get().load_program(pte_path)
    method = program.load_method("forward")

    def step(token, position):
        return method.execute([torch.tensor([token]), torch.tensor([position])])[0].reshape(-1)

    # ExecuTorch can reuse the output buffer on the next call.
    logits = torch.stack([step(token, pos).clone() for pos, token in enumerate(tokens)])

    # Start at position 0 for the new prompt. The causal mask hides old cache
    # slots until the new sequence overwrites them.
    generated = list(prompt)
    for pos in range(len(prompt) + NEW_TOKENS - 1):
        scores = step(generated[pos], pos)
        if pos >= len(prompt) - 1:
            next_token = int(scores.argmax())
            generated.append(next_token)
            if next_token in eos_token_ids:
                break
    return logits, generated[len(prompt):]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_path", nargs="?", default="artifacts/qwen25_xnnpack_w8_kv8.pte")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = Qwen2ForCausalLMCompress.from_pretrained(MODEL, dtype=torch.float32).eval()
    tokens = tokenizer.encode("The Eiffel Tower is in Paris, the capital of France. "
                              "It was built for the World's Fair of 1889.")
    prompt = tokenizer.encode("The capital of France is")
    with torch.no_grad():
        expected = model(torch.tensor([tokens])).logits[0]
        original_completion = model.generate(
            torch.tensor([prompt]), max_new_tokens=NEW_TOKENS, do_sample=False
        )[0, len(prompt):]

    # XNNPACK's torchao path uses dynamic per-token INT8 activations for the MLP linears.
    manager = Qwen2CompressionSchemesManager(model)
    for name, value in {
        "precision": 8, "method": "vanilla", "granularity": "per_channel",
        "precision_activation": 8, "method_activation": "dynamic_per_token",
    }.items():
        manager.set("quantization", name, value, criteria=["mlp"])
    manager.apply(hard=True, criteria=["mlp"])

    config = XNNPACKExportConfig(
        output_path=args.output_path,
        max_seq_len=CACHE_LEN,
        quant_path="torchao",
        # INT8 computes its scale for each KV row during inference.
        convert_options={"cache_impl": "mutable", "kv_precision": "int8"},
    )
    start = time.perf_counter()
    result = export_to_backend(model, config)
    print(f"Exported {result.pte_path} ({Path(result.pte_path).stat().st_size / 2**20:.0f} MiB) "
          f"in {time.perf_counter() - start:.1f} s")
    del manager, model

    manifest = load_llm_manifest(result.pte_path)
    assert manifest["kv_precision"] == "int8", manifest["kv_precision"]
    assert manifest["cache_impl"] == "mutable", manifest["cache_impl"]
    assert all(layer[kind]["dtype"] == "uint8" for layer in manifest["caches"] for kind in ("key", "value"))
    print(f"Manifest confirms uint8 key and value storage for {manifest['num_layers']} layers")

    got, completion = run_exported(result.pte_path, tokens, prompt, manifest["eos_token_id"])
    assert got.shape == expected.shape, (got.shape, expected.shape)
    error = ((got - expected).norm() / expected.norm()).item()
    same = (got.argmax(-1) == expected.argmax(-1)).float().mean().item()
    print(f"Relative logit error {error:.3f}, same next-token prediction at {same:.0%} of {len(tokens)} positions")
    assert error < MAX_RELATIVE_LOGIT_ERROR, f"Relative logit error exceeds {MAX_RELATIVE_LOGIT_ERROR}: {error}"
    print(f"XNNPACK, 8-bit cache: {tokenizer.decode(completion, skip_special_tokens=True)!r}")
    print(f"Original float model: {tokenizer.decode(original_completion, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
