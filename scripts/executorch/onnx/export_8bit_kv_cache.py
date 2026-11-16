"""Export Qwen2.5-0.5B with 8-bit MLP weights and an 8-bit KV cache to ONNX.

Run the model in ONNX Runtime on the CPU, compare its logits with the original model,
and print greedy completions from both. Run in the activated cesoia environment:

    python scripts/executorch/onnx/export_8bit_kv_cache.py [output.onnx]
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress
from transformersurgeon.export import export_to_backend, load_llm_manifest, zero_caches_from_manifest
from transformersurgeon.export.backends import ONNXExportConfig


MODEL = "Qwen/Qwen2.5-0.5B"
CACHE_LEN = 128
MAX_INPUT_LEN = 32
NEW_TOKENS = 20
MAX_RELATIVE_LOGIT_ERROR = 0.4


class OnnxDecoder:
    """Runs the exported graph in ONNX Runtime, with the caches updated in place.

    Each present_* output is bound to the buffer of its past_* input, so the graph's
    TensorScatter writes the new rows straight into the cache (the io_inplace contract).
    """

    def __init__(self, onnx_path, manifest):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.manifest = manifest
        self.reset()

    def reset(self):
        """Empty caches, for a new sequence."""
        keys, values = zero_caches_from_manifest(self.manifest)
        self.caches = {f"{kind}_{i}": ort.OrtValue.ortvalue_from_numpy(cache.numpy())
                       for kind, caches in (("key", keys), ("value", values)) for i, cache in enumerate(caches)}

    def __call__(self, tokens, position):
        """Logits of the last of ``tokens``, whose rows are written into the caches from ``position`` on."""
        binding = self.session.io_binding()
        binding.bind_cpu_input("input_ids", np.asarray(tokens, dtype=np.int64))
        binding.bind_cpu_input("pos_id", np.asarray([position], dtype=np.int64))
        binding.bind_output("logits")
        for name, cache in self.caches.items():
            binding.bind_ortvalue_input(f"past_{name}", cache)
            binding.bind_ortvalue_output(f"present_{name}", cache)
        self.session.run_with_iobinding(binding)
        return torch.from_numpy(binding.get_outputs()[0].numpy()).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_path", nargs="?", default="artifacts/qwen25_onnx_w8_kv8.onnx")
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

    # Weight-only INT8 MLP linears: the exporter stores int8 weights read through DequantizeLinear.
    manager = Qwen2CompressionSchemesManager(model)
    for name, value in {"precision": 8, "method": "vanilla", "granularity": "per_channel"}.items():
        manager.set("quantization", name, value, criteria=["mlp"])
    manager.apply(hard=True, criteria=["mlp"])

    config = ONNXExportConfig(
        output_path=args.output_path,
        max_seq_len=CACHE_LEN,
        max_input_len=MAX_INPUT_LEN,
        # An opset-24 TensorScatter writes the uint8 rows into the cache the caller binds.
        convert_options={"cache_impl": "io_inplace", "kv_precision": "int8"},
    )
    start = time.perf_counter()
    result = export_to_backend(model, config)
    weights = os.path.getsize(result.onnx_path + ".data")
    print(f"Exported {result.onnx_path} ({os.path.getsize(result.onnx_path) / 2**20:.1f} MiB "
          f"+ {weights / 2**20:.0f} MiB of weights) in {time.perf_counter() - start:.1f} s")
    del manager, model

    manifest = load_llm_manifest(result.manifest_path)
    assert manifest["kv_precision"] == "int8", manifest["kv_precision"]
    assert manifest["cache_impl"] == "io_inplace" and not manifest["cache_rows"], manifest["cache_impl"]
    assert all(layer[kind]["dtype"] == "uint8" for layer in manifest["caches"] for kind in ("key", "value"))
    print(f"Manifest confirms uint8 key and value storage for {manifest['num_layers']} layers "
          f"and int8 weights in {len(manifest['quantized_layers'])} linears")

    decoder = OnnxDecoder(result.onnx_path, manifest)
    assert {cache.data_type() for cache in decoder.caches.values()} == {"tensor(uint8)"}
    start = time.perf_counter()
    got = torch.stack([decoder([token], pos) for pos, token in enumerate(tokens)])
    per_token = (time.perf_counter() - start) / len(tokens)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    error = ((got - expected).norm() / expected.norm()).item()
    same = (got.argmax(-1) == expected.argmax(-1)).float().mean().item()
    print(f"Relative logit error {error:.3f}, same next-token prediction at {same:.0%} of {len(tokens)} positions "
          f"({per_token * 1e3:.0f} ms per token)")
    assert error < MAX_RELATIVE_LOGIT_ERROR, f"Relative logit error exceeds {MAX_RELATIVE_LOGIT_ERROR}: {error}"

    # The whole prompt in one call (the graph takes up to MAX_INPUT_LEN tokens), then greedy decoding.
    decoder.reset()
    completion = [int(decoder(prompt, 0).argmax())]
    while len(completion) < NEW_TOKENS and completion[-1] not in manifest["eos_token_id"]:
        completion.append(int(decoder([completion[-1]], len(prompt) + len(completion) - 1).argmax()))
    print(f"ONNX Runtime, 8-bit cache: {tokenizer.decode(completion, skip_special_tokens=True)!r}")
    print(f"Original float model: {tokenizer.decode(original_completion, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
