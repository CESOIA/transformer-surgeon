"""Export Qwen2.5-0.5B, compressed (8-bit MLP weights), to QNN with an 8-bit KV cache, then check it on the host.

The export calibrates the fixed-point ranges (16-bit activations, 8-bit KV cache) on CALIBRATION_WINDOWS random
windows of CACHE tokens from WikiText-2 (the `datasets` package downloads it on first use). The script then checks
that every KV-cache write is 8-bit, and compares the fake-quantised model the .pte is lowered from
(QNNExportConfig.keep_reference_module) with the original float model, on text that is not in the calibration data.
About 13 minutes, most of it calibrating (fewer CALIBRATION_WINDOWS is faster), and 16 GiB of RAM (QNN SDK 2.37,
ExecuTorch 1.5.1). Run in the activated cesoia environment:

    python scripts/executorch/qnn/export_8bit_kv_cache.py [output.pte]
"""
import gc
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress
from transformersurgeon.calibration import wikitext2_loader
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.backends import QNNExportConfig

MODEL = "Qwen/Qwen2.5-0.5B"
CACHE = 128  # KV-cache slots: the longest prompt + answer the .pte can hold
CALIBRATION_WINDOWS = 32  # WikiText-2 windows of CACHE tokens that set the fixed-point ranges
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qwen25_qnn_w8_kv8.pte"
os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = Qwen2ForCausalLMCompress.from_pretrained(MODEL, dtype=torch.float32).eval()

# 1. Compress: 8-bit per-channel weights in every MLP linear (HTP runs them as 16a8w).
mlp = ["mlp"]
manager = Qwen2CompressionSchemesManager(model)
for name, value in {"precision": 8, "method": "vanilla", "granularity": "per_channel",
                    "sparsity": 0.0, "sparse_method": "magnitude", "eps": 1e-6}.items():
    manager.set("quantization", name, value, criteria=mlp)
manager.apply(hard=True, criteria=mlp)

# 2. Export: every activation in 16-bit fixed point (quant_scope="all"), the KV cache in 8-bit. HTP keeps static
# ranges, measured while the exporter decodes the calibration text token by token; windows of CACHE tokens fill
# every cache slot. Without calibration_token_ids the exporter would use a few built-in sentences.
calibration = wikitext2_loader(tokenizer, num_samples=CALIBRATION_WINDOWS, seq_len=CACHE)
config = QNNExportConfig(
    output_path=OUTPUT,
    soc_model="SM8850",
    max_seq_len=CACHE,
    quant_scope="all",
    float_attention_core=False,
    kv_cache_bits=8,
    calibration_token_ids=[sample["input_ids"][0].tolist() for sample in calibration],
    # The 8-bit cache needs a cache written by an index operation: "mutable" (default) or "io_scatter".
    # masked_softmax keeps the 16-bit attention-score ranges tight (AGENTS.md, QNN activation calibration).
    convert_options={"max_cache_len": CACHE, "cache_impl": "mutable", "masked_softmax": True},
    keep_reference_module=True,  # the fake-quantised torch model the .pte is lowered from
    run_weight_mismatch_check=False,
)
start = time.perf_counter()
result = export_to_backend(model, config)
print(f"\nexported {result.pte_path}: {os.path.getsize(OUTPUT) / 2**20:.0f} MiB "
      f"in {(time.perf_counter() - start) / 60:.1f} min")
del model, manager
gc.collect()

# 3. Every KV-cache write stores values dequantised from an 8-bit range.
reference = result.reference_module
writes = [n for n in reference.graph.nodes if n.op == "call_function" and "index_copy" in str(n.target)]
levels = [w.args[3].args[4] for w in writes
          if isinstance(w.args[3], torch.fx.Node) and "dequantize" in str(w.args[3].target)]
print(f"KV-cache writes: {len(writes)}, of which 8-bit: {sum(q <= 255 for q in levels)}")


# 4. On the host: the fake-quantised model against the original float model, on the same tokens.
def step(token, pos):
    out = reference(torch.tensor([token]), torch.tensor([pos]))
    return (out[0] if isinstance(out, (tuple, list)) else out).float().reshape(-1)


original = Qwen2ForCausalLMCompress.from_pretrained(MODEL, dtype=torch.float32).eval()
text = "The Eiffel Tower is in Paris, the capital of France. It was built for the World's Fair of 1889."
tokens = tokenizer(text).input_ids
with torch.no_grad():
    expected = original(torch.tensor([tokens])).logits[0]
    got = torch.stack([step(t, p) for p, t in enumerate(tokens)])
error = ((got - expected).norm() / expected.norm()).item()
same = (got.argmax(-1) == expected.argmax(-1)).float().mean().item()
print(f"relative logit error {error:.3f}; same next-token prediction at {same:.0%} of {len(tokens)} positions")

prompt = tokenizer("The capital of France is").input_ids
ids = list(prompt)
with torch.no_grad():
    for pos in range(len(prompt) + 19):
        logits = step(ids[pos], pos)
        if pos >= len(prompt) - 1:
            ids.append(int(logits.argmax()))
    greedy = original.generate(torch.tensor([prompt]), max_new_tokens=20, do_sample=False)[0]
print(f"compressed, 8-bit cache: {tokenizer.decode(ids[len(prompt):])!r}")
print(f"original float model   : {tokenizer.decode(greedy[len(prompt):])!r}")
