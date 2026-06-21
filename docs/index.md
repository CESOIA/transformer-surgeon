# TransformerSurgeon

**Compression-aware transformer models built on PyTorch and HuggingFace.**

Apply low-rank decomposition, pruning, or quantization to any supported model in
a few lines — then keep calling `model.generate()` as usual. Compression is
reversible by default.

---

## Quick Install

```bash
git clone https://github.com/CESOIA/transformer-surgeon.git
cd transformer-surgeon
pip install -e .
```

---

## What's Included

**Compression methods**

- `lrd` — low-rank decomposition (SVD, calibration-aware SVD-LLM-v2, AA-SVD)
- `structured_pruning` — output neuron removal by magnitude, gradient, or random
- `unstructured_pruning` — weight-level sparsity masks
- `quantization` — fixed-point and binary weights

**Core abstractions**

- `CompressionScheme` — per-layer compression config and application
- `CompressionSchemesManager` — model-level orchestration with flexible layer filtering

**Compression-ready building blocks**

- `LinearCompressed` — drop-in `nn.Linear` replacement with built-in LRD support
- `VCONBlock` — smooth transition between original and compressed modules during fine-tuning

**Export**

- `convert_for_export` — rewrite model to compact custom decoder/encoder graphs
- `export_to_hf` — push compressed model to HuggingFace Hub
- ExecuTorch export pipeline under `transformersurgeon.export`

---

## Supported Model Families

| Model | Type | Import |
|---|---|---|
| Qwen2 | Causal LM | `Qwen2ForCausalLMCompress` |
| Llama | Causal LM | `LlamaForCausalLMCompress` |
| Qwen2-VL | Vision-language | `Qwen2VLForConditionalGenerationCompress` |
| Qwen2.5-VL | Vision-language | `Qwen2_5_VLForConditionalGenerationCompress` |
| BERT | Encoder | `BertForSequenceClassificationCompress` |
| DistilBERT | Encoder | `DistilBertForSequenceClassificationCompress` |
| ViT | Vision encoder | `ViTForImageClassificationCompress` |

---

## Basic Example

```python
import torch
from transformers import AutoTokenizer
from transformersurgeon import Qwen2ForCausalLMCompress, Qwen2CompressionSchemesManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2ForCausalLMCompress.from_pretrained("Qwen/Qwen2-0.5B-Instruct", torch_dtype="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

manager = Qwen2CompressionSchemesManager(model)
manager.set("lrd", "rank", 640, criteria=[[0, "self_attn.q_proj"]])
manager.set("lrd", "rank", 640, criteria=[[1, "mlp.up_proj"]])
manager.apply(hard=False)   # reversible

inputs = tokenizer("Write a sentence about compression.", return_tensors="pt").to(device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0], skip_special_tokens=True))

manager.restore()  # undo all compression, back to original weights
```

---

## Where to Go Next

- [**Concepts**](concepts.md) — understand the mental model before diving into the API
- [**Examples**](examples/01_basic_lrd.md) — runnable scripts for common workflows
- [**API Reference**](api/utils/manager.md) — full method-level documentation
