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
pip install -e ".[dev]"    # library + test dependencies
pip install mkdocs mkdocs-material "mkdocstrings[python]"   # to build this site: mkdocs serve
```

---

## What's Included

**One pipeline, compression → export**, for many transformer families and four device backends.

| Step | Folder | What it offers |
|---|---|---|
| Describe the model | [`models/`](folders/models.md) | one folder per family: a model dictionary and a family dictionary |
| Compress | [`compression/`](folders/compression.md) | low-rank decomposition (SVD, SVD-LLM-v2, AA-SVD), structured / block / unstructured pruning, quantization, SmoothQuant; reversible (`hard=False`) or permanent |
| Calibrate | [`calibration/`](folders/calibration.md) | covariance, activation ranges, gradients; standard or block-by-block (cascade) |
| Build the export graph | [`blocks/`](folders/blocks.md) | a simplified transformer made of selectable atoms, listed in `BLOCK_CATALOG` |
| Export | [`export/`](folders/export.md) | `export_to_backend` for XNNPACK, QNN, ONNX, TensorRT; choices per backend in `BACKEND_OPERATIONS`; `export_to_hf` for Hugging Face |

---

## Supported Model Families

| Model | Type | Import |
|---|---|---|
| Qwen2 | Causal LM | `Qwen2ForCausalLMCompress` |
| Llama | Causal LM | `LlamaForCausalLMCompress` |
| Mistral | Causal LM | `MistralForCausalLMCompress` |
| Qwen2-VL | Vision-language | `Qwen2VLForConditionalGenerationCompress` |
| Qwen2.5-VL | Vision-language | `Qwen2_5_VLForConditionalGenerationCompress` |
| BERT | Encoder | `BertForSequenceClassificationCompress` |
| ModernBERT | Encoder | `ModernBertForSequenceClassificationCompress` |
| DistilBERT | Encoder | `DistilBertForSequenceClassificationCompress` |
| ViT | Vision encoder | `ViTForImageClassificationCompress` |
| DINOv2 | Vision encoder | `Dinov2ForImageClassificationCompress` |

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

- [**Concepts**](concepts.md) — the folders, the two dictionaries and the three-layer compression model
- [**Pipeline**](pipeline.md) — the compression → export flow, step by step
- [**Folders**](folders/models.md) — one page per package folder, with the dictionaries' tables
- [**How to add…**](recipes.md) — a model family, a method, a statistic, an atom, a backend
- [**Compression Methods**](compression_methods.md) — every compression type, method and parameter
- [**Examples**](examples/01_basic_lrd.md) — runnable scripts for common workflows
- [**API Reference**](api/compression.md) — generated from the docstrings
