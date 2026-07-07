# 🩺 transformer-surgeon

**Compression-aware transformer models built on PyTorch and Hugging Face.**

Apply low-rank decomposition, pruning, or quantization to any supported model in a few lines — then keep calling `model.generate()` as usual. Compression is reversible by default.

## ⚡ Quick Install

```bash
git clone https://github.com/CESOIA/transformer-surgeon.git
cd transformer-surgeon
pip install -e ".[test]"   # add [test] to also get pytest for running the test suite
```

## 🗂️ What's Included

**Compression methods**
- `lrd` — low-rank decomposition (SVD, calibration-aware SVD-LLM-v2, AA-SVD)
- `structured_pruning` — output neuron removal (magnitude/gradient/random), with per-head `granularity`, shared masks across coupled layers (`auto_groups`), and hard removal that cascades into the next layer's inputs
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
- `export_to_hf` — push compressed model to Hugging Face Hub
- `export_to_backend` — export to ExecuTorch (`xnnpack`, `qnn`) or `tensorrt` under `transformersurgeon.export`, with mixed-precision export driven by per-layer compression metadata

## 🤖 Supported Model Families

| Model | Type | Import |
|---|---|---|
| Qwen2 | Causal LM | `Qwen2ForCausalLMCompress` |
| Llama | Causal LM | `LlamaForCausalLMCompress` |
| Qwen2-VL | Vision-language | `Qwen2VLForConditionalGenerationCompress` |
| Qwen2.5-VL | Vision-language | `Qwen2_5_VLForConditionalGenerationCompress` |
| BERT | Encoder | `BertForSequenceClassificationCompress` |
| ModernBERT | Encoder | `ModernBertForSequenceClassificationCompress` |
| DistilBERT | Encoder | `DistilBertForSequenceClassificationCompress` |
| ViT | Vision encoder | `ViTForImageClassificationCompress` |

All follow the same three-class pattern: a compressed config, a compressed model, and a compression manager — all HuggingFace-compatible.

## 🚀 Usage Examples

### Basic: low-rank decomposition on a causal LM

```python
import torch
from transformers import AutoTokenizer
from transformersurgeon import Qwen2ForCausalLMCompress, Qwen2CompressionSchemesManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2ForCausalLMCompress.from_pretrained("Qwen/Qwen2-0.5B-Instruct", torch_dtype="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

manager = Qwen2CompressionSchemesManager(model)
manager.set("lrd", "rank", 640, criteria=[[0, "self_attn.q_proj"]])  # block 0, q_proj
manager.set("lrd", "rank", 640, criteria=[[1, "mlp.up_proj"]])       # block 1, up_proj
manager.apply(hard=False)   # reversible

inputs = tokenizer("Write a sentence about compression.", return_tensors="pt").to(device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0], skip_special_tokens=True))

manager.restore()  # undo all compression, back to original weights
```

### Calibration-aware LRD (SVD-LLM-v2)

```python
manager = Qwen2CompressionSchemesManager(model)
manager.set("lrd", "method", "svd-llm-v2")
manager.set("lrd", "rank", 128, criteria="mlp.up_proj")

manager.set_calibration_data(calibration_loader)  # any torch DataLoader
manager.apply(device=device)
```

### Cascade calibration with indexing-defined groups (AA-SVD)

`set_calibration_mode(mode="cascade")` uses model indexing metadata (`calibration_groups`) to define layer groups that can be calibrated in parallel.

```python
manager = Qwen2CompressionSchemesManager(model)
manager.set("lrd", "method", "aa-svd")
manager.set("lrd", "rank", 128, criteria="all")

manager.set_calibration_mode(mode="cascade")
manager.set_calibration_data(calibration_loader)
manager.apply(device=device, verbose=True)
```

With `verbose=True`, cascade calibration prints shifted-activation sanity diagnostics for shifted summaries:
- `pairs`: number of paired `activation` / `activation_shifted` samples
- `mean_rel_l2_diff`: mean relative $\ell_2$ difference between base and shifted activations
- `max_rel_l2_diff`: maximum relative $\ell_2$ difference observed in the stage

**Not every family supports cascade mode.** An indexing block can set
`'no_cascade_calibration': True` to make `apply(..., mode="cascade")` raise
immediately instead of running. BERT and ModernBERT are marked this way today
(bidirectional/grouped-QKV layouts and, for ModernBERT, per-layer-type rotary
embeddings that the single-flow cascade position-embedding injection doesn't
model) — use `set_calibration_mode(mode="standard")` for those families.

### Structured pruning with grouping (`auto_groups`)

Remove output neurons from layers, keeping coupled layers consistent. `auto_groups()`
reads the model's indexing (`pruning.coupled_masks` / `coupled_masks_all`) and builds
the groups of layers that must share a pruning mask (e.g. each block's `gate_proj`/`up_proj`,
`q_proj`/`k_proj`). `share_mask` makes a group prune the *same* neurons; hard pruning then
removes them and cascades the input pruning into the next layer (`down_proj`).

```python
manager = Qwen2CompressionSchemesManager(model)

# 1. Build coupled-mask groups from indexing, and share one mask per MLP group.
groups = manager.auto_groups()                       # {"group1": [...gate_proj, up_proj], ...}
mlp_groups = {g: p for g, p in groups.items() if any("gate_proj" in x for x in p)}
for g in mlp_groups:
    manager.set("structured_pruning", "share_mask", True, group=g)   # group-only option
    manager.set("structured_pruning", "reduce_op", "add",  group=g)  # how to reduce scores

# 2. Configure pruning (random needs no calibration; "gradient" needs a loss callback).
manager.set("structured_pruning", "method", "random", criteria=None)
manager.set("structured_pruning", "ratio", 0.1, criteria="mlp.gate_proj")
manager.set("structured_pruning", "ratio", 0.1, criteria="mlp.up_proj")

# 3. Hard apply: removes neurons, resizes matrices, cascades down_proj's input.
manager.apply(hard=True)
print(model.model.layers[0].mlp.gate_proj.out_features)  # shrunk by ~10%
```

Notes:
- **Group-only options** (`share_mask`) must be set through `group=`, never `criteria`;
  enabling one resets that scheme's non-group config. `reduce_op` is a regular option
  (also used per-layer by `repeated_pattern`), settable via `criteria` or `group=`.
- `granularity=<int>` prunes uniformly within each chunk (e.g. per attention head);
  `repeated_pattern=True` lets layers with different chunk counts (GQA `q_proj`/`k_proj`) share one mask.
- Only MLP pruning is wired through convert/export today; attention pruning (GQA `head_dim`) is deferred.
- See `scripts/compression/prune_qwen.py` (gradient pruning) and
  `scripts/compression/prune_quant_export_tensorrt.py` (prune + quant → TensorRT).

### Convert to export-friendly graph, then compress

```python
from transformersurgeon import convert_for_export
from transformersurgeon.utils import CompressionSchemesManager

converted = convert_for_export(model, options={"use_sdpa": False})
decoder = converted["text"].to(device).eval()

manager = CompressionSchemesManager(decoder, decoder.indexing)
manager.set("lrd", "rank", 256, criteria=[[0, "attn.q_proj"]])
manager.apply(hard=False)
```

### Export to a deployment backend (ExecuTorch / TensorRT)

`export_to_backend` lowers a model (compressed or not) to `xnnpack`/`qnn` (ExecuTorch `.pte`) or `tensorrt`, driving mixed-precision export entirely from each layer's compression metadata:

```python
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.tensorrt import TensorRTExportConfig

config = TensorRTExportConfig(output_path="model.ep", backend="tensorrt", device="cuda:0")
result = export_to_backend(model, config=config)
print(result.engine_path)
```

Device placement is handled internally — `model` can live on CPU or CUDA; components are traced on CPU and the compiled engine is placed on `config.device` automatically.

> TensorRT requires `pip install -e ".[tensorrt]"` and a CUDA device. See `scripts/tensorrt/run_export.sh` for a CLI runner and `test/e2e/test_export_pipelines.py` for end-to-end examples covering `xnnpack`/`tensorrt`/`qnn`.

## 🎯 Filtering Layers with `criteria`

The `criteria` argument selects which layers to target. It supports strings, block IDs, and nested logic:

```python
manager.set("lrd", "rank", 64, criteria="mlp")           # all layers with "mlp" in the path
manager.set("lrd", "rank", 32, criteria=2)               # block 2 only
manager.set("lrd", "rank", 16, criteria=[["mlp", 5]])    # block 5 AND "mlp"  (AND)
manager.set("lrd", "rank", 16, criteria=["q_proj", 3])   # "q_proj" OR block 3 (OR)
```

## 🔀 VCON — Smooth Compression Transitions

`VCONBlock` runs original and compressed modules in parallel during fine-tuning, blending their outputs with a scalar `beta`:

- `beta=1.0` → original output only
- `beta=0.0` → compressed output only
- Intermediate values → smooth interpolation

```python
manager.init_vcon(criteria="mlp")           # wrap target layers in VCONBlock
manager.apply(hard=False, criteria="mlp")   # compress only the secondary block
manager.set_vcon_beta(0.5, criteria="mlp")  # blend at 50%
# fine-tune, gradually lower beta → 0, then:
manager.cancel_vcon(keep_block_b=True)      # collapse to compressed module
```

## 🔧 Soft vs Hard Apply

| | `hard=False` (default) | `hard=True` |
|---|---|---|
| Reversible | ✅ `manager.restore()` works | ❌ |
| Use case | Exploration, iterative fine-tuning | Final deployment |

## 📋 Manager API

| Method | Description |
|---|---|
| `set(compression, prop, value, criteria, group)` | Configure compression on filtered layers (or a whole `group`) |
| `apply(hard, criteria, ...)` | Apply all configured schemes |
| `restore(topology, criteria)` | Undo compression |
| `auto_groups()` | Build coupled-mask pruning groups from indexing; returns `{name: [paths]}` |
| `create_group(schemes, name)` / `delete_group(name)` | Manage shared-mask groups manually |
| `init_vcon(criteria)` | Wrap target layers with `VCONBlock` |
| `set_vcon_beta(beta, criteria)` | Set blending weight |
| `cancel_vcon(keep_block_b, criteria)` | Collapse `VCONBlock` to one branch |
| `set_calibration_data(dataloader)` | Attach calibration `DataLoader` |
| `set_calibration_mode(mode)` | Select `standard` or `cascade` calibration scheduling |
| `run_calibration(criteria, ...)` | Run calibration pass |
| `iter_filtered(criteria)` | Iterate over matching schemes |

## 🧩 Indexing Calibration Groups

For cascade mode, each model indexing block can declare explicit parallel calibration groups:

```python
"calibration_groups": [
	["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
	["mlp.gate_proj", "mlp.up_proj"],
]
```

These groups are consumed by `CompressionSchemesManager` exactly like previous user-provided group criteria, but now live in model indexing to keep calibration scheduling model-specific and reproducible.

## 📂 References

- [test/e2e/test_model_families.py](test/e2e/test_model_families.py) — manager basics, LRD/quant/pruning, convert-then-compress, all 7 families
- [test/e2e/test_export_pipelines.py](test/e2e/test_export_pipelines.py) — HF roundtrip, XNNPACK, TensorRT, QNN export (capability-gated)
- [test/unit/test_known_bugs.py](test/unit/test_known_bugs.py) — pinned regressions for open framework bugs
- [FRAMEWORK_STRUCTURE.md](FRAMEWORK_STRUCTURE.md) — package internals and extension guide
- [FRAMEWORK_PROBLEMS.md](FRAMEWORK_PROBLEMS.md) — known issues and test-suite hardening notes

## License

MIT · Maintainer: Luciano Prono · Politecnico di Torino & KAUST
