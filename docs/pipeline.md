# The compression → export pipeline

One pipeline serves every supported model and backend. Each step belongs to one folder.

```
 models/            compression/           calibration/       export/                     blocks/
 ───────            ────────────           ────────────       ───────                     ───────
 family dict ──► generated classes
 model dict  ──► manager + schemes ◄────── statistics
                 set() / apply()
                 (compressed HF model) ─────────────────────► convert_for_export ◄────── atoms chosen by
                                                              │                          the model dict and
                                                              │                          BACKEND_OPERATIONS
                                                              ▼
                                                     Exporter.export → backend lower() → artifact + manifest
```

1. **Model classes** (`models/`). `build_family` generates the configuration, model and manager classes from the
   family dictionary. At init the model replaces the layers listed in the model dictionary with compressible
   ones (`LinearCompressed`, ...).
2. **Configuration** (`compression/`). `manager.set(type, option, value, criteria=...)` records options on the
   matching schemes; nothing touches the model yet. `manager.auto_groups()` builds shared-mask groups.
3. **Calibration** (`calibration/`, optional). The compressors declare which statistics they need; the
   calibration collects them over a data loader.
4. **Apply** (`compression/`). `manager.apply(hard=False)` simulates the compression reversibly;
   `hard=True` makes it permanent before export.
5. **Conversion** (`export/convert.py` + `blocks/`). For every component of the model dictionary with a
   `structure`, a `blocks/` stack is built from the atoms the dictionary names and the implementations the
   export options choose (completed from `BACKEND_OPERATIONS`). Shapes, ranks, biases and quantisation state are
   copied from the compressed layers.
6. **Export** (`export/`). The exporter wraps embedding, decoder and output head, reads the per-layer
   quantisation, traces with `torch.export`, quantises, lowers with the backend toolchain, and writes the
   artifact and its manifest.

```python
import torch
from transformersurgeon import LlamaForCausalLMCompress, LlamaCompressionSchemesManager
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.backends import XNNPACKExportConfig

model = LlamaForCausalLMCompress.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float32)
manager = LlamaCompressionSchemesManager(model)
manager.set("lrd", "rank", 512, criteria="mlp")
manager.set("quantization", "precision", 8, criteria="mlp")
manager.set("quantization", "granularity", "per_channel", criteria="mlp")
manager.apply(hard=True)

result = export_to_backend(model, XNNPACKExportConfig(output_path="out/llama.pte", max_seq_len=1024))
```

Saving for Hugging Face instead of a device: `export_to_hf(model, repo_id, manager=manager)` (`hf/`).
