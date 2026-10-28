# `utils/` — generic helpers

`utils/` holds only functions that work on any PyTorch module. It never mentions a model, a compression
type or the export.

| Module | Contents |
|---|---|
| `modules.py` | `get_submodule`, `set_submodule`, `get_parent_module` (dotted paths such as `model.layers.3.mlp`), `replace_parameter` |
| `indexing.py` | generic readers of a model dictionary: `flatten_index_paths`, `get_block_config`, `num_blocks` |
| `batches.py` | `normalize_batch`, `pack_batch`: calibration batches as `(args, kwargs)` |
| `devices.py` | device helpers |
| `progress.py` | progress bars |

```python
from transformersurgeon.utils import get_submodule

q_proj = get_submodule(model, "model.layers.0.self_attn.q_proj")
```

API: [utils](../api/utils.md).
