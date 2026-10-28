# How to add…

Every addition stays inside one folder (plus a new atom in `blocks/` when an operation does not exist yet).
These recipes were checked on the code; the model-family one by actually adding Mistral.

## A model family

Only `models/` changes.

1. Copy the closest family folder, e.g. `models/llama_c/` → `models/mistral_c/`.
2. In `indexing_mistral_c.py` (model dictionary) adjust paths, couplings and export keys if the layout differs.
   For an architecture identical to an existing one, only the names change.
3. In `family_mistral_c.py` (family dictionary) point `hf_config` / `hf_model` to the Hugging Face classes and
   rename `class_names`.
4. Add the folder to `models/__init__.py::FAMILIES`.

```python
import transformersurgeon as T
model = T.MistralForCausalLMCompress(T.MistralConfigCompress(...))
manager = T.MistralCompressionSchemesManager(model)
```

For the tests, add a tiny builder to `test/_helpers/model_factory.py::FAMILIES`.

## A compression method variant

Only `compression/` changes. Example: a new low-rank decomposition.

1. New file `compression/lrd_methods/my_svd.py` with a function of the same signature as `low_rank_svd`.
2. One entry in `compression/lrd_methods/__init__.py::METHOD_FUNCTIONS` (`VALID_METHODS` derives from it).
3. If it needs statistics, list them in `compression/lrd.py::CALIBRATED_METHOD_DICT`.

```python
manager.set("lrd", "method", "my-svd", criteria="mlp")
```

The same applies to `quantization_methods/` and `unstructured_pruning_methods/`.

## A compression type

1. Subclass `compression/abstract.py::Compressor` (`set_calibration_store`, `needs_calibration`, `apply`,
   `restore`, `_to_compress`, `__repr__`; optionally `prepare_for_save`, `check_coupling`).
2. Register it in `compression/registry.py`: `COMPRESSOR_DICT`, `COMPRESSION_REGISTRY` (options with defaults
   and validators from `compression/validators.py`) and `APPLICATION_ORDER`.

## A calibration statistic

Only `calibration/` changes.

1. Subclass `calibration/summaries/base.py::CalibrationSummary` (`name`, `required_raw_data`,
   `update_from_raw`, optionally `update_runtime`).
2. Register an instance in `calibration/summaries/registry.py::SUMMARY_REGISTRY`.
3. If it needs a new raw stream, add a collector in `calibration/raw_data/` and its registry.

The compressor that uses it names it in `needs_calibration()`.

## An atom

1. One class per file in the right `blocks/` sub-folder (e.g. `blocks/norm/my_norm.py`).
2. One entry in `blocks/catalog.py::BLOCK_CATALOG` with `implements` and `selected_by`.
3. If the export chooses it, add it to the backends that support it in `export/operations.py::BACKEND_OPERATIONS`.

## A backend

Only `export/` changes.

1. In `export/backends/`, subclass `Exporter` with `backend`, `config_class` (a `BackendExportConfig`
   subclass) and `lower(components, config)`; ExecuTorch backends can start from
   `executorch_base.py::ExecuTorchExporter`.
2. Decorate it with `@register_exporter` and import it in `export/backends/__init__.py`.
3. Add the backend to every entry of `BACKEND_OPERATIONS`.
4. Add a capability-gated test in `test/export/test_exporter.py`.
