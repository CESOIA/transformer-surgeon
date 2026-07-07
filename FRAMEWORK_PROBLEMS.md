# transformer-surgeon — Framework Problems Report

Branch: `test-framework-hardening` (off `draft-pruning`)
Date: 2026-07-06
Method: static reading of the package + docs, plus a diagnostic sweep that actually
loads → compresses → exports every model family and every export backend. Diagnostic
harnesses live under `scratchpad/` and are distilled into the new pytest suite
(`test/unit/`, `test/e2e/`). Every finding below was reproduced, not inferred.

## Environment used for the sweep

| Component | Value |
|---|---|
| Python / torch | 3.12 / 2.12.0+cu132 |
| transformers | 5.12.0 |
| GPUs | 2 × CUDA (device 0/1) |
| executorch | installed |
| torch-tensorrt / tensorrt | 2.12.1 / 10.16.1 |
| QNN SDK | **not present** |
| Cached checkpoints | Qwen2-0.5B, Qwen2-0.5B-Instruct, Qwen2.5-0.5B |

Capability-gated backends behave as expected for this box: XNNPACK + TensorRT are
exercised for real; QNN is unavailable and is skipped by the suite (and fails hard
if forced — see P5).

---

## Severity summary

| # | Severity | Area | One-liner |
|---|---|---|---|
| P1 | **High** | BERT / manager | **Fixed.** `BertCompressionSchemesManager` could not be constructed at all — list-form `calibration_groups` was rejected by a parser that only understood the dict form. Also fixed in the same pass: the `no_cascade_calibration` enforcement in `apply_cascade` was dead code (an unpopulated `selected_by_block` dict), so BERT's cascade-calibration exclusion was never actually enforced. |
| P2 | Medium | quantization / docs | **Fixed (docs).** `AGENTS.md` wrongly documented `precision="int8"/"int4"/"int2"`; the validator was already correct (only `"full"`/`"binary"`/integer `8/4/2` are valid) — the doc table has been corrected, no code change needed. |
| P3 | Medium | structured pruning | Hard-pruning one half of a coupled gate/up pair silently produces a model that crashes on the next forward, with no validation error at `apply()`. |
| P4 | Medium | TensorRT export / docs | The documented one-liner `export_to_backend(model, config)` fails with a cryptic FakeTensor device error when the model is on CUDA; export only works with CPU-resident components. |
| P5 | Low–Med | QNN export | **Fixed.** `export_with_qnn` had no capability guard, so calling it on a machine without the full Qualcomm toolchain surfaced a bare `ImportError: install py-cpuinfo` deep inside the export pipeline. |
| P6 | Low | VL structured pruning | **Fixed.** `create_group()` had no guard against a shared-mask group spanning both the vision and text towers; it now rejects cross-tower groups immediately with a clear error. Structured MLP pruning on the text tower is verified working end-to-end (128→96 shrink, correct cascade, valid forward) — the earlier "still unsupported" conclusion was a bug in the test fixture's criteria, not the framework. |
| T* | — | test suite | ~~`pytest` is not a default dependency~~ **fixed** (`test`/`dev` extras + docs). Remaining: the existing suite did not collect (15–16 collection errors from non-pytest CLI scripts, duplicate module names, stale APIs, a `sys.modules` poisoning bug) — now isolated under `test/test_deprecated/` and excluded from default collection. |
| N1 | — | new model family | Added `models/modernbert_c/` (`ModernBertForSequenceClassificationCompress`) as a more reliable encoder family than BERT: no calibration-groups parser edge case, and its fused `Wqkv`/`Wi` projections sidestep the coupled-pruning pitfalls in P3/P6 by declaring structured pruning unsupported up front instead of silently breaking. Also marked `no_cascade_calibration: True` (see P1) — its per-layer-type rotary embeddings for alternating local/global attention aren't modeled by the single-flow cascade algorithm either. |

---

## P1 — BERT compression is completely broken (list-form `calibration_groups`) — FIXED

**Status:** Fixed. `_get_calibration_groups_from_indexing` (`transformersurgeon/utils/manager.py`)
now has an `isinstance(block_groups, list)` branch that treats each element as a
parallel group of fully-qualified layer names, matching the documented semantics
in `README.md` and `FRAMEWORK_STRUCTURE.md` §5.1. `BertCompressionSchemesManager(model)`
builds successfully and the full compression loop (LRD, quantization, unstructured
pruning, soft apply/restore) works on BERT. Regression tests:
`test/unit/test_known_bugs.py::test_bert_manager_builds`,
`test/e2e/test_model_families.py::test_build_manager[bert]` (and the other
per-family tests parametrized over `"bert"`).

**Cascade calibration is a separate, deliberate limitation, not part of this bug.**
BERT's indexing already declared `'no_cascade_calibration': True` — but the
enforcement in `apply_cascade` (`utils/cascade.py`) was dead code: it built a
`selected_by_block` dict that was never populated (a leftover from a refactor that
removed a `scheme.block_name` population loop after that attribute stopped
existing on `CompressionScheme`), so the `ValueError` guard could never fire.
Requesting `set_calibration_mode(mode="cascade")` on BERT silently ran cascade's
block-wise calibration instead of failing loudly. This has been fixed by
populating `selected_by_block` from `manager.schemes` (the same lookup the main
per-block loop already does) before the guard check. BERT is confirmed genuinely
incompatible with cascade mode — see "BERT and cascade calibration" below — so the
fix makes the existing `no_cascade_calibration` flag actually take effect instead
of changing BERT's behavior. Regression test:
`test/unit/test_cascade_calibration.py::test_bert_cascade_calibration_raises_clear_error`
(plus `test_qwen2_cascade_calibration_still_works` to confirm the guard fix didn't
disable cascade for families that support it).

### BERT and cascade calibration

Tested empirically (tiny random-weight BERT, `aa-svd` LRD, `set_calibration_mode("cascade")`):
before the guard fix above, cascade calibration ran to completion without
raising and produced a shape-correct (but not validated for correctness) output.
It isn't adopted as supported, though, because:
- `_collect_preprocessing_outputs`/`_collect_loader_inputs` in `utils/cascade.py`
  discard `attention_mask` after the embeddings step — only the raw hidden-state
  tensor is threaded between blocks. Bidirectional encoders like BERT rely on
  `attention_mask` far more heavily than causal decoders (padded batches, no
  causal mask to fall back on), so calibrating without it is not a safe stand-in
  for real inference.
- BERT's `calibration_groups` are declared as list-of-lists at the block level, and
  cascade staging requires each group's layers to be **consecutive** in the
  flattened `path_list` order (`_build_layer_stages_for_block`) — workable for
  BERT's single group (`query`/`key`/`value`), but brittle for any future grouping.

Given the framework structure cannot thread `attention_mask` through cascade's
block-isolated flow without changing `utils/cascade.py` itself, BERT is kept
**not compatible with cascade calibration** — `no_cascade_calibration: True`
stays set, and the guard above now actually enforces it. Use `"standard"`
calibration mode (`"svd"` or `"svd-llm-v2"` LRD) for BERT instead.

**Impact (before the fix):** `BertCompressionSchemesManager(model)` raised before any
compression could be configured, so the entire BERT family (advertised in
`README.md`) was unusable. The same would have hit any new model that followed the
*documented* list form of `calibration_groups` — which is exactly why ModernBERT
(added alongside this fix) could be indexed with the same list form from day one.

**Reproduce (historical — no longer raises):**
```python
from transformersurgeon import BertForSequenceClassificationCompress, BertConfigCompress, BertCompressionSchemesManager
m = BertForSequenceClassificationCompress(BertConfigCompress(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, vocab_size=256))
BertCompressionSchemesManager(m)
# TypeError: Indexing field 'calibration_groups' must be a list or a dict. Got <class 'list'>.
```

**Root cause:** `transformersurgeon/utils/manager.py:69-108`
(`_get_calibration_groups_from_indexing`). The parser has a branch for the **dict**
form only; when `block_groups` is a **list** it falls straight through to the
`raise TypeError(...)` at lines 105-108 — whose message ironically states that a
list is acceptable. BERT's indexing uses the list form
(`transformersurgeon/models/bert_c/indexing_bert_c.py:27`), which is the exact
format documented as valid in `README.md` (“Indexing Calibration Groups”) and
`FRAMEWORK_STRUCTURE.md` §5.1.

**Fix applied:** added the `isinstance(block_groups, list)` branch described above.

**Regression test:** `test/unit/test_known_bugs.py::test_bert_manager_builds` (now a
plain, non-xfail assertion), `test/e2e/test_model_families.py::test_build_manager[bert]`
(now passes, no longer xfailed).

---

## P2 — Quantization `precision` string values in the docs were rejected — FIXED (docs)

**Status:** Fixed by correcting `AGENTS.md`, not the validator. `validate_precision`
(`transformersurgeon/compression/quantization.py:386-400`) accepts only `"full"`,
`"binary"`, or an `int` in `[2,16]` — it never accepted `"int8"`/`"int4"`/`"int2"`
strings. The table in `AGENTS.md` (“`"quantization"`” section) was wrong and has
been corrected to `"full"`, `"binary"`, or `int` in `[2, 16]` (e.g. `8`, `4`, `2` —
explicitly **not** the strings `"int8"`/`"int4"`/`"int2"`). No code changed; the
validator's behavior was already correct, only the doc was misleading.

**Reproduce (still the real, intended behavior):**
```python
mgr.set("quantization", "precision", "int8", criteria="mlp.gate_proj")
mgr.apply(hard=False)
# ValueError: Precision must be an integer, 'full' or 'binary', but got 'int8'.  <- correct, by design
mgr.set("quantization", "precision", 8, ...)   # <- this is the real API
```

**Regression tests:** `test/unit/test_known_bugs.py::test_quantization_rejects_int8_string_with_clear_error`
(pins that the string form keeps raising a clear error) and
`::test_quantization_precision_int_is_the_real_api` (pins the working integer form).

---

## P3 — Partial coupled MLP pruning silently yields a broken model

**Impact:** Hard-pruning only `mlp.gate_proj` (leaving its coupled partner
`mlp.up_proj` at full width) succeeds without warning, then the model crashes on the
next forward pass:
`RuntimeError: The size of tensor a (96) must match the size of tensor b (128)`.
A user who prunes one layer of a coupled pair gets no signal until inference.

**Reproduce:** `test/unit/test_known_bugs.py::test_partial_coupled_gate_prune_is_guarded`.

**Root cause:** gated MLPs compute `act(gate(x)) * up(x)`; `gate_proj` and `up_proj`
must keep identical output widths. `apply(hard=True)` resizes `gate_proj` and does not
validate that co-masked partners were pruned consistently. The framework already
knows the coupling (`pruning.coupled_masks` in indexing) but does not enforce it at
apply time.

**Fix direction:** at hard-apply, validate that every member of a `coupled_masks`
group is being pruned with a compatible mask (or auto-extend the mask to partners),
and raise a clear error otherwise. The correct multi-layer path (shared mask via
`auto_groups()` + `share_mask` + `reduce_op`) works and is covered by
`test/e2e/test_model_families.py::test_structured_prune_hard_mlp`.

---

## P4 — Documented TensorRT one-liner fails for a GPU-resident model [FIXED]

**Fix applied:** `export_to_backend` (`transformersurgeon/export/export.py`) and
`resolve_components_and_wrapper` (`transformersurgeon/export/common.py`) now
call a shared `_normalize_component_devices()` helper that moves
`embedding`/`decoder`/`final_layer` onto CPU before tracing, regardless of
the device the caller's model/components started on. `config.device` is
still what `_compile_tensorrt` uses afterward to place the compiled TensorRT
engine. Regression test:
`test/e2e/test_export_pipelines.py::test_export_tensorrt_cuda_resident_model`
(passes a `.to("cuda")` model straight to `export_to_backend`, exactly the
documented one-liner — previously fatal, now passes). `README.md`/`AGENTS.md`
updated to state the device-handling contract explicitly.

**Impact (historical):** `README.md` (“Export to a deployment backend”) and `AGENTS.md`
(“Export Backends”) show:
```python
config = TensorRTExportConfig(output_path="model.ep", backend="tensorrt", device="cuda:0")
result = export_to_backend(model, config=config)
```
On a CUDA box the natural thing is to have `model` on the GPU. Doing so fails with:
`RuntimeError: Unhandled FakeTensor Device Propagation for aten.embedding.default,
found two different devices cuda:0, cpu` during `torch.export`.

**What works (confirmed):** keep the model/components on **CPU** and pass the component
dict; the exporter places the graph on `device` itself:
```python
model = ModelCls.from_pretrained(name, torch_dtype=torch.float16)   # stays on CPU
comps = {"embedding": ..., "decoder": convert_for_export(model)["text"],
         "final_layer": model.lm_head, "config": model.config}
export_to_backend(comps, config=TensorRTExportConfig(..., device="cuda:0"))
```
This is exactly what `test/tensorrt_tests/exporter_function/tensorrt/test_pretrained_quant_export.py`
does — but the top-level docs never say the model must not be pre-moved to CUDA.

**Fix direction:** either normalize component devices inside `export_to_backend`
(move to CPU for tracing, then to `config.device`), or document the CPU-resident
requirement prominently in `README.md`/`AGENTS.md`.

**Test:** `test/e2e/test_export_pipelines.py::test_export_tensorrt` (uses the working
CPU-components pattern; gated by `requires_tensorrt`).

---

## P5 — QNN export import crashes instead of degrading gracefully [FIXED]

**Fix applied:** `transformersurgeon/export/executorch_exporters/qnn/qnn_export.py`
now exposes `is_qnn_available()`, a capability probe that checks the cheap
preconditions (`executorch` and `cpuinfo` resolvable via `importlib.util.find_spec`,
plus `QNN_SDK_ROOT`/`QUALCOMM_SDK_ROOT` set) **without importing**
`executorch.backends.qualcomm` — `find_spec` locates a module without executing its
`__init__.py`, which is exactly what was triggering the undeclared `py-cpuinfo`
import. `export_with_qnn()` checks `is_qnn_available()` up front and raises a clear
`RuntimeError` ("QNN backend unavailable: install the Qualcomm ExecuTorch
toolchain...") instead of proceeding; the real Qualcomm import later in the function
is also wrapped in a `try/except ImportError` that re-raises the same clear message,
as a fallback in case the cheap probe passes but the toolchain still isn't fully
usable. `is_qnn_available` is re-exported from
`transformersurgeon.export.executorch_exporters.qnn` so callers can check
availability ahead of time, mirroring the test suite's `_helpers/capabilities.HAS_QNN`.
Regression test:
`test/unit/test_known_bugs.py::test_qnn_unavailable_raises_clear_error_instead_of_cpuinfo_importerror`.

**Impact (historical):** On any machine without the full Qualcomm ExecuTorch
toolchain, calling `export_with_qnn` (e.g. via `export_to_backend(model, config)`
with `backend="qnn"`) surfaced:
```python
from executorch.backends.qualcomm.utils.utils import generate_htp_compiler_spec
# ImportError: Please install the cpuinfo with pip install py-cpuinfo.
```
The failure was a missing transitive dependency (`py-cpuinfo`, pulled in by
`executorch.backends.qualcomm/__init__.py`), surfaced long before any “QNN SDK not
found” check, with no capability guard and no way to probe availability without
paying the crash.

---

## P6 — VL structured pruning mixes vision and text towers [FIXED]

**Fix applied:** `create_group()` (`transformersurgeon/utils/manager.py`) now resolves
which indexing block/tower (e.g. `"vision"` vs `"text"`) each member scheme belongs to
(via a new `_scheme_block_name()` helper) and raises a clear `ValueError` up front if a
group would span more than one tower, instead of allowing the group to be built and
only failing later — deep inside `compression/coupled_pruning.py:70` with a confusing
`ValueError: Coupled pruning mask length ... does not match the input features ... of
LinearCompressed(...)` — once `apply()` tries to share one mask across towers with
different hidden sizes.

`auto_groups()` itself already only ever calls `create_group()` with members from a
single tower's `pruning.coupled_masks`/`coupled_masks_all` (it iterates
`self.indexing.items()` one tower at a time), so the new guard is a no-op on the
existing automatic path — it protects the actually-risky path: manually-assembled
groups (`mgr.create_group([...])`) or any future indexing that isn't perfectly
tower-partitioned, both of which previously had no structural safeguard since
`_find_scheme()` searches every tower's scheme dict by full path with no scoping check.

Structured MLP pruning on the **text tower** of VL families (Qwen2-VL / Qwen2.5-VL)
now works end-to-end, verified empirically (tiny model, `ratio=0.25` on
`mlp.gate_proj`/`mlp.up_proj` with a shared mask via `auto_groups()`): both
projections shrink `128 -> 96`, the coupled cascade correctly resizes
`mlp.down_proj`'s input to `96`, and the forward pass produces a valid,
correctly-shaped output. The earlier "unsupported" conclusion in this report was
itself wrong — it was caused by a bug in the *test fixture*
(`test/_helpers/model_factory.py`), not the framework: the criteria
`["language_model", "mlp.gate_proj"]` looks like an AND (block/tower name plus
layer name) but the criteria grammar treats a bare multi-item list as **OR across
items**, not AND (AND requires the extra nesting `[["language_model",
"mlp.gate_proj"]]`) — so the fixture was accidentally setting `ratio=0.25` on
*every* language-model layer (attention projections, layernorms, ...), which is
what actually produced the original mask-length mismatch. Fixed by using plain
substrings (`"mlp.gate_proj"`/`"mlp.up_proj"`, unambiguous since the vision tower
uses `fc1`/`fc2` naming, not `gate_proj`/`up_proj`).

Note in passing: `pruning_supported: []`, declared in both `vision`/`text`
sub-dicts of `models/qwen2_vl_c/indexing_qwen2_vl_c.py` (and the 2.5 variant) and
in `models/modernbert_c/indexing_modernbert_c.py`, is dead metadata — grepping the
package shows nothing ever reads it. It doesn't gate or block anything today; if
it was meant to, that wiring is missing.

**Regression test:** `test/unit/test_known_bugs.py::test_cross_tower_group_is_rejected`
(cross-tower `create_group()` call now raises `ValueError` mentioning both towers) and
`::test_auto_groups_still_works_per_tower` (confirms `auto_groups()` still produces only
single-tower groups post-fix). `test/e2e/test_model_families.py::test_structured_prune_hard_mlp`
now runs for real (no longer skipped) for both `qwen2_vl` and `qwen2_5_vl`.

---

## T* — Test-suite structural problems (motivating the refactor)

- **`pytest` is not installed by default.** ~~It lives only in the `dev` extra
  (`setup.py`), so `pytest ...` — the command every doc tells you to run — fails with
  `No module named pytest` on a plain `pip install -e .`.~~ **Fixed:** `setup.py`
  now also exposes a `test` extra (`pytest`, identical to `dev`), and every Quick
  Install block (`README.md`, `docs/index.md`) installs `.[test]` so following the
  docs verbatim leaves you with a working `pytest`. `pytest` is deliberately kept
  out of `install_requires` — it's a test tool, not something downstream consumers
  of this package as a library should be forced to install.
- **The existing suite does not collect.** `python -m pytest` at the repo root
  produces **15 collection errors** (16 under `--import-mode=importlib`) and cannot
  run as a suite. Causes:
  - Many “tests” are `argparse` + `main()` CLI scripts; their `test_*` functions take
    an `args` positional (e.g. `test/compression_tests/compression_test.py`,
    `test/bert_tests/inference_test.py`), so pytest can’t call them.
  - Duplicate module basenames without package `__init__.py`
    (`inference_test.py` in three dirs, `test_messages.py` in three dirs) →
    `import file mismatch`.
  - Local-helper imports that only work as scripts (`_common`, `qwen_vl_utils`,
    `sys.path.append("../test_data")`).
  - One or more legacy modules leave `transformersurgeon` **partially imported** in
    `sys.modules`, cascading `ImportError: cannot import name ... (unknown location)`
    into unrelated modules — a test-isolation defect.
- **Stale APIs in committed tests.** e.g. `manager.set("pruning", ...)` /
  `manager.set("pruning", "mode", ...)` in `test/bert_tests/inference_test.py`
  (the registry has `structured_pruning` / `unstructured_pruning`, not `pruning`).
- **CI-hostile defaults.** `test/hf_export_tests/test_hf_roundtrip.py` hardcodes
  `MODEL_TYPE = "qwen2_5_vl_c"` → downloads **Qwen2.5-VL-3B** and prunes block
  indices 26/27 that only exist in the full model. There is no small default.
- **No shared infrastructure:** no `conftest.py`, no `pytest.ini`, no markers, no
  capability-based skips, inconsistent `test_*.py` vs `*_test.py` naming.
- **Stale editable install:** the installed editable metadata is `0.7.3` while
  `setup.py` declares `0.8.0` (`pip install -e .` was not re-run after a bump).

---

## What the new suite adds

```
test/
  pytest.ini                 # markers (unit/e2e/slow/gpu/download/executorch/tensorrt/qnn), importlib mode
  conftest.py                # _helpers on sys.path, capability banner, device/out_dir fixtures
  _helpers/
    capabilities.py          # HAS_CUDA / HAS_EXECUTORCH / HAS_TENSORRT / HAS_QNN + requires_* skip marks
    model_factory.py         # tiny random-weight builder for all 8 families (no downloads)
  unit/
    test_known_bugs.py         # P2/P3 pinned as xfail regressions (XPASS = fixed); P1 now a plain passing assertion
    test_cascade_calibration.py # no_cascade_calibration guard: BERT/ModernBERT refuse cascade mode, Qwen2 cascade still works
  e2e/
    test_model_families.py   # load→compress(LRD/quant/prune)→restore→convert, parametrized over 8 families
    test_export_pipelines.py # HF / convert / XNNPACK / TensorRT / QNN, capability-gated
```

Current results on this box:

- `pytest test/unit test/e2e/test_model_families.py` → **52 passed, 3 skipped, 2 xfailed** in ~6 s, CPU-only, no network.
- `pytest test/e2e/test_export_pipelines.py` → **4 passed, 1 skipped** in ~3.5 min:
  HF roundtrip + convert + XNNPACK (`.pte`) + TensorRT (`.pt2` engine) exercised for
  real on Qwen2-0.5B; QNN skipped (no SDK on this box).

The tiny-model factory is the key idea: every family’s framework-specific path
(layer replacement, manager construction, apply/restore, convert) is exercised from
a small config in well under a second, so the model-matrix runs anywhere. Real
checkpoints are only used where the export lowering genuinely needs real weights, and
those tests are marked `download`/`slow` and skip when a backend is unavailable.

## How to run

```bash
pip install -e ".[dev]"                       # brings in pytest
pytest test/unit                              # fast pure-logic + bug regressions
pytest test/e2e/test_model_families.py        # all 7 families, no downloads
pytest test/e2e/test_export_pipelines.py      # export backends (downloads Qwen2-0.5B)
pytest -m "not slow and not download"         # everything cheap
```
