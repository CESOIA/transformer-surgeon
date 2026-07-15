# Compression Methods Reference

A plain-language, complete reference for every compression type, method, and
parameter in TransformerSurgeon. If you just want the terse table for quick
lookup while coding, see [AGENTS.md](https://github.com/CESOIA/transformer-surgeon/blob/main/AGENTS.md#compression-parameter-reference) —
this page explains *what each option actually does* and *why you'd pick it*.

All parameters are set the same way, regardless of type:

```python
manager.set(<compression_type>, <param>, <value>, criteria=<criteria>)
```

The four `compression_type` strings are `"lrd"`, `"structured_pruning"`,
`"unstructured_pruning"`, and `"quantization"`. Every scheme can combine all
four at once — they're applied in this fixed order (`lrd` reshapes the
weight, so it must run first; `quantization` must see the final float
weights, so it runs last):

```
lrd  →  structured_pruning  →  unstructured_pruning  →  quantization
```

---

## 1. `"lrd"` — Low-Rank Decomposition

**What it does:** Replaces a layer's weight matrix `W` (`[out_features, in_features]`)
with two smaller matrices `U` (`[out_features, rank]`) and `V` (`[rank, in_features]`)
such that `W ≈ U @ V`. The forward pass becomes two smaller matmuls instead of
one big one. Parameter count and (for `rank < in_features/2`-ish) compute both
drop. This is a *reversible reshape*, not a masking trick — with `rank="full"`
the layer behaves exactly like the original `nn.Linear`.

| param | default | valid values | what it means |
|---|---|---|---|
| `rank` | `"full"` | positive `int`, or `"full"` | Target inner dimension. `"full"` disables LRD entirely (no-op). Smaller `rank` = more compression, more approximation error. |
| `method` | `"svd"` | `"svd"`, `"svd-llm-v2"`, `"aa-svd"` | Which factorization algorithm computes `U`/`V`. See below. |
| `eps` | `1e-6` | `float > 0` | Numerical floor used when whitening/clamping eigenvalues (only matters for the calibrated methods). |

### Methods

| Method | Calibration needed | What it does |
|---|---|---|
| `svd` | none | Plain truncated SVD of the weight matrix: `W = U S Vᵀ`, keep the top-`rank` singular values/vectors. Fast, works everywhere, but treats every input direction as equally important — it doesn't know which directions in the input actually get used at inference time. |
| `svd-llm-v2` | `"covariance"` (standard mode) | **Whitened SVD.** Before factorizing, rescales the input-feature space by the calibrated activation covariance so that directions the model actually activates on are preserved more faithfully, then un-rescales after truncation. Needs one calibration pass to collect `E[xᵀx]` per layer. Better accuracy than plain `svd` at the same rank, at the cost of running calibration data through the model once. |
| `aa-svd` | `"cross_covariance"` + `"shifted_covariance"` (**cascade mode only**) | **Activation-aware SVD** that additionally accounts for how errors introduced in one layer propagate into the *next* layer's input, using a shifted (position-offset) copy of the model to estimate that cross-layer effect. Requires `manager.set_calibration_mode("cascade")` — see [Concepts → Standard vs. Cascade Calibration](concepts.md#standard-vs-cascade-calibration). The most accurate of the three, and the most expensive to calibrate. |

**Rule of thumb:** start with `svd` to sanity-check a rank budget cheaply; move
to `svd-llm-v2` once you have a calibration loader; reach for `aa-svd` only
when `svd-llm-v2` isn't accurate enough and you can afford cascade
calibration.

**Restrictions:** LRD only applies to 2-D matmul weights — it raises if you
target a Conv2d/Conv3d patch-embed kernel (`weight.dim() > 2`).

---

## 2. `"structured_pruning"` — Output Neuron Removal

**What it does:** Removes whole output neurons (rows of the weight matrix) —
e.g. entire attention heads or MLP intermediate channels — rather than
individual weights. This is what makes pruning actually *shrink* a dense
matmul (unlike unstructured pruning, which just adds zeros). Because removing
a layer's output rows also shrinks the *input* the next layer expects, hard
pruning cascades that column removal into coupled downstream layers
automatically.

| param | default | valid values | what it means |
|---|---|---|---|
| `ratio` | `0.0` | `float` in `[0, 1)` | Fraction of output neurons to remove. `0.1` = drop the least-important 10%. |
| `method` | `"magnitude"` | `"magnitude"`, `"gradient"`, `"random"` | How neuron *importance* is scored (lower score = pruned first). See below. |
| `granularity` | `"layer"` | `"layer"`, or a positive `int` | `"layer"` scores/prunes the whole output dimension as one pool. An `int` (e.g. `128` = head size) scores and prunes independently within each consecutive chunk of that size — e.g. prune the same fraction out of *each* attention head rather than letting pruning drain one head entirely. |
| `repeated_pattern` | `False` | `False`, `True`/`"max"`, or a positive `int` N | When set, instead of an independent mask per chunk, scores are combined (`reduce_op`) across chunks into **one shared length-`granularity` pattern**, which is then tiled back over the full output. `True`/`"max"` combines across *every* chunk; an int `N` combines within each run of `N` consecutive chunks and repeats that pattern only across that run. This is what lets layers with a different number of chunks (e.g. GQA `q_proj` vs `k_proj`, different head counts) share one pruning pattern. |
| `coupled_repeated_pattern` | `False` | `False`, or a positive `int` N | Only affects what gets *cascaded* onto coupled downstream layers during hard pruning — this layer's own output is still pruned by the un-expanded mask. Each length-`granularity` chunk of the keep-mask is repeated `N` times in place before being used to prune a coupled layer's input columns. For a downstream layer whose input width is `N`× this layer's own pruned output width. |
| `reduce_op` | `None` | `None`, `"add"`, `"multiply"` | How per-chunk or per-sibling scores are combined when `repeated_pattern` and/or `share_mask` are active. |
| `share_mask` | `False` | `bool` — **group-only**, set via `group=` in `manager.set(...)`, never via `criteria=` | Forces every layer in a group (e.g. `gate_proj`/`up_proj`, or GQA `q_proj`/`k_proj`) to prune the *identical* neurons, computed once from combined scores and reused by every group member. Required before hard-pruning any coupled group — see below. |

### Methods (importance scoring)

| Method | Calibration needed | What it does |
|---|---|---|
| `magnitude` | none | Scores each output row by its L2 norm — rows with small weights are assumed less important. Cheap, no data needed, a reasonable default. |
| `gradient` | `"weight_grad"` (needs `manager.set_calibration_loss(...)`) | Scores each row by the L2 norm of `weight * weight_grad` (a first-order Taylor estimate of how much removing that row would change the loss). More data-driven than magnitude, but needs a calibration pass with a loss function and backward pass. |
| `random` | none | Assigns random scores. Useful as an ablation baseline, or when you don't care which specific neurons go (e.g. structural/latency experiments). |

### Soft vs. hard apply

- **Soft** (`manager.apply(hard=False)`) zeroes the pruned rows in place —
  same shape, fully reversible via `manager.restore()`. Good for
  fine-tuning/eval before committing.
- **Hard** (`hard=True`) actually deletes the rows and resizes
  `weight`/`bias`/`out_features`, then **cascades** the column removal into
  every coupled downstream layer (driven by `pruning.output_dependence` in
  the model's `indexing_*.py`). Irreversible.

### Grouped pruning (shared masks)

Some layers must be pruned identically for the model to stay well-formed —
e.g. `gate_proj`/`up_proj` (element-wise multiplied together, so their
surviving channels must line up) or GQA `q_proj`/`k_proj` (share head
structure). `manager.auto_groups()` builds these groups from the model's
indexing metadata; `share_mask=True` (set via `group=`, never `criteria=`)
makes the first compressor in the group compute one reduced mask that every
member reuses. Hard-pruning a coupled group without `share_mask=True` on all
members raises a `ValueError` rather than silently producing mismatched
shapes.

```python
groups = manager.auto_groups()
for g in groups:
    manager.set("structured_pruning", "share_mask", True, group=g)
    manager.set("structured_pruning", "reduce_op", "add", group=g)
manager.set("structured_pruning", "ratio", 0.1, criteria="mlp")
manager.apply(hard=True)
```

**Scope:** only MLP structured pruning is wired end-to-end through
convert/export today. Attention (q/k/v) hard pruning changes `head_dim`
(and the GQA v→o coupling isn't 1:1), which needs attention-forward/config
changes — deferred.

---

## 3. `"unstructured_pruning"` — Weight-Level Sparsity

**What it does:** Zeroes out individual weight *elements* (not whole rows)
based on an importance score, producing a sparse-but-same-shaped matrix. This
doesn't reduce compute/parameter count on ordinary dense hardware (the zeros
are still stored and multiplied) — its value is either as a regularizer
during fine-tuning, or as a stepping stone to a sparse kernel/format later.

| param | default | valid values | what it means |
|---|---|---|---|
| `ratio` | `0.0` | `float` in `[0, 1)` | Fraction of weight *elements* to zero out. |
| `method` | `"magnitude"` | `"magnitude"`, `"gradient"`, `"random"` | Same scoring idea as structured pruning, but applied per-element instead of per-row. See below. |
| `granularity` | `"layer"` | `"layer"`, `"neuron"`, or a positive `int` | `"layer"` pools all elements of the weight together and prunes the lowest-scoring `ratio` fraction globally. `"neuron"` prunes independently within each output row. An `int` chunks the flattened weight and prunes within each chunk. |

### Methods (importance scoring)

| Method | Calibration needed | What it does |
|---|---|---|
| `magnitude` | none | Scores each weight element by `abs(weight)` — smallest-magnitude elements pruned first. |
| `gradient` | `"weight_grad"` (needs `manager.set_calibration_loss(...)`) | Scores each element by `abs(weight * weight_grad)` — a per-element loss-sensitivity estimate. |
| `random` | none | Random per-element scores — ablation baseline. |

Masks are registered as a `weight_mask` buffer and survive `restore()`
(so straight-through-estimator fine-tuning can keep using them across
compress/restore cycles) — call `manager.remove_masks()` to explicitly drop
them. `manager.reapply_masks()` re-zeroes pruned elements after an optimizer
step (masked elements still receive gradient during backward, since
`dL/dW = xᵀ · dL/dy` doesn't depend on `W`, so a plain optimizer step would
otherwise un-zero them).

There's currently no sparse-tensor/hardware-accelerated backend for hard
unstructured pruning — `hard=True` falls back to the same in-place zeroing as
soft pruning (with a warning).

---

## 4. `"quantization"` — Reduced-Precision Weights and Activations

**What it does:** Represents weight (and optionally activation) values with
fewer bits — e.g. mapping a float32 weight to an 8-bit integer plus a
per-tensor or per-channel float `scale`, such that `weight ≈ qdata * scale`.
Soft apply *fake-quantizes* (round-trips through the quantized representation
but stores the result back as float, so accuracy effects are visible without
an actual bit-width change); hard apply calls into `torchao` to produce
genuinely packed low-bit tensors.

| param | default | valid values | what it means |
|---|---|---|---|
| `method` | `"vanilla"` | `"vanilla"`, `"gptq"` | Weight quantization algorithm. See below. |
| `precision` | `"full"` | `"full"`, `"binary"`, or `int` in `[2, 16]` | Weight bit-width. `"full"` disables quantization. `int` is a **bit count**, e.g. `8` for int8, `4` for int4 — not the string `"int8"`. Hard apply additionally requires `precision <= 8` (and doesn't support `"binary"`). |
| `granularity` | `"per_tensor"` | `"per_tensor"`, `"per_channel"` | Whether one scale is computed for the whole weight matrix (`per_tensor`) or one scale per output row (`per_channel`, generally more accurate). |
| `sparsity` | `0.0` | `float` in `[0, 1)` | Optional extra unstructured-pruning pass folded into the (soft) quantization step: this fraction of quantized weight elements is masked out and the corresponding original float values kept instead. |
| `sparse_method` | `"magnitude"` | `"magnitude"`, `"random"` | Scoring method for the `sparsity` mask above. |
| `precision_activation` | `"full"` | `"full"`, or `int` in `[2, 16]` | Bit-width for **fake-quantizing input and output activations** via forward hooks (weights are quantized structurally; activations are always fake-quant only, at runtime). `"full"` disables activation quantization. Requires calibration (`"activation_range"` / `"output_activation_range"` summaries) — set up automatically once you call `manager.set_calibration_data(...)`. |
| `method_activation` | `"maxmin"` | `"maxmin"` | How activation scale/zero-point are derived from the calibrated min/max range. Currently the only implemented option. |
| `scheme_activation` | `"asymmetric"` | `"symmetric"`, `"asymmetric"` | `"symmetric"` uses one scale around zero (`zero_point = 0`); `"asymmetric"` fits the scale/zero-point to the actual observed `[min, max]` range — usually a tighter fit for activations that aren't centered at zero (e.g. post-ReLU/SiLU). |
| `eps` | `1e-6` | `float > 0` | Numerical floor added to the weight scale denominator to avoid division by zero. |

### Weight quantization methods

| Method | Calibration needed | What it does |
|---|---|---|
| `vanilla` | none | Standard max-abs (or, for `precision="binary"`, sign+mean-abs) round-to-nearest quantization: pick a scale from the weight's own value range, round every element to the nearest representable level. Fast, no data needed, but spreads no reconstruction error correction across weights. |
| `gptq` | `"covariance"` | **GPTQ** (Frantar et al., 2022): uses the calibrated input covariance (`E[xᵀx]`) as a Hessian approximation and quantizes columns one at a time, propagating each column's rounding error into the still-unquantized columns via Cholesky-based Hessian inversion — minimizing `‖WX − W_qX‖²` rather than just rounding each weight independently. Meaningfully better accuracy than `vanilla` at low bit-widths (e.g. 4-bit), at the cost of a calibration pass and heavier compute during `apply()`. Falls back to vanilla round-to-nearest if no covariance is available. |

### Soft vs. hard apply

- **Soft** (`hard=False`, default): fake-quantizes in place — weight values
  are replaced by their round-tripped (quantize → dequantize) float
  approximation, so the module keeps the exact same shape/dtype and stays
  fully reversible. This is what lets you *see* accuracy impact before
  committing.
- **Hard** (`hard=True`): calls `torchao.quantization.quantize_()` to
  produce a genuinely packed low-bit weight tensor (`Int8WeightOnlyConfig`
  for 8-bit, `IntxWeightOnlyConfig` for 2–7 bit). Requires `torchao`
  installed, `precision <= 8`, and doesn't support `"binary"`. Irreversible,
  and only applies to `nn.Linear` submodules in current `torchao` — targeting
  something else (e.g. an embedding or conv layer) raises rather than
  silently no-op-ing.

### LRD interaction

If a layer has already been LRD-factored (`module.rank` is an `int`), soft
quantization of its `weight` (the `U` factor) automatically corrects the
calibrated covariance to `V @ C @ Vᵀ` before computing the scale — because the
effective input to `U` is `V·x`, not the original `x`. No extra configuration
needed; this happens automatically whenever `gptq` (or any covariance-based
method) is combined with LRD on the same layer.

---

## Criteria Language

Every `manager.set(...)`/`apply(...)`/`restore(...)` call takes a `criteria`
argument that selects which layers the setting applies to:

| criteria | matches | example |
|---|---|---|
| `None` or `"all"` | every scheme | `manager.set("lrd", "rank", 64)` |
| `int` | all layers in that block index | `criteria=2` |
| `str` | layers whose path contains the substring | `criteria="mlp"` |
| `[[str, int, ...]]` | AND of all items in the inner list | `criteria=[["mlp", 5]]` → block 5 AND "mlp" |
| `[str, int, ...]` | OR across items | `criteria=["q_proj", 3]` → "q_proj" OR block 3 |

`share_mask` (and any other option listed in `GROUP_OPTIONS`) is the one
exception — it can only be set through `group=`, never `criteria=`.

---

## See Also

- [Concepts](concepts.md) — the `Compressor`/`CompressionScheme`/manager
  mental model, calibration pipeline, and VCON smooth-compression workflow.
- [Basic LRD example](examples/01_basic_lrd.md)
- [Calibrated LRD (SVD-LLM-v2) example](examples/02_calibrated_lrd_svd_llm_v2.md)
- [`scripts/compression/compress_infer_qwen.py`](https://github.com/CESOIA/transformer-surgeon/blob/main/scripts/compression/compress_infer_qwen.py) — a runnable, heavily-commented script combining LRD + structured pruning + quantization on Qwen2-0.5B.
