# Recap: transformersurgeon/blocks/

All paths relative to `/home/AD/luciano.prono/container-py310-executorch-qnn/transformer-surgeon/` (repo root = "R"). Files read fully. Line refs are `file:line` inside `transformersurgeon/blocks/` unless prefixed.

## 1. Purpose
`blocks/` holds (a) the compressible leaf layers the compression library swaps into HF models (`LinearCompressed`, `EmbeddingCompressed`, `Conv2d/3dCompressed`, `VCONBlock`) and (b) a simplified, export-ready re-implementation of the transformer (`TransformerDecoder`/`TransformerEncoder`, MHA, MLP, RMSNorm, RoPE, KV-cache write strategies, two custom torch ops) that `utils/convert.py::convert_for_export` builds from an HF model + indexing dict. Blocks are "frozen already-compressed" modules: pruning geometry is decided once in `finalize_*` hooks, forward is static for `torch.export`. It is the lowest layer (`compression/` and `export/` import it); `pruning_dims.py` lives here to avoid a cycle. Mixed concerns: leaf-compression layers, graph model, config classes, backend op definitions.

## 2. File tree (lines)
```
blocks/
  __init__.py              9   star-imports: linear_compressed, embedding_compressed, conv_compressed, pruning_dims, vcon_block, decoder, encoder, config, rope
  atomic.py               14   AtomicSum
  vcon_block.py           70   VCONBlock
  linear_compressed.py   112   LinearCompressed
  embedding_compressed.py 145  EmbeddingCompressed
  conv_compressed.py      30   Conv2dCompressed, Conv3dCompressed
  pruning_dims.py         74   effective_num_pruned / effective_out_features / pruning_config_dims
  norm.py                 64   RMSNorm
  rope.py                424   RoPE freq variants, cos/sin precompute, apply, prune projection
  mlp.py                 190   MLP, MLPGated, activation_map
  mha.py                 930   attention(), MHABase, MHAEncoder, MHAEncoderFusedProj, MHACausal
  kv_cache_ops.py         29   torch.ops.tsurgeon.kv_cache_update
  paged_attention_ops.py 106   torch.ops.tsurgeon.paged_attention + page helpers
  decoder.py             325   TransformerDecoderBlock, TransformerDecoder
  encoder.py             306   TransformerEncoderBlock, TransformerEncoder
  config.py              299   CustomDecoderConfigCompress, CustomEncoderConfigCompress
  (total 3126; __pycache__ ignored)
```
`__init__` does NOT export mha/mlp/norm/atomic/kv_cache_ops/paged_attention_ops names (they are reached by module path). It exports `TransformerDecoder` (decoder `__all__`), `TransformerEncoder(+Block)`, config classes, all `rope` `__all__` names, leaf layers, `VCONBlock`, pruning_dims funcs. Import order matters: `mha.py:9` and `mlp.py:4` do `from . import LinearCompressed` (relies on `__init__` having already imported linear_compressed first).

## 3. Per-file inventory

### atomic.py
- `AtomicSum(nn.Module)`: `__init__()`, `forward(x1, x2) -> x1+x2`; `input_num=2` attribute. Purpose: named add node for the residual join so exporters/quantizers can target it (the add is a module, so it shows in `nn_module_stack`). No config.

### vcon_block.py
- `VCONBlock(block_a, block_b=None)`: `beta=None`. `forward(input)`: beta None or >=1 -> `block_a`; <=0 -> `block_b`; else `beta*a + (1-beta)*b`. `set_beta(beta)` validates [0,1]. `__repr__/__str__`. Compression-training-time construct (not export).

### linear_compressed.py
- `LinearCompressed(nn.Linear)(in_features, out_features, bias=True, rank=None, device, dtype)`.
  - `__init__`: if `out_features<=0` sets `self.skip=True` and returns BEFORE `super().__init__()` (fully-pruned layer, `forward` returns input unchanged). Else `skip=False`, `linear_V=None`, `init_lrd(rank)`.
  - `init_lrd(rank)`: `self.rank = "full" if None else rank`; if int: creates `linear_V = nn.Linear(in, rank, bias=False)` and reshapes `weight` to `[out, rank]` (kaiming init). Called by compression/lrd.py.
  - `cancel_lrd()`: back to full; drops `linear_V`, re-allocates `weight [out,in]` (random init; weights are restored elsewhere).
  - `forward(x)`: branches: `skip` passthrough; flatten leading dims to rank-2 for >2-D input then restore (line 88-105, comment says "QNN FullyConnected more robust", TODO make optional/QNN-only) ; `rank` int -> `linear_V` then `F.linear`.
  - Attributes read/written by other code: `block_prune_mask`, `rope_prune_mask` (buffers set by compression), `skip`, `rank`, `linear_V`.

### embedding_compressed.py
- `EmbeddingCompressed(nn.Module)(num_embeddings, embedding_dim, rank=None, device, dtype)`: same skeleton as LinearCompressed (skip when `embedding_dim<=0`, `init_lrd`, `cancel_lrd`), `forward`: `F.embedding` (full) or `F.embedding(table[N,rank]) @ linear_V.weight` (LRD). Has `in_features/out_features/bias=None` aliases. Big TODO(P3a) comment (no weight-quant path). `skip` branch returns before setting `num_embeddings` (so `__repr__` would fail for skipped).

### conv_compressed.py
- `Conv2dCompressed(nn.Conv2d)`, `Conv3dCompressed(nn.Conv3d)`: only a `__repr__`. No LRD by design. Docstring names ViT patch-embed and "Qwen-VL temporal patch-embed".

### pruning_dims.py
- `effective_num_pruned(dim, ratio, granularity="layer"|int) -> int`: rows removed (`int(ratio*dim)` or per-chunk).
- `effective_out_features(dim, ratio, granularity) -> int`: `dim - removed`.
- `pruning_config_dims(layer_config: dict, dim) -> int`: reads `layer_config["structured_pruning"]["ratio"|"granularity"]`.
- Single source of truth shared with `compression/structured_pruning_methods/mask_generation.py`.

### norm.py
- `RMSNorm(hidden_size, dtype=None, prescale=True, upcast=False, eps=1e-5)`; `weight` param; `forward(x)`: 3 atoms in one class selected by flags:
  1. `upcast=True` (HF/TensorRT style: fp32 normalize, cast back, then `weight*x`; takes precedence)
  2. `prescale=True` (divide by clamp(max|x|,1) first; fp16-overflow guard)
  3. plain (`x*x` mean, rsqrt, `weight*x`; written as `x*x` not `pow` so QNN RecomposeRmsNorm matches).
- Docstring/comments contain QNN/HTP/TensorRT rationale.

### rope.py (all free functions)
- `rope_parameters_from_config(config) -> dict`: flattens HF `rope_parameters` (tf5) / `rope_scaling` (tf4, legacy `type`) + `rope_theta`; maps legacy `"mrope"` type to `"default"`.
- `rope_theta_from_config(config, default=1e6) -> float`: warns if scaled type (encoder path only supports plain RoPE).
- `_inv_freq_default(head_dim, params, device)`, `_inv_freq_llama3(...)`: variants (llama3 requires keys factor/low_freq_factor/high_freq_factor/original_max_position_embeddings).
- `ROPE_INV_FREQ = {"default":..., "llama3":...}`: the registry (already an "atom dictionary").
- `resolve_rope_type(config, indexing_rope_type="default")`: indexing value `"auto"` takes config type; fixed type mismatch raises. Called by `config.py:165`.
- `precompute_rope_inv_freqs_for(rope_type, head_dim, params, device)`: dispatch into registry.
- `precompute_rope_inv_freqs(head_dim=128, base=1e6, device)`: plain inv-freq (docstring wrongly says MRoPE).
- `precompute_mrope_inv_freqs(head_dim, section_dims=None, base, section_bases=None, device)`: list of per-section inv-freqs (or plain tensor if no sections).
- `precompute_rope_cos_sin_half(inv_freq, seq_len, start_pos)` -> `(cos, sin)` each `(seq,1,half)`. NOTE: `start_pos` is required (no default).
- `precompute_mrope_cos_sin_half(inv_freq_sections, seq_len, position)`: concatenates per-section cos/sin. Calls `precompute_rope_cos_sin_half(..., position=section_pos)` (kwarg does not exist, see section 7).
- `apply_rope_multihead(x[seq,heads,head_dim], cos, sin)`: rotate-half (chunk in 2), returns in x dtype.
- `_derive_rope_prune_pattern(keep_mask, head_dim, num_heads, num_kv_groups, source)`: validates and extracts per-kv-group kept-frequency pattern from a structured-pruning mask.
- `build_rope_prune_projection(q_keep_mask, k_keep_mask, head_dim, q_num_heads, k_num_heads)` -> `(kv_groups, kept, half)` 0/1 selection matrices. Pure compression-geometry helper; also called by `compression/structured_pruning.py:425`.

### mlp.py
- `activation_map = {"gelu":GELU, "relu":ReLU, "silu":SiLU}` (module global; unknown name -> KeyError).
- `MLP(embed_dim, hidden_dim, bias_required=None, activation='gelu', compression_config=None, dtype=None)`: `up_proj -> act -> down_proj`. Reads `compression_config[up_proj|down_proj]["lrd"]["rank"]` and `["structured_pruning"]` via `pruning_config_dims` for effective hidden/out dims; `bias_required[up_proj|down_proj]`.
- `MLPGated(...)`: same + `gate_proj`; `forward = down(act(gate(x))*up(x))`. Eff hidden taken from `up_proj` config only.
- Layer names are fixed (`up_proj/gate_proj/down_proj`) and are a contract with `utils/convert.py` layer_matching and compression config paths.

### kv_cache_ops.py
- `torch.library.custom_op("tsurgeon::kv_cache_update")` `kv_cache_update(cache, update, write_index)`: functional `index_copy` along dim 2 (BHSD); `register_fake`. Docstring: ONNX translation (`export/onnx`) is opset-24 `TensorScatter`, TensorRT aliases buffer. The op itself is backend-neutral; only the docstring is backend-specific.

### paged_attention_ops.py
- `KV_PAGE_SIZE=128` (= Edge-LLM `kTOKENS_PER_PAGE`), `num_kv_pages(max_cache_len)`, `identity_page_table(num_pages) -> (1,2,P) int32`, `_rope(x, cos, sin)` (rotate-half, fp32 math), `paged_attention(qkv, kv_pool, context_lengths, rope_cos_sin, kvcache_start_index, kv_page_table, num_q_heads, num_kv_heads, head_size, attention_scale)` custom op (eager reference, batch 1 only; fused RoPE + page write + causal attention over filled slots) + `register_fake`. The whole contract (tensor layouts) is dictated by TensorRT Edge-LLM `AttentionPlugin` (backend-specific by design). `kvcache_start_index` is accepted but unused in the eager body (positions derive from `context_lengths`).

### mha.py
Module constants: `CACHE_IMPLS = ("mutable","io_scatter","io_concat","io_inplace","io_paged")`, `ATTN_IMPLS = ("manual","sdpa","custom_sdpa","paged")`.
- `attention(query, key, value, attn_mask=None, kv_layout="shd", masked_softmax=False, scale=None)` (line 11): explicit GQA-by-broadcast SDPA. Options: `kv_layout` "shd"|"hsd" (transposes k/v); mask mode `masked_softmax` False -> `scores + mask`, True -> `where(mask==0, scores, amin(scores)-20)`; `scale` None -> 1/sqrt(q_head_dim). Returns `(q_heads, q_len, v_dim)`. Long QNN/HTP rationale in comments.
- `MHABase(nn.Module)(embed_dim, num_heads, bias_required=None, kv_num_heads=None, use_sdpa=False, compression_config=None, dtype=None, **kwargs)` (line 99): builds `q_proj,k_proj,v_proj,out_proj` as `LinearCompressed` (rank from `compression_config[<proj>]["lrd"]["rank"]`; defaults to "full"; bias from `bias_required`). Buffer `rope_freq_proj`, flag `rope_pruned`. `**kwargs` swallows decoder-only keys. Methods:
  - `score_scale` (property): 1/sqrt(original head_dim) (keeps softmax temp under q/k pruning).
  - `key_head_dim`, `value_head_dim` (properties): from current `k_proj/v_proj.out_features` (prune-aware).
  - `finalize_head_pruning()`: reads `q_proj.block_prune_mask`, sets `num_heads` to kept heads; raises if combined with rope_prune_mask. Called by `utils/convert.py:402`.
  - `finalize_rope_pruning()`: compares q/k head dims to `head_dim`, reads `rope_prune_mask` buffers, builds `rope_freq_proj` via `build_rope_prune_projection`; sets `rope_pruned`. Called by convert.py:404.
  - `_project_rope(cos, sin) -> (cos_q, sin_q, cos_k, sin_k)`: identity if not pruned, else per-kv-group matmul with `rope_freq_proj` then `repeat_interleave` for q.
- `MHAEncoder(MHABase)` (line 347): `forward(x[seq,embed], rope=None)`; unbatched, no cache, non-causal. Branches: `use_sdpa` -> `F.scaled_dot_product_attention(..., enable_gqa=True, scale=score_scale)` else `attention()`. TODO(custom_sdpa-encoder).
- `MHAEncoderFusedProj(nn.Module)` (line 407): single `qkv_proj` (3*embed), no GQA, no pruning awareness, no dtype; own `compression_config["qkv_proj"]`. Branches: `use_sdpa`/`attention()`. Comment "Qwen-style".
- `_ensure_custom_sdpa_ops()` (line 485): imports `executorch.extension.llm.custom_ops.custom_ops` (registers `torch.ops.llama.custom_sdpa/update_cache`), RuntimeError if missing.
- `MHACausal(MHABase)` (line 504): ctor reads from kwargs: `cache_impl`, `add_batch_dim`, `max_cache_len`, `attn_impl`, `masked_softmax`; `use_sdpa` (via base) as fallback for default `attn_impl`. Validates combos (`paged` <-> `io_paged`; `masked_softmax` needs `manual`; `custom_sdpa` needs `mutable`). Registers non-persistent `key_cache`/`value_cache` buffers (always, shape from `cache_shape`; `(0,)` for io_paged) and for custom_sdpa dedicated fp32 `custom_sdpa_key_cache/value_cache` `(1,L,Hkv,D)`. Methods:
  - `cache_shape(which)`: io_paged -> `(2,P,128,Hkv,Dv)`; io_inplace -> BHSD `(1,Hkv,L,D)`; else SHD `(L,Hkv,D)`. Used by `export/common.py` (lines 135,140,141,1045-1053).
  - `_cache_geometry_ok()`, `_check_custom_sdpa_dims()`, `_check_cache_geometry()`: forward-time guards.
  - `finalize_cache_geometry()`: resize internal buffers after weights loaded (convert.py:408).
  - **Cache-write atoms** (selected by `cache_impl` in forward, lines 883-890): `_write_mutable` (in-place `index_copy_` on module buffer; returns only output), `_write_scatter` (`index_put`, functional, io), `_write_concat` (one-hot mask + `where`, in_seq_len==1 only), `_write_inplace` (`kv_cache_update` custom op, BHSD, multi-token), plus `io_paged` handled wholesale in `_forward_paged`.
  - **Attention-kernel atoms** (selected by `attn_impl`): `manual` -> `attention()`; `sdpa` -> `F.scaled_dot_product_attention` with mask; `custom_sdpa` -> `_forward_custom_sdpa` (ExecuTorch llama custom ops, own fp32 cache, ignores `attn_mask`, `is_causal=True`); `paged` -> `_forward_paged` (packs q|k|v, fused op).
  - **Layout atom**: `add_batch_dim` (accepts `(1,seq,embed)`; q/k/v `.view` absorbs batch; output reshaped back).
  - `forward(x, pos_id, attn_mask, key_cache=None, value_cache=None, rope=None, paged=None)`: returns `output` (mutable) or `(output, k_cache, v_cache)` (io_*; io_paged returns `(out, pool, None)`). Return arity depends on `cache_impl` (decoder block mirrors this at decoder.py:343-355).

### decoder.py
- `TransformerDecoderBlock(config, block_index)` (line 4): reads config keys `hidden_size, num_attention_heads, intermediate_size, hidden_act, attn_type, mlp_type, norm_type, use_sdpa, attn_impl, masked_softmax, max_cache_len, cache_impl, add_batch_dim, rmsnorm_prescale, rmsnorm_upcast, rms_norm_eps, dtype, num_key_value_heads, compression_config, bias_required`. Splits `config.compression_config` by prefix `blocks.{i}.` into `{"attn":{}, "mlp":{}}`. Selection ifs: `norm_type=="rmsnorm"` only; `attn_type=="mha_causal"` only; `mlp_type in {"mlp_gated","mlp"}`. Submodules fixed names `norm_in, attn, norm_out, mlp, atomic_sum`. `forward(x,pos_id,attn_mask,key_cache,value_cache,rope,paged)`: pre-norm residual; branches on `cache_impl=="mutable"` for return arity.
- `TransformerDecoder(config)`: `blocks` ModuleList, final `norm` (hard-coded `RMSNorm`, ignores `norm_type`), buffers `q_pos,k_pos` (persistent, `(L,1)`/`(1,L)`), `rope_cos/rope_sin` (persistent, `[L,1,half]`, from `precompute_rope_inv_freqs_for(config.rope_type, head_dim, rope_parameters_from_config(config))`), `mask_penalty=-65504`, and for io_paged non-persistent `rope_cos_sin` and `kv_page_table`. `forward(x, pos_id, key_caches=None, value_caches=None)` has 3 mask/cache paths:
  1. `io_paged`: no mask, builds `paged=(start+seq, rope_cos_sin, start, page_table)`; returns `(norm(x), pools, None)`.
  2. `io_inplace`: multi-token mask `where(k_pos > positions[:,None], penalty, 0)[None]`; rope gathered at `pos_id+arange`.
  3. others: row `pos_id` of `where(q_pos<k_pos, penalty, 0)`; `mutable` returns `x`, io_scatter/io_concat return `(x, kcs, vcs)`.
  Comment 414-419 explains -65504 choice (Qwen2.5 k-bias logits).

### encoder.py
- `TransformerEncoderBlock(config, block_index)`: keys `hidden_size,num_attention_heads,intermediate_size,hidden_act,attn_type,mlp_type,norm_type,norm_position,use_sdpa,num_key_value_heads,compression_config,bias_required,rms_norm_eps`. Selection: norm `rmsnorm|layernorm` (layernorm = plain `nn.LayerNorm`, no compression handle); attn `mha_encoder|mha_encoder_fused_proj`; mlp `mlp_gated|mlp`. `forward(x, rope=None)`: `norm_position` `"pre"` or `"post"` residual ordering (2 atoms). No dtype passed to submodules (always default dtype).
- `TransformerEncoder(config)`: keys `position_embedding_type` (`none|rope|mrope|trained_absolute`), `enable_patchify_input, patch_size, num_channels, use_cls_token, patch_bias` (read via getattr from HF passthrough config), `use_final_norm`, `norm_type`, `mrope_section_dims/bases/positions`, `max_position_embeddings`. Methods: `_image_to_patches(images)` (unfold patchify), `_prepare_input(x, preprocess_images)`, `_build_rope(seq_len, rope_pos)`, `_apply_absolute_positional_embeddings(x, position_ids)`, `forward(x, preprocess_images=False, position_ids=None, rope_pos=0)`. Own `patch_projection` Linear (not compressible `LinearCompressed`), `cls_token`, `inv_freq` parameter(s).

### config.py
- `CustomDecoderConfigCompress(PretrainedConfig)` (`model_type="transformersurgeon-custom-decoder"`): keys `num_hidden_layers,hidden_size,num_attention_heads,intermediate_size,hidden_act="silu",attn_type="mha_causal",mlp_type="mlp_gated",norm_type="rmsnorm",num_key_value_heads,compression_config,bias_required,use_sdpa,attn_impl,max_cache_len=2048,cache_impl="mutable",add_batch_dim,rmsnorm_prescale,rmsnorm_upcast,rope_type="default",masked_softmax,indexing`. With `indexing` given calls `utils.configuration.init_compressed_config`. `from_source_config(source_config_obj, source_indexing, converted_indexing, compression_config, bias_required, use_sdpa, attn_impl, max_cache_len, cache_impl, add_batch_dim, rmsnorm_prescale, rmsnorm_upcast, masked_softmax)`: maps source attr names through indexing keys `num_blocks_attr, embed_dim_attr, num_heads_attr, mlp_hidden_dim_attr, mlp_activation_attr, kv_num_heads_attr, attn_type, mlp_type, norm_type, rope_type`; passes remaining HF fields through as kwargs.
- `CustomEncoderConfigCompress`: same pattern, keys `norm_position, use_final_norm, position_embedding_type`, defaults gelu/mha_encoder/mlp/layernorm/pre. Default `mlp_type` differs between ctor ("mlp_gated") and `from_source_config` fallback ("mlp") on the decoder (config.py:21 vs :154).
- Config is the de-facto selection dictionary today (flat string keys: `attn_type, mlp_type, norm_type, cache_impl, attn_impl, rope_type, norm_position, position_embedding_type`) plus boolean flags.

## 4. Dependencies

### Internal (within blocks) and outward imports
- `__init__`: star-imports 9 modules (listed section 2).
- `mha.py` -> `.rope` (apply_rope_multihead, build_rope_prune_projection), `.kv_cache_ops`, `.paged_attention_ops` (KV_PAGE_SIZE,num_kv_pages,paged_attention), `from . import LinearCompressed`; lazily `executorch.extension.llm.custom_ops.custom_ops` (external).
- `mlp.py` -> `from . import LinearCompressed`, `.pruning_dims.pruning_config_dims`.
- `decoder.py` -> `.mha.MHACausal`, `.mlp`, `.norm.RMSNorm`, `.rope` (3 funcs), `.atomic`, `.paged_attention_ops` (identity_page_table,num_kv_pages).
- `encoder.py` -> `.atomic`, `.mha` (MHAEncoder, MHAEncoderFusedProj), `.mlp`, `.norm`, `.rope` (5 funcs).
- `config.py` -> `transformers.PretrainedConfig`, `..utils.configuration.init_compressed_config` (UPWARD dependency out of blocks/), `.rope.resolve_rope_type`.
- Others (`linear_compressed, embedding_compressed, conv_compressed, vcon_block, norm, atomic, kv_cache_ops, rope, pruning_dims, paged_attention_ops`): torch only (+typing/math/warnings).

### Who imports what (grep over whole repo, excl. blocks/)
- `LinearCompressed`: `utils/modeling.py:5`, `utils/manager.py:22`, `utils/scheme.py:10`, `compression/{abstract,structured_pruning,coupled_pruning}.py`, `calibration/raw_data/base.py`, `export/common.py:206`, `export/export.py:34`, `export/executorch_exporters/qnn/qnn_export.py:477` (string/`isinstance` mentions), `models/modernbert_c/indexing_modernbert_c.py:25` (comment), `benchmarks/nonhom_lrd/run_benchmark.py:57`.
- `EmbeddingCompressed`: `compression/structured_pruning.py:15` (+ special branches), `coupled_pruning.py:117`, `quantization.py:509`, `export/export.py:34`, `utils/{modeling,scheme,manager}.py`.
- `Conv2dCompressed/Conv3dCompressed`: `utils/{modeling,scheme,manager}.py`, `compression/quantization.py:508`, `scripts/compression/compress_infer_vit.py:13`.
- `VCONBlock`: `utils/scheme.py:13` (+usage at 41,83,261,279,377), example `examples/03_hf_export_roundtrip.py:108` (via manager.init_vcon).
- `effective_num_pruned/effective_out_features`: `compression/structured_pruning_methods/{__init__,mask_generation}.py`. `pruning_config_dims`: only `mlp.py`.
- `TransformerDecoder`: `export/export.py:8`, `utils/convert.py:7`, tests, `export/executorch_exporters/qnn/qnn_export.py:478,696` (string match), `scripts/compression/compress_infer_qwen.py:266`, `examples/04_cache_impl_export_interface.py:149`. `TransformerEncoder`: `utils/convert.py:8`, `scripts/compression/compress_infer_vit.py:250`. Block classes: `utils/convert.py:399` (mention).
- `CustomDecoderConfigCompress/CustomEncoderConfigCompress`: `utils/convert.py:9,329,361`; tests `test_io_inplace_cache`, `test_add_batch_dim`.
- `MHACausal`: tests; `export/common.py:164,249` (comments); `qnn_export.py:460,476` (matches the class name inside `nn_module_stack` strings: `"MHACausal" in t` — a stringly coupling to a class name). `MHABase`: `compression/block_pruning.py:33` (docstring).
- `MHAEncoder`: test_gqa_rope_pruning only. `MHAEncoderFusedProj`, `AtomicSum`, `CACHE_IMPLS`, `ATTN_IMPLS`, `activation_map`, `MLP`/`MLPGated` names, `num_kv_pages`, `identity_page_table`, `precompute_mrope_*`, `apply_rope_multihead`: no importer outside blocks/ (MLP/MLPGated only inside blocks; the many "MLP" grep hits elsewhere are unrelated words).
- `kv_cache_update`: `export/onnx/onnx_export.py:56-58` (import to register the op + `torch.ops.tsurgeon.kv_cache_update.default` translation), tests. `paged_attention`: `export/tensorrt/edgellm_attention.py:68` (register), `export/onnx/onnx_export.py:25,151,153`. `KV_PAGE_SIZE`: test.
- RoPE funcs: `transformersurgeon/__init__.py:4` imports `precompute_rope_inv_freqs` (re-export, otherwise unused in repo code); `ROPE_INV_FREQ`/`resolve_rope_type` referenced in comments of `models/{llama_c,qwen2_c}/indexing_*.py` (they set indexing `rope_type`); `build_rope_prune_projection`: `compression/structured_pruning.py:425`; `rope_parameters_from_config`, `rope_theta_from_config`, `precompute_rope_*`: tests only.
- `RMSNorm`: no code import outside blocks; `export/common.py:614`, `compression/smoothquant.py:10,147`, `coupled_pruning.py:115`, `utils/{scheme,manager}.py` reference norm modules by class-name string/type name (smoothquant.py:10 may import by name; unverified).
- Methods called from outside: `finalize_head_pruning/rope_pruning/cache_geometry` (`utils/convert.py:402-409`, via `hasattr`), `cache_shape` (`export/common.py`), `init_lrd/cancel_lrd/linear_V` (compression/*, export/*).
- Strings coupled to config keys/names: `utils/convert.py` builds options `use_sdpa, attn_impl, max_cache_len, cache_impl, add_batch_dim, rmsnorm_prescale, rmsnorm_upcast, masked_softmax` (lines ~305-345) and the layer-name conventions `blocks.{i}.attn`, `norm_in/norm_out`.

## 5. Violations of target rules

### (a) Model names / model-specific logic
- `rope.py:19` `"mrope"` legacy Qwen2-VL marker remapped to `"default"` (comment names Qwen2-VL; model-specific special-case).
- `rope.py:46-53,74` `_inv_freq_llama3` / `"llama3"` registry key (a RoPE variant named after a model family; algorithm itself is generic -> rename e.g. `rope_freq_scaled_bands`).
- `rope.py:82-84` docstring examples Qwen2/Llama/TinyLlama; `rope.py:88` `"auto"` semantics.
- `mlp.py:94` "Qwen-style gated MLP"; `mha.py:407` "Qwen-style fused projection MHA (No GQA)".
- `decoder.py:225` comment "Qwen2.5 (k biases ~300)" justifying `mask_penalty`; `norm.py:52` "Qwen2.5's ~5e3 activations"; `embedding_compressed.py:21-22` Qwen2.5/TinyLlama numbers; `conv_compressed.py:6,24` "ViT patch-embed", "Qwen-VL temporal patch-embed" (comments only).
- `mha.py:349,363` etc. use "llama" only as the ExecuTorch op namespace `torch.ops.llama.*` (`mha.py:745,746,772`): vendor namespace, counted under (b).
- `encoder.py:137-150`: patchify/ViT image path (`enable_patchify_input`, `patch_size`, `num_channels`, `use_cls_token`, `patch_bias`) is vision-model-specific logic in a "generic" encoder; `encoder.py:168-194` `mrope`/`trained_absolute` position types (VL/BERT-specific).
- `decoder.py:232`, `encoder.py:49`, `decoder.py:377`, `encoder.py:158` read HF-specific field `rms_norm_eps`; `config.py` `from_source_config` is built around HF-config + indexing-dict mapping (HF coupling, not model names).
- `config.py:61`: default `hidden_act="silu"`, `attn_type="mha_causal"`, `mlp_type="mlp_gated"` = Llama/Qwen defaults baked in.

### (b) Backend-specific logic or knowledge
- `mha.py:485-501` `_ensure_custom_sdpa_ops` + `attn_impl="custom_sdpa"` path (`mha.py:596-617, 638-641, 689-700, 732-781, 874-879`): ExecuTorch `torch.ops.llama.custom_sdpa/update_cache` (XNNPACK/portable CPU). Real backend code in `blocks/`.
- `mha.py:481-482,510,544-...` `cache_impl` names are export-backend driven: `mutable` ("QNN-only", mha.py:510), `io_scatter` ("QNN ScatterNd / TensorRT ScatterND", 514), `io_concat` ("backends lacking index_put", 516), `io_inplace` ("ONNX TensorScatter, TensorRT in-place", 518-523), `io_paged` ("TensorRT Edge-LLM XQA plugin", 524-529).
- `mha.py:783-800 _forward_paged`, `paged_attention_ops.py` (entire file; layout copied from Edge-LLM 0.10.1; `KV_PAGE_SIZE` = Edge-LLM constant), `kv_cache_ops.py` (ONNX `TensorScatter` motive).
- `mha.py:61-77` QNN/HTP VTCM/DDR numbers in `attention()` comments; `mha.py:29-35` `masked_softmax` = "vendor enable_masked_softmax form" (QNN idiom) and `masked_softmax` flag threaded through `config.py:34,61,88`, `decoder.py:226`, `utils/convert.py:310,320`.
- `norm.py:9-20,49-53` QNN/HTP rationale; `RMSNorm.prescale` (fp16 overflow, QNN) and `upcast` (TensorRT Edge-LLM) flags are backend-motivated numerics variants; the `x*x`-not-`pow` spelling is dictated by QNN `RecomposeRmsNorm` (`norm.py:49-54`).
- `linear_compressed.py:88-105` rank-2 flatten/reshape applied unconditionally "for QNN FullyConnected" (own TODO: make optional/QNN-only). Affects all backends' graphs.
- `decoder.py:259` comment XNNPACK dispatch overhead; `mha.py:555,580-583` QNN/TensorRT/XNNPACK notes; `mha.py:703` "QNN peak path"; `mha.py:77` XNNPACK partitioning; `mha.py:556` references AGENTS.md export option.
- `decoder.py:405-412` builds `rope_cos_sin` float32 table + `kv_page_table` buffers only for io_paged (TensorRT plugin inputs); `decoder.py:451-461` io_paged forward path.
- `config.py` carries `cache_impl, attn_impl, masked_softmax, rmsnorm_prescale, rmsnorm_upcast, add_batch_dim` as export-backend knobs inside a model config.
- Back-references from export code to block class names (reverse coupling): `export/executorch_exporters/qnn/qnn_export.py:460,476-478` (`"MHACausal" in t`, `TransformerDecoder'>`).

### (c) Compression-decision logic inside blocks
- `mlp.py:47-50,129-133` + `pruning_dims.py` : MLP sizes its Linear dims from `compression_config[...]["structured_pruning"]` (ratio, granularity). Intended single source of truth, but it is compression-config interpretation in the block (and `MLPGated` assumes gate shares up's mask). Also `mha.py` has no equivalent: MHA derives pruned dims by reading `out_features` of its (already resized) Linears instead.
- `mha.py:146-149,163-186` reading `lrd.rank` per projection; `mlp.py:44,125-127` same; `linear_compressed.py:43-79` `init_lrd/cancel_lrd` implement LRD state transitions (used by compression/lrd.py).
- `mha.py:199-345` `score_scale`, `finalize_head_pruning`, `finalize_rope_pruning`, `_project_rope` and `rope.py:284-409` (`_derive_rope_prune_pattern`, `build_rope_prune_projection`): all are structured-pruning (`position_linked`, `coupled_masks`, `block_prune_mask`, `rope_prune_mask`) knowledge. `rope.py:291-397` messages reference `structured_pruning`, `repeated_pattern='auto'`, `manager.auto_groups()`.
- `mha.py:154-161` v-pruning/GQA cascade comments (`coupled_repeated_pattern=group_size`); `mha.py:226` reads `q_proj.block_prune_mask`.
- `embedding_compressed.py:50-55` comments on StructuredPruner axis; `conv_compressed.py` comments about StructuredPruner/Quantizer layout.
- `vcon_block.py`: training-time compression scheme (not export).
- `utils/convert.py` is the other half of this logic (calls the `finalize_*` hooks by `hasattr`).

## 6. Duplicated code
1. `MLP` vs `MLPGated` (`mlp.py:14-92` vs `94-185`): identical ctor skeleton (default compression_config, rank extraction, `pruning_config_dims`, bias defaults, `activation_map` lookup, `down_proj`). Unify: one `MLP` with `gated: bool` or a shared base building `up/(gate)/down` from a list of proj names.
2. Default-dict/bias_required/compression_config/rank boilerplate repeated in `MHABase.__init__` (`mha.py:122-149`), `MHAEncoderFusedProj.__init__` (`mha.py:423-442`), `MLP` (`mlp.py:37-61`), `MLPGated` (`mlp.py:117-146`). Unify: helper `build_linear(name, in, out, compression_config, bias_required, dtype)`.
3. `LinearCompressed.init_lrd/cancel_lrd` (`linear_compressed.py:43-79`) vs `EmbeddingCompressed.init_lrd/cancel_lrd` (`embedding_compressed.py:93-127`): same logic, differ only in which dim is "input". Extract an `LRDMixin`. Also the `skip` early-return pattern (`linear_compressed.py:28-30`, `embedding_compressed.py:70-72`).
4. Compression-config prefix split `blocks.{i}.` -> `{"attn","mlp"}` copy-pasted in `decoder.py:237-249` and `encoder.py:32-44`. Also the "pre-norm residual block" forward (`decoder.py:332-355` vs `encoder.py:100-109`), the norm/attn/mlp construction ifs, and the final-norm construction (`decoder.py:376`, `encoder.py:156-164`). Unify: one generic `TransformerBlock` (norm_position, attn atom, mlp atom, norm atom).
5. `CustomDecoderConfigCompress.from_source_config` vs `CustomEncoderConfigCompress.from_source_config` (`config.py:73-169` vs `222-297`): `_get_attr_from_source`, `consumed_attrs`, passthrough filtering duplicated; ctor `compression_config`/`init_compressed_config` tail duplicated (`config.py:64-71`, `213-220`).
6. RoPE inverse-frequency: `precompute_rope_inv_freqs` (`rope.py:154-171`) vs per-section formula in `precompute_mrope_inv_freqs` (`rope.py:146-150`): same `1/(base**(arange(0,dim,2)/dim))`; mrope can call the plain function per section. `_inv_freq_default` (`rope.py:42`) is a thin wrapper of the same.
7. Rotate-half RoPE: `apply_rope_multihead` (`rope.py:249-282`) vs `_rope` in `paged_attention_ops.py:43-48` (same math, fp32 + partial-rotation variant). Also Edge-LLM plugin eager implementation in `export/tensorrt/edgellm_attention.py:54+` (not read in detail, guess: another copy).
8. GQA attention math: `attention()` (`mha.py:11-97`) vs the einsum core inside `paged_attention` (`paged_attention_ops.py:89-96`) and the SDPA calls in `MHAEncoder` (`mha.py:391-397`), `MHAEncoderFusedProj` (`470-473`), `MHACausal` (`897-902`). Unify behind one attention-kernel registry `{manual, sdpa, custom_sdpa, paged}`.
9. Attention-mask construction: two inline variants in `TransformerDecoder.forward` (`decoder.py:463-477`) plus `masked_softmax` alternative inside `attention()`; consider a mask atom.
10. Cache shape knowledge duplicated: `MHACausal.cache_shape` (mha.py:619-631) is used by `export/common.py:124-145` but `_write_concat`/`_forward_custom_sdpa`/`finalize_cache_geometry` re-spell shapes `(1,L,Hkv,D)` independently (`mha.py:610,615,693-699`).
11. `finalize_cache_geometry` re-allocation blocks (`mha.py:680-699`) are four copies of the same `torch.zeros(shape, dtype, device)` pattern.
12. `RMSNorm` constructed with kwargs at 3 places (`decoder.py:255-258`, `decoder.py:376`, `encoder.py:50,158`) with different argument subsets (the encoder drops `dtype/prescale/upcast`).
13. `activation_map` partially duplicates HF ACT2FN (outside blocks; guess).

## 7. Stale / unused / broken code
Evidence = grep over whole repo (`*.py`), see section 4.
- `MHAEncoderFusedProj` (`mha.py:407-479`): no importer outside `encoder.py`; the string `mha_encoder_fused_proj` appears in no indexing file or test. Dead branch (`encoder.py:67-74`). Also likely BROKEN (guess, not run): `qkv` is permuted to `(heads, seq, dim)` and passed to `attention()`, which expects `(seq, heads, dim)` and transposes internally; and `apply_rope_multihead` receives `(heads, seq, dim)` while its `cos` is `(seq,1,half)`.
- `TransformerEncoder` rope/mrope paths are BROKEN (verified signature): `encoder.py:240-245,251-256` call `precompute_rope_cos_sin_half(..., static=False)` and `precompute_mrope_cos_sin_half(..., static=False)`; `rope.py:173-177` has no `static` param -> TypeError. `precompute_mrope_cos_sin_half` (`rope.py:237-241`) in turn passes `position=` while the callee param is `start_pos` -> TypeError. No indexing file sets `position_embedding_type` other than `"none"` (bert/distilbert/vit indexing), so `rope`, `mrope`, `trained_absolute` are unreachable; the stale `static` mentions are in `rope.py:219` docstring too.
- `encoder.py:272` `max_pos = self.position_embeddings.size(1)` is the embed dim, should be `size(0)` (guess/bug); path unreachable anyway.
- Unreachable config keys (no setter anywhere: `enable_patchify_input`, `patch_projection`, `use_cls_token`, `patch_bias`, `mrope_section_dims`, `mrope_section_bases`, `mrope_positions`, `max_position_embeddings` for trained_absolute): `encoder.py:137-150,177-178,191,247-249`. The patchify path (`_image_to_patches`, `_prepare_input(preprocess_images=True)`, `patch_projection`, `cls_token`) is dead as far as this grep shows (`scripts/compression/compress_infer_vit.py:250` uses `TransformerEncoder` but with defaults). `encoder.forward` args `preprocess_images`, `position_ids`, `rope_pos` are only exercised with defaults.
- `precompute_mrope_inv_freqs`, `precompute_mrope_cos_sin_half` (`rope.py:113-152, 206-247`): only used by the dead encoder mrope branch.
- `AtomicSum.input_num` attribute (`atomic.py:8`): not read anywhere.
- `pruning_config_dims`: used only by `mlp.py`; `identity_page_table`/`num_kv_pages`: used only inside blocks (decoder/mha).
- `MHACausal._forward_custom_sdpa` param `attn_mask` unused by design (documented); `kvcache_start_index` in `paged_attention` eager body unused; `_write_mutable` ignores its `key_cache/value_cache` args.
- `MHABase.use_sdpa`/`MHABase.__init__(**kwargs)` swallow keys silently; `MHAEncoder.__init__` is a pure pass-through (`mha.py:367-368`).
- `transformersurgeon/__init__.py:4-6` re-exports `precompute_rope_inv_freqs` inside try/except; no other code uses `T.precompute_rope_inv_freqs` (grep shows only that file).
- Stale comments: `mha.py:510` ("mutable ... QNN-only" while XNNPACK exports also use it, per `scripts/executorch/xnnpack/*` and test_export_pipelines); `rope.py:160-168` docstring describes MRoPE for a plain function; `decoder.py:204-215` docstring placed after `super().__init__()` (not a docstring); `linear_compressed.py` docstring says `bias` default False but signature is True; `mlp.py` docstrings mention `input_dim/output_dim/bias` args that do not exist.
- `TransformerDecoderBlock.mha_type/mlp_type/norm_type/...` stored but only used in ctor (harmless).
- `EmbeddingCompressed` skip branch leaves object without `num_embeddings` (`__repr__` would AttributeError).
- `LinearCompressed` skip branch returns before `nn.Linear.__init__` (module has no `weight`, `in_features`, `rank`); `__repr__`/`cancel_lrd`/`init_lrd` on a skipped layer would fail (works in `forward`).
- `config.py` decoder ctor default `mlp_type="mlp_gated"` vs `from_source_config` fallback `"mlp"` (inconsistent defaults).

## 8. Test coverage (R/test)
Direct (import from blocks):
- `unit/test_add_batch_dim.py`: `MHACausal` x attn_impl (batched == unbatched), `_project_rope`, pruned-RoPE batch dim, `TransformerDecoder` output shape/parity, `finalize_rope_pruning`, `finalize_cache_geometry`, `CustomDecoderConfigCompress`.
- `unit/test_io_inplace_cache.py`: `kv_cache_update` (functional + writes), `cache_impl` `io_inplace` vs `io_scatter` token-by-token, multi-token prefill, `cache_shape`.
- `unit/test_paged_attention.py`: `KV_PAGE_SIZE`, `io_paged` vs `io_inplace` across pages, rejects rope-pruned, `attn_impl='paged'` requirement, ONNX backend rejection, TensorRT plugin eager parity (`paged_attention` via `export/tensorrt/edgellm_attention.py`).
- `unit/test_mha_causal_custom_sdpa.py`: `custom_sdpa` vs manual, requires mutable, rejects unknown attn_impl, rejects q/v dim mismatch (needs ExecuTorch custom ops; likely skipped without them).
- `unit/test_gqa_rope_pruning.py`: `_derive_rope_prune_pattern`/`build_rope_prune_projection` (5 tests), `MHAEncoder` forward with pruned RoPE, `MHACausal` pruned cache dims per `cache_impl`.
- `unit/test_rope_variants.py`: `ROPE_INV_FREQ`/`precompute_rope_inv_freqs_for` llama3 vs HF, `rope_parameters_from_config`, `resolve_rope_type`, converted llama equals HF with scaling.
- `unit/test_export_fixes.py`: `rope_theta_from_config`, converted decoder uses config eps/theta, `cache_shape` via manifest.
Indirect (via `convert_for_export` / `TransformerDecoder`): `unit/test_attention_semantics.py` (masked_softmax == additive mask, within-head pruning keeps score scale), `unit/test_block_pruning.py` (`finalize_head_pruning` through conversion), `unit/test_qnn_bias_split.py`, `unit/test_qnn_activation_only.py`, `unit/test_onnx_tensorrt_export.py`, `unit/test_known_bugs.py`, `e2e/test_model_families.py::test_convert_for_export` (causal families only: `is_causal=True` filter, test/_helpers/model_factory.py:149-154), `e2e/test_gqa_attention_pruning.py` (hard-prune decode, io cache sizes, `finalize_cache_geometry` guard via `test_unfinalized_pruned_cache_raises`), `e2e/test_export_pipelines.py` (xnnpack, xnnpack custom_sdpa, tensorrt, qnn; gated by backend availability).
`LinearCompressed`, `EmbeddingCompressed`, `Conv*Compressed`, `VCONBlock`, `effective_*`: covered only indirectly by compression/manager tests (test_model_families lrd/quant/prune, test_repeated_pattern, test_smoothquant etc.), no dedicated unit test of `blocks` leaf layers (`init_lrd/cancel_lrd` round trip, `skip` path, Embedding LRD forward).
Untested (no hit in test/): `MHAEncoderFusedProj`, `TransformerEncoder`/`TransformerEncoderBlock` (all of it: pre/post norm, layernorm, rope/mrope/absolute/patchify; `convert_for_export` on bert/vit/distilbert not exercised; only `scripts/compression/compress_infer_vit.py` uses it), `CustomEncoderConfigCompress`, `AtomicSum` (only transitively), `io_concat` equivalence (only in examples/scripts/test lists: grep shows `io_concat` once in test/unit), `precompute_mrope_*`, `apply_rope_multihead` direct unit, `RMSNorm` variants (`prescale=False`, `upcast=True` numerics; `rmsnorm_prescale/upcast` appear in 3 unit + 1 e2e files as option plumbing only), `MLP` (non-gated) in decoder path, `activation_map` entries other than silu, `norm_position="post"`, `attention()` with `kv_layout` variants beyond the io_inplace test, `MHABase.finalize_head_pruning` error branches, `VCONBlock` forward/`set_beta` errors, `pruning_dims` granularity>0 branch error messages, `_ensure_custom_sdpa_ops` failure path.

## 9. Proposed split into atoms
Principles: one structure per file; a registry dict per family; a single human-readable selection dict (replacing the flat config strings); no model names; backend knowledge only as an optional "capability" tag in the atom, with op definitions moved to `export/<backend>/ops`.

```
blocks/
  registry.py            # tiny: Registry class {name -> factory}, decorator @register("family","name"), build(family, name, **kw)
  spec.py                # BlockSpec dataclass / dict schema + validation + defaults; replaces CustomDecoder/EncoderConfigCompress selectors
  config.py              # PretrainedConfig subclasses keep only numeric/shape fields + `blocks_spec` dict (from_source_config shared helper)
  leaves/                # compressible leaf layers (kept; compression library swaps these)
    linear.py            # LinearCompressed  (+ lrd mixin)
    embedding.py         # EmbeddingCompressed
    conv.py              # Conv2d/3dCompressed
    vcon.py              # VCONBlock
    lrd_mixin.py         # init_lrd/cancel_lrd shared (dup #3)
    pruning_dims.py      # unchanged
  norm/
    rmsnorm.py           # key "rmsnorm"            (plain)
    rmsnorm_prescale.py  # key "rmsnorm_prescale"   (max-abs rescale)
    rmsnorm_upcast.py    # key "rmsnorm_upcast"     (fp32 normalize)
    layernorm.py         # key "layernorm"          (wrap nn.LayerNorm)
  mlp/
    plain.py             # key "plain"  (up, act, down)
    gated.py             # key "gated"  (gate*up -> down)  [share base builder]
    activations.py       # activation registry {"gelu","relu","silu",...}
  rope/
    freq_default.py      # key "default"
    freq_scaled_bands.py # key "scaled_bands" (today "llama3"; band-interpolated)
    freq_sectioned.py    # key "sectioned" (today mrope sections; fix signature bugs or delete)
    apply.py             # apply_rope_multihead (+ one shared fp32 rotate-half used by paged ref)
    prune_projection.py  # build_rope_prune_projection/_derive_* (move to compression-geometry helper; blocks keeps only consumption)
    params.py            # rope_parameters_from_config, rope_theta_from_config, resolve_rope_type (HF-config adaptation; "mrope"->"default" remap moves to the model indexing)
  attention/
    common.py            # MHABase (projections, head dims, score_scale, finalize_head/rope/cache hooks)
    kernels/
      manual.py          # attention()                       key "manual"
      masked_softmax.py  # mask mode as a kernel option      key "manual+masked_softmax" or option mask_mode
      sdpa.py            # F.scaled_dot_product_attention    key "sdpa"
      (custom_sdpa.py, paged.py live under export/<backend> or backend plugins, see below)
    masks/
      additive_row.py    # q_pos<k_pos row gather  (decoder.py:475)
      additive_window.py # multi-token (io_inplace) mask (decoder.py:463-472)
      none.py            # paged (kernel derives causality)
    kv_cache/
      mutable.py         # _write_mutable (module-state buffer)
      io_scatter.py      # index_put
      io_concat.py       # one-hot + where
      io_inplace.py      # BHSD + tsurgeon::kv_cache_update (op def in kv_cache/ops_inplace.py or export/)
      io_paged.py        # pool + page table helpers (num_kv_pages, identity_page_table, KV_PAGE_SIZE)
      layouts.py         # cache_shape per layout ("shd","bhsd","paged")
    variants/
      causal.py          # MHACausal assembled from (kv_cache atom, kernel atom, mask atom, add_batch_dim)
      encoder.py         # MHAEncoder
      fused_qkv.py       # MHAEncoderFusedProj (only if kept; fix or delete)
  layers/
    block.py             # ONE TransformerBlock(spec): norm_position pre|post, attn atom, mlp atom, norm atom, AtomicSum
    atomic.py            # AtomicSum
    stack.py             # TransformerStack: depth, final norm, position strategy, io threading (replaces TransformerDecoder / TransformerEncoder; differences become atoms: positions + input prep)
  positions/
    rope_table.py        # precomputed cos/sin buffers (decoder style)
    rope_dynamic.py      # per-call cos/sin (encoder style)  [needs fix]
    absolute.py          # trained absolute embeddings       [fix or drop]
    none.py
  inputs/
    patchify.py          # ViT-style image patchify + cls token (drop out of generic blocks into a model-input adapter)
  backend_ops/           # OPTIONAL, ideally moved to export/<backend>/ops/ and registered into the kernel/cache registries on import
    custom_sdpa.py       # torch.ops.llama.* kernel + `_ensure_custom_sdpa_ops`, fp32 own cache
    paged_attention.py   # tsurgeon::paged_attention op (Edge-LLM contract)
```

Selection dictionary (proposed key names, replaces `attn_type/mlp_type/norm_type/cache_impl/attn_impl/rope_type/norm_position/position_embedding_type` + `use_sdpa/masked_softmax/rmsnorm_prescale/rmsnorm_upcast/add_batch_dim`):
```python
blocks_spec = {
  "norm":      {"kind": "rmsnorm"|"rmsnorm_prescale"|"rmsnorm_upcast"|"layernorm", "eps": 1e-5},
  "norm_position": "pre" | "post",
  "attention": {
      "variant":   "causal" | "bidirectional" | "fused_qkv",
      "kernel":    "manual" | "sdpa" | "custom_sdpa" | "paged",
      "mask_mode": "additive" | "masked_softmax" | "none",
      "kv_cache":  "mutable" | "io_scatter" | "io_concat" | "io_inplace" | "io_paged" | "none",
      "max_cache_len": 2048,
      "batch_dim": False,
  },
  "mlp":       {"kind": "plain" | "gated", "activation": "silu"},
  "positions": {"kind": "rope" | "rope_sectioned" | "absolute" | "none",
                "freq": "default" | "scaled_bands" | ..., "params": {...}},
}
```
Compatibility rules to encode in the registry (today scattered ValueErrors, `mha.py:587-604`): `kernel=="paged"` <-> `kv_cache=="io_paged"`; `mask_mode=="masked_softmax"` needs `kernel=="manual"`; `kernel=="custom_sdpa"` needs `kv_cache=="mutable"`; `io_concat` needs seq_len 1; `io_paged` needs unpruned RoPE and equal q/k/v head dims.

Notes for the refactor lead:
- The cache atom determines forward return arity and decoder threading (mutable returns bare tensor, io_* return caches); a uniform contract (always return `(out, caches)` with `caches=None` for mutable) would remove the `if cache_impl == "mutable"` branches in `decoder.py:343-355,486-495` and let one `TransformerBlock` serve both stacks.
- Hooks `finalize_head_pruning/finalize_rope_pruning/finalize_cache_geometry` are called by `utils/convert.py` through `hasattr`; keep as a documented protocol on the attention atom.
- External contracts to preserve or migrate: fixed submodule names `blocks.{i}.attn|mlp|norm_in|norm_out`, projection names `q_proj,k_proj,v_proj,out_proj,up_proj,gate_proj,down_proj`, `LinearCompressed` attrs (`rank, linear_V, skip, block_prune_mask, rope_prune_mask`), `MHACausal.cache_shape(...)`, class-name strings `"MHACausal"` / `"TransformerDecoder"` matched in `export/executorch_exporters/qnn/qnn_export.py:460-478`, op names `tsurgeon::kv_cache_update` and `tsurgeon::paged_attention` (registered by import in `export/onnx/onnx_export.py:56` and `export/tensorrt/edgellm_attention.py:68`), `blocks/__init__` re-exports used by `utils/*`, `compression/*`, `export/export.py`.
- Move to `export/` (backend-specific): custom_sdpa kernel and its fp32 cache, paged op and Edge-LLM constants, kv_cache_update ONNX rationale; keep in `blocks/` only neutral ops with eager reference. The `LinearCompressed` rank-2 reshape and `RMSNorm` `x*x` spelling are exporter pattern-matching accommodations: either keep as documented neutral formulations or move to an exporter pre-pass.
- Compression decisions to externalize: `MLP*` should receive already-resolved dims (call `pruning_config_dims` in convert.py) so `blocks` stops reading `structured_pruning` config; `rope/prune_projection.py` and the `finalize_*` mask logic belong with `compression/` (the block should only consume a ready `rope_freq_proj` buffer).
