# Recap: transformersurgeon/models/

Paths below are relative to `transformersurgeon/` unless absolute. "UNUSED" = grep over whole repo (package, test, examples, scripts, benchmarks) found no reader.

## 1. Purpose and structure

`models/` holds one folder per supported HF family. Each folder = (a) an INDEXING dict (structure/paths/pruning/export metadata) and (b) a thin "define_" file that subclasses the HF Config/Model and a `CompressionSchemesManager`, injecting the indexing. NOTE: there are NO `config_*.py`, `modeling_*.py`, `manager_*.py` files; all three classes live together in `define_<name>.py`.

```
models/__init__.py                      42   safe star-import of 8 families (try/except swallow ALL exceptions -> silent missing family)
models/bert_c/        __init__ 1, indexing_bert_c.py 101, define_bert_c.py 38
models/distilbert_c/  __init__ 1, indexing_distilbert_c.py 104, define_distilbert_c.py 37
models/llama_c/       __init__ 3, indexing_llama_c.py 144, define_llama_c.py 43
models/modernbert_c/  __init__ 2, indexing_modernbert_c.py 73, define_modernbert_c.py 39
models/qwen2_c/       __init__ 2, indexing_qwen2_c.py 150, define_qwen2_c.py 39
models/qwen2_vl_c/    __init__ 2, indexing_qwen2_vl_c.py 143, define_qwen2_vl_c.py 62
models/qwen2_5_vl_c/  __init__ 2, indexing_qwen2_5_vl_c.py 186, define_qwen2_5_vl_c.py 64
models/vit_c/         __init__ 3, indexing_vit_c.py 123, define_vit_c.py 65, legacy_conversion.py 118
```
Total 1587 lines. Each folder `__init__` is `from .indexing_X import *; from .define_X import *`. Top-level `transformersurgeon/__init__.py:1` does `from .models import *`, so every `*Compress` class is exported as `transformersurgeon.<Name>` (tests use `T.Qwen2ForCausalLMCompress`).

Naming: indexing dict var `<FAMILY>_C_INDEXING`, `__all__ = [that]`. Top-level keys of the dict = "towers" (sub-models): `'text'` (llama, qwen2), `'bert'`, `'distilbert'`, `'modernbert'`, `'vit'`, `'vision'`+`'text'` (qwen2_vl, qwen2_5_vl). Tower name is arbitrary except: `export/export.py:37` hardcodes `"text"`, `export/export.py:61` `converted.get("text")`.

Each define_ file defines 3 things (see section 3): `XConfigCompress(XConfig)`, `XForYCompress(XForY)` with class attr `indexing = INDEXING` and `__init__` calling `replace_layers_upon_init` per tower, `XCompressionSchemesManager(CompressionSchemesManager)` whose `__init__(model)` = `super().__init__(model, INDEXING)`. `model.indexing` (class attr) is read by `utils/convert.py:284` and `export/export.py:37`.

## 2. Indexing dictionary schema (most important)

Structure: `INDEXING = { tower_name: { key: value, ... }, ... }`. Placeholders: `{block_index}`, `{path}` in `path_template`. "path" = dotted module path relative to a block (e.g. `self_attn.q_proj`). `flatten_index_paths(path_list)` (utils/utils.py:~10-37) turns a dict `{sub: [leaves]}` into `["sub.leaf",...]`; an empty list means the sub itself is a leaf (norm layers); a plain list passes through. ORDER of the flattened list matters: `layer_matching` must be aligned by position (convert.py:`_flatten_layer_matching` raises on length mismatch).

Models column: Q2=qwen2_c, L=llama_c, B=bert_c, D=distilbert_c, M=modernbert_c, V=vit_c, Q2VL-v/-t = qwen2_vl vision/text, Q25-v/-t = qwen2_5_vl vision/text.

### 2.1 HF-config attribute names
| key | type/meaning | used by | read at |
|---|---|---|---|
| `config_attr` | str; attr of root HF config that holds this tower's config ("" = root, `'vision_config'`, `'text_config'`). REQUIRED by readers (`indexing['config_attr']` KeyError if absent) | all | utils/modeling.py:26-27; utils/convert.py:300-303; utils/manager.py:345,422,1035-1039 (`_generate_schemes` reads it twice, line 1039 comment "done twice"); compression/smoothquant.py:59-60; (converted template) utils/convert.py:238 |
| `num_blocks_attr` | str; config attr = number of blocks (`num_hidden_layers`, `depth` for vision, `n_layers` distilbert). REQUIRED | all | utils/modeling.py:34; utils/configuration.py:44; utils/manager.py:347,424,1036; utils/convert.py:306; compression/smoothquant.py:61; blocks/config.py:111,143,255,281 |
| `embed_dim_attr` | str; hidden size attr (`hidden_size`; distilbert `dim`) | Q2,L,B,D,M,V,Q25-t (NOT Q2VL-t, NOT vision towers) | blocks/config.py:112,144,256,282 (only via from_source_config -> needs `structure`) |
| `num_heads_attr` | str; heads attr (`num_attention_heads`; distilbert `n_heads`) | same set | blocks/config.py:113,145,257,283 |
| `mlp_hidden_dim_attr` | str; intermediate size attr (distilbert `hidden_dim`) | same set | blocks/config.py:114,146,258,284 |
| `mlp_activation_attr` | str; act fn attr (`hidden_act`; distilbert `activation`; modernbert `hidden_activation`) | same set | blocks/config.py:115,147,259,285 |
| `kv_num_heads_attr` | str; GQA kv-heads attr (`num_key_value_heads`) | Q2, L, Q25-t only (optional; absent for encoders) | blocks/config.py:117-118,149-150 |

blocks/config.py:`from_source_config` raises KeyError if an *_attr key is missing -> `structure` towers MUST define embed/heads/mlp/act. Note bert/vit/distilbert/modernbert define them even though modernbert has no `structure`.
Also, anything in the HF config not "consumed" is passed through as kwargs to the block config (blocks/config.py:119-140, 262-280) incl. `rms_norm_eps`, `rope_theta`, `rope_parameters`, `max_position_embeddings`.

### 2.2 HF module layout (used for compression + weight mapping)
| key | type/meaning | models | read at |
|---|---|---|---|
| `path_list` | dict `{sub: [leaf,...]}` (or flat list). Defines every compressible/replaceable layer within a block. `[]` = norm layer (leaf). REQUIRED | all 10 towers | utils/modeling.py:32 (replace Linear->LinearCompressed); utils/configuration.py:47; utils/manager.py:1037; utils/convert.py:(`flatten_index_paths(indexing['path_list'])` 151,222,233,246,320,352,383); utils/cascade.py:(`_extract_path_entries(block_indexing["path_list"])` ~98-100, stage ordering) ; compression/structured_pruning.py:582 reads scheme.path_list |
| `path_template` | str with `{block_index}`,`{path}`: full HF path of a block member. REQUIRED. Values: `model.layers.{..}.{..}` (L,Q2,M), `model.language_model.layers...` (Q2VL-t,Q25-t), `model.visual.blocks...` (VL vision), `bert.encoder.layer...`, `distilbert.transformer.layer...`, `vit.layers...` | all | utils/modeling.py:31; utils/configuration.py:43; manager.py:344,421,1038; convert.py:223,382; smoothquant.py:62; cascade.py:~99 |
| `skip_connections` | list of `[A,B]` pairs: input of A -> output of B residual (block-level only; internal HF residuals not listed). Note distilbert uses `['attention','attention'],['ffn','ffn']`, bert/modernbert have none | Q2,L,D,V,Q2VL(both),Q25(both) | ONLY utils/cascade.py:98 (cascade calibration) |
| `calibration_groups` | dict `{sub: [[leaf,...],[leaf...]]}` (grouped by sub) OR flat list of lists of dotted names. Layers whose inputs are calibrated together | all except... (bert: flat list with ONE group `[q,k,v]`; others dict) | utils/manager.py:86-138 (`_get_calibration_groups_from_indexing`) |
| `smoothing` | dict `{producer_path: [consumer_paths]}` for SmoothQuant | Q2, L only | compression/smoothquant.py:55; manager.py:305,314 (docstring) |
| `extra_layers` | list of HF full paths (not per block) that are copied into the converted graph (final norm) | L, Q2, Q25-t, V | utils/convert.py:412-416 |
| `extra_layers_matching` | list of converted-graph names, aligned with `extra_layers`; default = last path component | L,Q2,Q25-t,V (always `["norm"]`) | utils/convert.py:413 |
| `final_norm` | str; HF full path of final norm (singleton sentinel in pruning) | L,Q2,Q2VL-t,Q25-t,V | utils/manager.py:`_resolve_final_norm_path`(~1066); compression/structured_pruning.py:545,565-578,599 |
| `final_layer` | str; HF full path of the head (`lm_head`, `classifier`, `model.visual.merger`) | all except Q2VL-vision | utils/manager.py:`_resolve_sentinel_path` (_SENTINEL_COMPRESSED_TYPE:30) ; structured_pruning.py:545; export/export.py:38,43,48 |
| `preprocessing` | str; HF full path of input embedding module | L,Q2,Q2VL-t,Q25-t,Q25-v,B,D,M,V | manager.py (sentinel); cascade.py:~104 (`block_indexing.get("preprocessing")` to compute block inputs); export/export.py:41 (text tower only) |
| `preprocessing_conv` | str; patch-embed Conv2d/3d path | V, Q2VL-v, Q25-v | manager.py:1066 resolve (Conv2dCompressed/Conv3dCompressed); structured_pruning.py:650-663 |
| `preprocessing_conv_extra_params` | list of param paths co-indexed with preprocessing_conv (cls_token, pos emb) | V only | manager.py:1073; scheme.py:~110-116; structured_pruning.py `_prune_extra_params` |
| `position_embeddings_source` | str; rotary module path whose output is fed to attention in cascade calibration | L,Q2,Q2VL-t,Q25-t | utils/cascade.py:835 |
| `position_embeddings_targets` | list of sub-block names that receive position_embeddings (`["self_attn"]`) | same | cascade.py:836 |
| `position_ids_ndim` | int 2 or 3 (3 = M-RoPE ids, Q25-t only; others 2) | L,Q2,Q2VL-t(2!),Q25-t(3) | cascade.py:841 |
| `no_cascade_calibration` | bool; raise if cascade mode requested | B, M | utils/cascade.py:58 |
| `qkv_paths` | list of fused-QKV paths | M, B[], D[q,k,v separate!], V[], Q2VL-v, Q2VL-t[] | **UNUSED** (no reader anywhere) |
| `pruning_supported` | list | M[], Q2VL-v[], Q2VL-t[], Q25-v[] | **UNUSED** |
| `lrd_supported` | list | Q2VL-t only | **UNUSED** |

### 2.3 `pruning` sub-dict (structured / block pruning)
Read through `block_indexing.get('pruning', {})` (manager.py:338,415,1067,1092,1119 -> passed as `pruning_indexing` to `CompressionScheme`, utils/scheme.py:~92).
| subkey | meaning | models | reader |
|---|---|---|---|
| `output_dependence` | dict `producer -> [consumers]`: when producer OUTPUT rows are pruned, consumers INPUT cols shrink. Special names: sentinels `preprocessing`, `preprocessing_conv`, `final_norm`, `final_layer` (singletons resolved off root paths, only on last block); `mlp.down_proj: ['input_layernorm','final_norm']` wraps to NEXT block | all (M: `{}`) | structured_pruning.py:581,608; manager.py:25 comment |
| `coupled_masks` | list of layer groups sharing one output mask within a block (q/k; gate/up) | all (M `[]`, Q2VL-v `[]`) | manager.py:339,416; abstract.py:132 (docs) |
| `coupled_masks_all` | groups sharing ONE mask across ALL blocks (residual-stream writers; incl. sentinel `preprocessing`/`preprocessing_conv`) | all (M `[]`) | manager.py:340,417 |
| `per_head_uniform` | layers needing uniform per-head ratio | all | **UNUSED** (comment in llama_c says "recorded only, not yet consumed") |
| `position_linked` | q/k rows tied to RoPE bands | L,Q2,Q2VL-t,Q25-t | utils/scheme.py:94 -> `scheme.position_linked` (structured_pruning.py:327,418,441; block_pruning.py:152) |
| `norm_layers` | norm names that are transparent / forwarded only | all except M | structured_pruning.py:577; manager.py:1067 |
| `block_pruning` | dict `{layer_path: {owner, block_size_attr, blocks_per_unit_attr, input_dependence}}` for head removal | L, Q2 only (q_proj) | compression/block_pruning.py:83-107,187; registry.py:100 |

### 2.4 Export-topology keys (needed to build `blocks/` model). Presence of `structure` is the gate: absent => `convert_for_export` warns and skips (convert.py:294-297).
| key | type/values | models having it | read at |
|---|---|---|---|
| `structure` | `'transformer_decoder'` / `'transformer_encoder'`; else NotImplementedError | Q2, L, Q25-t, B, D, V. MISSING: M, Q2VL (both), Q25-v | convert.py:294,308,347,375 |
| `attn_type` | `'mha_causal'` / `'mha_encoder'` (default mha_causal dec, mha_encoder enc) | same six | blocks/config.py:153,286 |
| `mlp_type` | `'mlp_gated'` / `'mlp'` | same | blocks/config.py:154,287 |
| `norm_type` | `'rmsnorm'` / `'layernorm'` | same | blocks/config.py:155,288 |
| `norm_position` | `'pre'`/`'post'` (encoder only; B,D post, V pre) | B,D,V | blocks/config.py:289; blocks/encoder.py:27 |
| `rope_type` | `'default'` / `'auto'` (L: auto -> llama3 or default) | L(auto),Q2(default),Q25-t(default) | blocks/config.py:165 -> blocks/rope.py `resolve_rope_type` |
| `layer_matching` | dict `{hf_sub: converted_name or [names]}` aligned with flattened `path_list`. Converted names are FIXED vocabulary: `norm_in`, `norm_out`, `attn.{q,k,v,out}_proj`, `mlp.{gate,up,down}_proj` (gated) / `mlp.{up,down}_proj` | six | convert.py:151-180 (also builds compression_config remap :222-226) |
| `bias_required` | dict `{converted_name: bool}` | six | convert.py:326,358,`_build_bias_required_config`; blocks/decoder.py:61, encoder.py:46 |
| `use_final_norm` | bool (encoder; default True). B,D False (post-norm), V True | B,D,V | convert.py:368 -> blocks/encoder.py:156 |
| `position_embedding_type` | `'none'` ('mrope' supported in encoder.py:176 but no model uses it) | B,D,V | convert.py:369 -> encoder.py:135 |

Observed differences in the export block per model (value table):
- Q2: bias qkv True, others False; rope default. L: all bias False; rope auto. Q25-t: same as Q2 + `position_ids_ndim` 3 + comment "text-only equal t/h/w ==> plain RoPE".
- B/D: all attn/mlp bias True, norm bias False(?! key `norm_in: False`); V same.
- Converted names for BERT: `attention.self`->`attn.q/k/v`, `attention.output`->`attn.out_proj`, `intermediate`->`mlp.up_proj`, `output`->`mlp.down_proj` (note layer_matching dict lists group 'attention.output.LayerNorm' as sub-key).

### 2.5 Hidden contracts (not keys but fixed vocabulary shared with blocks/)
- Converted graph module names are hardcoded in blocks/: block has `attn`, `mlp`, `norm_in`, `norm_out` (blocks/decoder.py:67-68,143,160; encoder.py:50-54,102-119); top module has `blocks` list (decoder.py:181) and `norm` (decoder.py:186; encoder.py:158-164). `path_template = "blocks.{i}.{path}"` in convert.py:240,339,356 and `attn` fetched by name at convert.py:~388.
- Decoder converted graph does NOT contain embedding or lm_head; export/export.py:35-52 takes them from the HF model via `indexing['text']['preprocessing'/'final_layer']`, falling back to `get_input_embeddings()`/`.lm_head`.
- Weight transfer is a 1:1 per-module `state_dict` copy (`_load_state_dict_dequantized`, convert.py ~30-130); no fused/split/permute support, which is why ModernBERT (`Wqkv`, `Wi`) and Qwen vision (`attn.qkv`) are unsupported.
- Pairing of `path_list` flattening order and `layer_matching` flattening order (dict iteration order) is positional and silent beyond length check.

### 2.6 Per-tower key matrix (quick reference)
- llama `text` & qwen2 `text`: identical key set (config attrs 7, path_list, skip_connections, smoothing, calibration_groups, pruning{output_dependence,coupled_masks,coupled_masks_all,per_head_uniform,position_linked,norm_layers,block_pruning}, path_template, extra_layers, final_norm, preprocessing, position_embeddings_*, position_ids_ndim, final_layer, structure..extra_layers_matching).
- qwen2_5_vl `text` = same minus smoothing and block_pruning, position_ids_ndim=3, plus nothing new. Q25 `vision`: no export keys, no embed/heads attrs, has `preprocessing`+`preprocessing_conv`+`final_layer=model.visual.merger`, `pruning_supported`.
- qwen2_vl `text`: only config_attr, num_blocks_attr (NO embed/heads/mlp attrs), path_list, skip_connections, calibration_groups, pruning (no block_pruning), path_template, preprocessing, final_layer, final_norm, position_embeddings_*, position_ids_ndim=2, qkv_paths, lrd_supported, pruning_supported. NO `structure`, NO `extra_layers`, NO `smoothing`, NO `kv_num_heads_attr`.
- qwen2_vl `vision`: no `preprocessing` / `final_layer`; has preprocessing_conv, qkv_paths.
- bert: calibration_groups flat list (dotted names), `no_cascade_calibration`, no skip_connections, no `final_norm`/`extra_layers` (post-norm), no `preprocessing_conv`.
- modernbert: minimal; `pruning` all empty; no structure, no skip_connections; `final_layer: "classifier"`.
- vit: only family with `preprocessing_conv_extra_params`, `norm_position='pre'` + `use_final_norm=True` + `extra_layers=["vit.layernorm"]`.

## 3. Per-model specifics (classes, signatures)

All Config classes follow: `__init__(self, **kwargs)`: pop/get `compression_config`, `super().__init__(**kwargs)`, `init_compressed_config(config_instance=self, indexing=INDEXING[tower], compression_config=...)` (utils/configuration.py:7; fills defaults for every layer path from COMPRESSION_REGISTRY, sets `config.compression_config`).
All Model classes: `config_class = XConfigCompress`, `indexing = INDEXING`, `__init__(config)`: `super().__init__(config)`, `replace_layers_upon_init(self, INDEXING[tower], config)` (utils/modeling.py:16; swaps nn.Linear->LinearCompressed, Embedding, Conv; asserts `type(old) is nn.Linear`).

- **llama_c** `define_llama_c.py`: `LlamaConfigCompress(LlamaConfig)` (pops `compression_config`), `LlamaForCausalLMCompress`, `LlamaCompressionSchemesManager`. Tower `'text'`. rope_type 'auto'. No special cases.
- **qwen2_c** `define_qwen2_c.py`: same as llama; `Qwen2ConfigCompress.__init__` uses `kwargs.get("compression_config")` (NOT pop -> compression_config also forwarded to HF `PretrainedConfig.__init__`; harmless but inconsistent). `import inspect` unused (line 1).
- **qwen2_vl_c** `define_qwen2_vl_c.py`: `Qwen2VLVisionConfigCompress(Qwen2VLVisionConfig)`, `Qwen2VLTextConfigCompress(Qwen2VLTextConfig)`, `Qwen2VLConfigCompress(Qwen2VLConfig)` with `sub_configs = {vision_config:..., text_config:...}` and a class-body hack `Qwen2VLTextConfigCompress.__init__.__signature__ = inspect.signature(Qwen2VLTextConfig.__init__)` (HF signature compatibility; executed in class body of the outer config, lines 33-39). Model calls `replace_layers_upon_init` twice (vision, text). Uses `kwargs.get` for compression_config.
- **qwen2_5_vl_c** `define_qwen2_5_vl_c.py`: same as qwen2_vl but uses `kwargs.pop`. DIFFERENCE/BUG-LIKE: model class declares `config: Qwen2_5_VLConfigCompress` (annotation, line 45) instead of `config_class = ...` (qwen2_vl line 43 uses `config_class`). Likely a bug: HF `from_pretrained` would use the parent's `config_class` (Qwen2_5_VLConfig) so sub-configs lose compression (guess; test factory builds the config explicitly so it is not caught).
- **vit_c** `define_vit_c.py`: `ViTConfigCompress(ViTConfig)` (uses get). `ViTForImageClassificationCompress.from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)` classmethod override: peeks checkpoint keys and injects `key_mapping` to translate legacy `encoder.layer.N.attention.attention.query` names to `layers.N.attention.q_proj` (see `legacy_conversion.py`: `build_legacy_vit_key_mapping(raw_keys)`, `peek_checkpoint_keys(path, **hub_kwargs)`, `_local_or_hub_file`, `_rename_legacy_key`, `_LEGACY_TO_MODERN_SUBSTRINGS`). This is the ONLY model-specific weight-mapping code (HF checkpoint->HF-compressed), and the only `from_pretrained` override. It exists because installed transformers' ViT uses flattened q_proj/layers naming (version-specific; GUESS: transformers >= 5).
- **bert_c / distilbert_c / modernbert_c** `define_*`: `XConfigCompress.__init__(self, compression=None, **kwargs)` -- `compression` param is UNUSED (dead arg), pops `compression_config`. Models are `...ForSequenceClassificationCompress` only (no MaskedLM / base model). `XCompressionSchemesManager`.
- Special structural knowledge only expressed in comments/indexing: ModernBERT layer 0 `attn_norm` is Identity (indexing_modernbert_c.py:21-24); alternating global/local rotary -> cascade unsupported (:50-58); fused Wqkv/Wi => no coupled pruning, no export.

## 4. Duplicated code and how to generalise

Define files are near-identical: strip names and each is ~30 lines.
- Config classes: llama_c/define_llama_c.py:12-19, qwen2_c/define_qwen2_c.py:12-19, bert_c/define_bert_c.py:11-19, distilbert_c/define_distilbert_c.py:11-19, modernbert_c/define_modernbert_c.py:11-19, vit_c/define_vit_c.py:11-18, qwen2_vl_c/define_qwen2_vl_c.py:16-39, qwen2_5_vl_c/define_qwen2_5_vl_c.py:16-41. Only differences: HF base class, indexing tower key, pop vs get, dead `compression=None` param.
- Model classes: llama 24-30, qwen2 22-28, bert 22-27, distilbert 22-27, modernbert 22-27, vit 21-28 (+from_pretrained), qwen2_vl 42-48, qwen2_5_vl 44-50. Difference: HF base class, list of towers passed to `replace_layers_upon_init`.
- Managers: 8 identical 3-line classes (llama 34-36, qwen2 30-32, bert 30-32, distilbert 30-32, modernbert 30-32, vit 31-33, qwen2_vl 51-53, qwen2_5_vl 53-55).
- Indexing: llama_c vs qwen2_c `text` differ ONLY in: comment line 118-121 (rope_type 'auto' vs 'default') and `bias_required` (qkv True for qwen2). Diff of the two files = ~6 lines. Qwen2.5-VL `text` is the same dict again (differs: `position_ids_ndim` 3, module root `model.language_model.*`, no smoothing/block_pruning) and Qwen2-VL `text` again with missing export keys. The ~60-line `pruning` block for decoders is copied 5 times (llama 47-103, qwen2 55-111, qwen2_vl text 87-130, qwen2_5_vl text 104-147) and the BERT-like post-norm one 2 times (bert 31-65, distilbert 33-67); vision `pruning` 3 times (qwen2_vl 37-62, qwen2_5_vl 42-70, vit 34-78). Long explanatory comments are duplicated too ("see llama_c for field semantics").
- legacy_conversion.py is the only truly unique helper.

Generalisation options (design suggestions, not facts):
1. One generic factory `make_compressed_family(hf_config_cls, hf_model_cls, indexing, towers=...)` returning `(ConfigCompress, ModelCompress, Manager)`; a family folder then reduces to its dict + one line of registration (HF class lookup via names, e.g. `transformers.models.<module>.modeling_<module>`). Multi-tower configs need `sub_configs` built from the tower names whose `config_attr != ""`; the signature hack and `class` attr vs annotation difference must be preserved/fixed centrally. Auto-derive class names (`Qwen2ForCausalLMCompress`) via `type()` so `from transformersurgeon import X` keeps working; `models/__init__.py` can iterate over a registry instead of 8 hard-coded names.
2. Share dict fragments: build indexing from reusable presets (e.g. `DECODER_RMS_GATED_PRUNING`, `ENCODER_POSTNORM_PRUNING`) with per-model overrides (module root path, rope_type, bias_required, attn/out names). With plain-python dicts (matching the "dictionary format" requirement) this can be `{**BASE_DECODER, 'rope_type': 'auto', ...}`.
3. Keep `legacy_conversion` as a per-model optional hook (e.g. indexing key `'checkpoint_key_mapping'` or a models/ registry hook) so `from_pretrained` override disappears from generic code.
4. Drop dead args/keys (see section 6).

## 5. Model-specific knowledge OUTSIDE models/ (grep qwen|llama|bert|vit|modernbert|distilbert|model_type|mrope|Qwen2 ... over package)

Executable (matters for "adding a model touches only models/"):
- export/export.py:37 `getattr(model, "indexing", {}).get("text")` and :61 `converted.get("text")`; :32 `hasattr(model,"lm_head")` + `get_input_embeddings()` fallback: export assumes a causal LM with tower named `"text"` and `lm_head`; encoders (bert/vit/distilbert) are not exportable through `export_to_backend` (only through `convert_for_export`). Fix: indexing-level `export_tower` key.
- utils/convert.py:240 `_CONVERTED_BLOCK_INDEXING_TEMPLATE` (num_blocks_attr "num_hidden_layers", path_template "blocks.{block_index}.{path}"); :339/:356 `"blocks.{i}.{path}"`; :388 `path_template.format(i=i, path="attn")` (assumes converted block submodule `attn`); :410-418 extra_layers default name logic; :294 skip when no `structure`; :308/:347 branch structure -> `TransformerDecoder`/`TransformerEncoder`; :368-369 encoder args; :331-345 decoder options (only decoders get rmsnorm_*, cache options).
- blocks/config.py:11 `model_type = "transformersurgeon-custom-decoder"`, :175 `"...custom-encoder"`; defaults (:153-155, 286-289) `mha_causal/mlp/rmsnorm` and `mha_encoder/mlp/layernorm`; passthrough kwargs set (:119-140,262-280).
- blocks/rope.py:16-20 legacy `"mrope"` Qwen2-VL marker rewritten to default (HF does the same); :46-74 `_inv_freq_llama3` and ROPE_INV_FREQ registry (`'default'`, `'llama3'`); :82-100 `resolve_rope_type` (handles indexing `'auto'`, raises on scaled configs, e.g. YaRN); rope.py:113-229 `precompute_mrope_*` (M-RoPE, used by encoder.py:9-10,176-186,247-251 via config attrs `mrope_section_dims`, `mrope_section_bases`, `mrope_positions`). NOTE: decoder export of Qwen2.5-VL text uses plain RoPE (text-only equal t/h/w).
- blocks/mha.py:407 `MHAEncoderFusedProj` is selected by indexing `attn_type: 'mha_encoder_fused_proj'` (blocks/encoder.py:67-68); no model currently uses that value (ModernBERT/Qwen-vision would need it plus a fused-weight mapping). blocks/mlp.py:94 `MLPGated` ('Qwen-style') selected by `mlp_type`.
- blocks/mha.py `torch.ops.llama.custom_sdpa / update_cache` (:349,487-498,600-772,876): ExecuTorch "llama" op namespace, not model-specific.
- export/executorch_exporters/qnn/qnn_export.py:676-729 `llama.fallback.default` (ExecuTorch sharding op), not model-specific. qnn_export.py:248-274 comments on Qwen2.5-0.5B measured behaviour.
- compression: structured_pruning.py:20-31, manager.py:24-34 `_SENTINEL_COMPRESSED_TYPE` ("preprocessing","final_layer","preprocessing_conv" -> compressed class): generic sentinel names defined in indexing vocabulary, hardcoded in code.
- utils/modeling.py:94-150 conv patch-embed (Conv2d vs Conv3d) and "BERT-style embeddings module not supported" (:150) handling; embedding replacement only for plain nn.Embedding.
- hf/hf_export.py:6-14 `auto_map` `AutoConfig`/`AutoModel`/`AutoModelForConditionalGeneration` by class name (no per-model branching; line 14 conditional for VL presumably). No `model_type` checks anywhere.
- Hardcoded numeric/model constants in comments only (no code branching): blocks/norm.py:52 (Qwen2.5 activation ~5e3), blocks/decoder.py:225 (Qwen2.5 k biases ~300 => finite mask -1e4 insufficient), export/tensorrt/attention_passes.py:6 (Qwen2.5-1.5B k bias ~317), export/tensorrt/tensorrt_export.py:68, precision_passes.py:10, blocks/embedding_compressed.py:21-22, export/common.py:745,876, qnn_export.py:264-274. These document why generic mechanisms exist; no `if "qwen"` branches found. Grep for `model_type`/`"qwen`/`isinstance(..Qwen` in package outside models/ returned nothing executable.
- utils/manager.py:914-915 docstring example uses `Qwen2ForCausalLMCompress`.
- Outside the package: `vlm-modules-importance/indexing/qwen_indexing.py` (own indexing copy, unrelated lib?), `scripts/compression/compress_infer_qwen.py|_vit.py`, `scripts/summaries/extract_summaries_qwen.py`, examples/01..04 import family classes.

## 6. Stale / unused code (verified by grep)
- Keys with no reader: `qkv_paths` (M,B,D,V,Q2VL both), `pruning_supported` (M,Q2VL both,Q25-v), `lrd_supported` (Q2VL-t), `pruning.per_head_uniform` (all). `preprocessing` in Q25-v/others is read only by cascade/export, fine.
- `compression=None` constructor arg: bert_c/define_bert_c.py:12, distilbert_c/define_distilbert_c.py:12, modernbert_c/define_modernbert_c.py:12 (unused).
- `import inspect` unused: qwen2_c/define_qwen2_c.py:1.
- Stale comments: qwen2_c/__init__.py:1 says "Qwen2-VL compressed models" (wrong); qwen2_vl_c/__init__.py:1 correct.
- `legacy_conversion.py` is only imported by vit_c/define_vit_c.py; no tests exercise it (grep test/ for legacy|key_mapping: none).
- Indexing for bert has `'qkv_paths': []`, D has real q/k/v list but not read.
- Inconsistent: only L/Q2 have `smoothing` and `block_pruning`; Q25-t lacks them (feature gap, not stale). Qwen2-VL text lacks all export keys and `extra_layers` (so even adding `structure` the final norm would not be copied: decoder `norm` stays random-init; GUESS-level inference from convert.py:412).
- `utils/manager.py:1035-1039` double read of `config_attr` (author's own comment).
- ViT test criteria `lrd_criteria="attention.attention.query"`, `mlp_prune_criteria="intermediate.dense"` (test/_helpers/model_factory.py:~172-174) are legacy names that do NOT match indexing paths `attention.q_proj`/`mlp.fc1` (GUESS: criteria silently match nothing, so LRD/pruning tests on vit are vacuous; not verified by running).
- `models/__init__.py:_safe_star_import` swallows every Exception silently (hides broken families).

## 7. Test coverage per model (test/)
Central: `test/_helpers/model_factory.py` builds tiny random models for all 8 families via `FAMILIES` (`_qwen2,_llama,_bert,_modernbert,_distilbert,_vit,_make_vl(...)`), `ALL_FAMILY_NAMES`, `CAUSAL_FAMILY_NAMES`. `e2e/test_model_families.py` parametrises ALL 8 families over: instantiate+forward, build manager, LRD soft apply+restore, int8 quant apply, unstructured pruning, structured hard MLP prune (skips modernbert, `struct_prune_supported=False`; VL families only get MLP gate/up), and `test_convert_for_export` ONLY for `is_causal` families = qwen2, llama (not qwen2_5_vl, not any encoder).
Per-family beyond that:
- qwen2: by far the most (test_known_bugs, test_cascade_calibration, test_smoothquant[qwen2,llama], test_qnn_bias_split, test_qnn_activation_only, test_extract_summaries, test_gqa_attention_pruning, test_block_pruning, test_lpbq_quantization, test_attention_semantics, test_paged_attention, test_export_fixes, test_onnx_tensorrt_export, e2e/test_export_pipelines).
- llama: test_rope_variants (llama3 vs default), test_block_pruning, test_smoothquant.
- bert: test_cascade_calibration:38 (cascade-calibration works), test_known_bugs:25.
- modernbert: test_cascade_calibration:58 (cascade refused via `no_cascade_calibration`).
- qwen2_vl: test_known_bugs:118,130.
- distilbert, vit, qwen2_5_vl: only the generic family tests (no dedicated test); no encoder export conversion test (convert_for_export encoder branch untested), no test of `from_pretrained`/legacy ViT mapping, no test of `from_pretrained` for any VL config (`config:` annotation bug above untested), no test of `layer_matching`/`bias_required` correctness against HF numerics except via qwen2/llama convert tests.
