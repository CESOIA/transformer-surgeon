"""Which device operation implements each block, for each export backend.

``BACKEND_OPERATIONS`` links the selectable atoms of ``blocks/`` (see
``blocks/catalog.py``) and the backend-specific lowering choices to what runs on
the device. It is organised by *block type*; for every backend it gives:

- ``default``: the implementation used when the caller chooses nothing;
- ``options``: every implementation known to work on that backend, each with a
  short description of the resulting device operation;
- ``requires_cache`` (optional, ``kv_precision`` only): the KV-cache atom each
  listed precision needs on that backend (see :func:`required_cache`);
- ``incompatible`` (optional, ``kv_precision`` only): precisions that cannot store
  the two caches of a ``(key, value)`` pair together on that backend;
- ``prefill`` (optional, ``kv_cache`` only): the caches with which that backend's
  exporter can export a whole-prompt graph (``max_input_len > 1``). Each listed cache
  must be able to write several tokens per call (``multi_token``), but not every such
  cache is exportable that way (the paged pool is not), so the list is per backend.
  A backend without the key exports no whole-prompt graph. Checked by :func:`check_prefill`;
- ``requires`` (optional): what every non-default option of a ``config_field``
  choice needs, ``{"config": {field: value}, "cache_impl": [...]}``, checked by
  :func:`check_config_requirements` before any tracing.

A block type with ``"per_cache": True`` takes one option for both caches or a
``(key, value)`` pair of options (each part must be listed).

Each block type also says how the choice reaches the code:

- ``"convert_option"``: a key of ``convert_options`` (the atoms used to build
  the export-ready model, for example ``cache_impl``);
- ``"config_field"``: a field of the backend's export configuration (for
  example the XNNPACK ``quant_path``);
- ``"model_key"``: the choice is not the exporter's -- the model dictionary
  names the operation under this key (``None``: one implementation only). These
  entries have ``default: None`` (or their single option) and only document
  which device operation each atom becomes.

The exporters read this dictionary: missing convert options and configuration
fields take the backend default from here, and a choice that is not listed for
the backend is rejected before any tracing starts.
"""

from ..blocks import ATTENTION_KERNELS, KV_CACHES, KV_PRECISIONS, canonical_kv_precision, kv_precision_pair

BACKENDS = ("xnnpack", "qnn", "onnx", "tensorrt")

BACKEND_OPERATIONS = {
    # ------------------------------------------------------------------ attention
    "kv_cache": {
        "description": "Storage and update of past keys/values (blocks/attention/kv_cache).",
        "convert_option": "cache_impl",
        "backends": {
            "xnnpack": {"default": "mutable", "options": {
                "mutable": "ExecuTorch mutable buffer updated in place (index_copy_).",
                "io_scatter": "Cache as graph input/output, functional index_put.",
                "io_concat": "Cache as graph input/output, where() on a one-hot position mask.",
            }},
            "qnn": {"default": "mutable", "options": {
                "mutable": "HTP mutable buffer updated in place (index_copy_).",
                "io_scatter": "Cache as graph input/output, QNN ScatterNd.",
                "io_concat": "Cache as graph input/output, QNN element-wise select.",
            }},
            "onnx": {"default": "io_inplace", "options": {
                "io_inplace": "ONNX TensorScatter (opset 24); consumers may update the buffer in place.",
                "io_rows": "Cache as a read-only graph input; the graph returns the new rows, which the consumer "
                           "writes at pos_id.",
                "io_scatter": "ONNX ScatterND.",
                "io_concat": "ONNX Where on a one-hot position mask.",
                "io_paged": "Only with a custom translation of tsurgeon::paged_attention "
                            "(the tensorrt backend provides one).",
            }, "prefill": ["io_inplace", "io_rows"]},
            "tensorrt": {"default": "io_inplace", "options": {
                "io_inplace": "TensorScatter executed in place by TensorRT (output aliased to input).",
                "io_rows": "Cache as a read-only engine input; the engine returns the new rows and the runner "
                           "writes them (the only cache for int8/int4/fp8/nvfp4 rows).",
                "io_scatter": "TensorRT ScatterND.",
                "io_concat": "TensorRT Select.",
                "io_paged": "Paged KV pool of the TensorRT Edge-LLM AttentionPlugin.",
            # io_paged writes several tokens per call (multi_token), but the ONNX lowering shared with onnx traces
            # the whole-prompt graph only with separate key/value caches, not with one paged pool per layer.
            }, "prefill": ["io_inplace", "io_rows"]},
        },
    },
    "kv_precision": {
        "description": "What one stored key/value row holds (blocks/attention/kv_precision). Integer rows are "
                       "uint8: symmetric codes plus one 2-byte scale per row (token, KV head); FP8 rows are "
                       "float8_e4m3fn with a unit scale; NVFP4 rows are FP4 (E2M1) with a second tensor of FP8 "
                       "block scales (one per 16 elements). All are decoded to float before attention; the attention "
                       "kernel never reads them natively. One option for both caches, or a (key, value) pair of "
                       "options to store them differently, e.g. ('int8', 'int4'). A backend entry may name the "
                       "cache a precision needs there (requires_cache).",
        "convert_option": "kv_precision",
        "per_cache": True,
        "backends": {
            "xnnpack": {"default": "full", "options": {
                "full": "Cache in the model dtype.",
                "int8": "uint8 mutable cache; encode/decode are element-wise float operations, partly outside "
                        "the XNNPACK delegate (portable ExecuTorch kernels).",
                "int4": "uint8 mutable cache with two codes per byte; encode/decode as for int8.",
            }},
            # HTP keeps the cache as an activation of the graph: an 8-bit cache comes from the QNN quantiser
            # (kv_cache_quantization / QNNExportConfig.kv_cache_bits), not from integer rows.
            "qnn": {"default": "full", "options": {"full": "Cache in the activation precision of the graph."}},
            "onnx": {"default": "full", "options": {
                "full": "Cache in the model dtype.",
                "int8": "uint8 cache I/O; encode/decode are standard element-wise ONNX operations.",
                "int4": "uint8 cache I/O with two codes per byte; encode/decode are standard ONNX operations.",
                "fp8": "float8e4m3fn cache I/O written by QuantizeLinear and read by DequantizeLinear (unit "
                       "scale); needs cache_impl='io_rows'.",
                # nvfp4 exports to valid ONNX (see the tensorrt entry) but ONNX Runtime 1.31 has no FLOAT4E2M1
                # DequantizeLinear, so it is offered on tensorrt only.
            }},
            # Checked with TensorRT 11.4: the KV-cache update layer (io_inplace) takes only a Float/Half/BFloat16
            # cache; uint8 tensors are refused inside the network ("does not support UINT8 types for intermediate
            # tensors") and FP8 tensors may only be produced by a quantize layer and read by a dequantize layer, so
            # a stored row can never be scattered or selected in the engine. With io_rows the engine only reads the
            # cache and returns the new rows (uint8 or FP8 graph outputs), which the runner writes.
            "tensorrt": {"default": "full", "options": {
                "full": "Cache in the model dtype.",
                "int8": "uint8 cache I/O; encode/decode are element-wise operations around the attention.",
                "int4": "uint8 cache I/O with two codes per byte; encode/decode as for int8.",
                "fp8": "FP8 cache I/O: a QuantizeLinear writes the rows, a DequantizeLinear reads the cache "
                       "(checked on compute capability 8.9; older GPUs, e.g. Jetson Orin, may refuse FP8). With "
                       "INT8 weights in the same engine it builds only on Blackwell or newer GPUs.",
                "nvfp4": "NVFP4 cache I/O: FP4 rows plus an FP8 block-scale tensor per cache (unit per-tensor "
                         "scale); QuantizeLinear writes both, a double DequantizeLinear reads them. Checked on compute "
                         "capability 8.9, where TensorRT warns that FP4 is not accelerated (Blackwell has FP4 "
                         "hardware). The scales are FP8, so the INT8-weights rule of fp8 applies. Not in a "
                         "(key, value) pair with int8 or int4: TensorRT 11.4 crashes building that engine.",
            }, "requires_cache": {"int8": "io_rows", "int4": "io_rows", "fp8": "io_rows", "nvfp4": "io_rows"},
                # Checked with TensorRT 11.4 (RTX 4070): the builder crashes (segmentation fault) on an engine with
                # an NVFP4 cache next to an integer (uint8) one; NVFP4 next to full or FP8 caches builds.
                "incompatible": [("nvfp4", "int8"), ("nvfp4", "int4")]},
        },
    },
    "kv_cache_quantization": {
        "description": "Bits of a KV cache that the backend's own quantiser keeps as a fixed-point activation "
                       "(the graph stores it in the model dtype, kv_precision='full').",
        "config_field": {"qnn": "kv_cache_bits"},
        "backends": {
            "qnn": {"default": 16, "options": {
                16: "Cache at the activation precision of the graph (16-bit with quant_scope='all').",
                8: "The cache writes get an 8-bit activation configuration (8a8w): an 8-bit fixed-point cache. "
                   "Needs quant_scope='all', float_attention_core=False and a cache written by an index "
                   "operation (cache_impl 'mutable' or 'io_scatter').",
                (16, 8): "(key bits, value bits): only the value-cache writes get the 8a8w configuration; the key "
                         "cache stays at 16 bits. A few key channels with large biases set a per-tensor 8-bit key "
                         "step that leaves the other channels a few codes. Same requirements as 8.",
                (8, 16): "(key bits, value bits): only the key-cache writes get the 8a8w configuration; the value "
                         "cache stays at 16 bits. Same requirements as 8.",
            }, "requires": {"config": {"quant_scope": "all", "float_attention_core": False},
                            "cache_impl": ["mutable", "io_scatter"]}},
            **{backend: {"default": "kv_precision", "options": {
                "kv_precision": "Not chosen by a quantiser: the convert option kv_precision decides the cache.",
            }} for backend in ("xnnpack", "onnx", "tensorrt")},
        },
    },
    "attention_kernel": {
        "description": "How attention scores and the weighted sum are computed (blocks/attention/kernels).",
        "convert_option": "attn_impl",
        "backends": {
            "xnnpack": {"default": "manual", "options": {
                "manual": "MatMul + Softmax + MatMul, delegated to XNNPACK.",
                "sdpa": "aten.scaled_dot_product_attention, decomposed at lowering.",
                "custom_sdpa": "ExecuTorch llama::custom_sdpa + llama::update_cache CPU kernels "
                               "(scans only the filled cache; needs the mutable cache).",
            }},
            "qnn": {"default": "manual", "options": {
                "manual": "HTP MatMul/Softmax/MatMul with grouped heads broadcast (no cache copy).",
                "sdpa": "aten.scaled_dot_product_attention, decomposed at lowering.",
            }},
            "onnx": {"default": "manual", "options": {
                "manual": "ONNX MatMul/Softmax/MatMul.",
                "sdpa": "Decomposed scaled-dot-product attention.",
                "paged": "Only with a custom translation of tsurgeon::paged_attention.",
            }},
            "tensorrt": {"default": "manual", "options": {
                "manual": "MatMul/Softmax/MatMul; scores are cast to float32 by the fp32_attention_scores pass.",
                "sdpa": "Decomposed scaled-dot-product attention.",
                "paged": "TensorRT Edge-LLM AttentionPlugin (XQA decode kernels).",
            }},
        },
    },
    "masked_softmax": {
        "description": "Masking by replacing masked scores instead of adding a penalty (manual kernel only).",
        "convert_option": "masked_softmax",
        "backends": {
            backend: {"default": False, "options": {
                False: "Additive mask (-65504 on masked positions).",
                True: "Masked scores replaced by (row minimum - 20); keeps quantised score ranges tight.",
            }} for backend in BACKENDS
        },
    },
    # -------------------------------------------------------------- normalisation
    "rmsnorm": {
        "description": "Implementation of RMS normalisation (blocks/norm).",
        "convert_option": "rmsnorm_impl",
        "backends": {
            backend: {"default": "rmsnorm_prescaled", "options": {
                "rmsnorm_prescaled": "Max-abs rescale then RMS normalisation (float16-overflow safe).",
                "rmsnorm": "Plain RMS normalisation"
                           + (" (fused into one HTP RmsNorm operation)." if backend == "qnn" else "."),
                "rmsnorm_upcast": "Normalisation computed in float32, cast back.",
            }} for backend in BACKENDS
        },
    },
    # ------------------------------------------------------------------ graph I/O
    "batch_dim": {
        "description": "Leading size-1 batch dimension on the graph input and hidden states.",
        "convert_option": "add_batch_dim",
        "backends": {
            "xnnpack": {"default": False, "options": {False: "input_ids is (seq,).", True: "input_ids is (1, seq)."}},
            "qnn": {"default": False, "options": {False: "input_ids is (seq,).", True: "input_ids is (1, seq)."}},
            "onnx": {"default": False, "options": {False: "input_ids is (seq,)."}},
            "tensorrt": {"default": False, "options": {False: "input_ids is (seq,)."}},
        },
    },
    # ----------------------------------------------------------- compressed linear
    "linear_quantization": {
        "description": "How quantised compressed linear layers (blocks/layers/linear_compressed.py) are lowered.",
        "config_field": {"xnnpack": "quant_path", "qnn": "quant_scope", "tensorrt": "int4_backend"},
        "backends": {
            "xnnpack": {"default": "pt2e", "options": {
                "pt2e": "PT2E quantise/dequantise pairs from a linear-only quantiser; "
                        "per-tensor or per-channel weights.",
                "torchao": "torchao IntxUnpackedToInt8Tensor weights with dynamic int8 per-token activations "
                           "(8daXw), dynamically-quantised XNNPACK partitioner; supports per-group scales.",
            }},
            "qnn": {"default": "linear", "options": {
                "linear": "Only compressed linears are quantised (W4 -> 16a4w, W8 -> 8a8w); every other "
                          "operation stays float16.",
                "all": "Every operation gets 16-bit activations (W4 -> 16a4w, W8 -> 16a8w, float linears "
                       "-> 16a16w): one HTP partition.",
            }},
            "onnx": {"default": "qdq", "options": {
                "qdq": "DequantizeLinear on integer weight initialisers (+ QuantizeLinear/DequantizeLinear "
                       "for static int8 activations).",
            }},
            "tensorrt": {"default": "dequantize", "options": {
                "dequantize": "DequantizeLinear -> Gemm (no extra dependency).",
                "edgellm_plugin": "TensorRT Edge-LLM Int4GroupwiseGemmPluginV2 for eligible INT4 layers.",
            }},
        },
    },
    "linear_lowering": {
        "description": "Operation that carries a linear layer on the device.",
        "config_field": {"qnn": "convert_linear_to_conv2d"},
        "backends": {
            "xnnpack": {"default": "fully_connected", "options": {"fully_connected": "XNNPACK fully-connected."}},
            "qnn": {"default": False, "options": {
                False: "QNN FullyConnected.",
                True: "1x1 Conv2d (forced when weights are block-quantised, LPBQ).",
            }},
            "onnx": {"default": "gemm", "options": {"gemm": "ONNX MatMul/Gemm."}},
            "tensorrt": {"default": "gemm", "options": {"gemm": "TensorRT matrix multiply."}},
        },
    },
    # ---------------------------------------------- operations chosen by the model
    "mlp": {
        "description": "Feed-forward block (blocks/mlp); the model dictionary picks plain or gated.",
        "model_key": "mlp_type",
        "backends": {
            backend: {"default": None, "options": {
                "mlp": f"Two {op} with the activation in between.",
                "mlp_gated": f"Three {op}; activation of the gate times the up projection.",
                "mlp_gated_fused": f"Two {op}; the first is split into gate and up halves.",
            }} for backend, op in (("xnnpack", "XNNPACK fully-connected layers"),
                                   ("qnn", "QNN FullyConnected (or 1x1 Conv2d) operations"),
                                   ("onnx", "ONNX MatMul/Gemm nodes"),
                                   ("tensorrt", "TensorRT matrix multiplies"))
        },
    },
    "rope_frequencies": {
        "description": "Rotary-embedding frequencies (blocks/positions); the model dictionary picks the variant.",
        "model_key": "rope_type",
        "backends": {
            backend: {"default": None, "options": {
                name: f"{text} cos/sin tables computed at conversion and stored as constants; "
                      "the rotation is element-wise multiply/add on the device."
                for name, text in (("default", "Plain"), ("llama3", "Band-scaled"))
            }} for backend in BACKENDS
        },
    },
    "embedding": {
        "description": "Input token embedding (blocks/layers/embedding_compressed.py).",
        "model_key": None,
        "backends": {
            backend: {"default": "embedding", "options": {"embedding": f"{op} on the embedding table."}}
            for backend, op in (("xnnpack", "Embedding lookup (ExecuTorch CPU kernel)"), ("qnn", "QNN Gather"),
                                ("onnx", "ONNX Gather"), ("tensorrt", "TensorRT Gather"))
        },
    },
    "output_head": {
        "description": "Projection of the last hidden state to the vocabulary (a compressed linear layer).",
        "model_key": None,
        "backends": {
            backend: {"default": "linear", "options": {"linear": "Same lowering as every other linear "
                                                                 "layer (see linear_lowering)."}}
            for backend in BACKENDS
        },
    },
}


def _entry(block_type, backend):
    try:
        return BACKEND_OPERATIONS[block_type]["backends"][backend]
    except KeyError:
        raise ValueError(f"No {block_type!r} entry for backend {backend!r}.") from None


def default_for(backend, block_type):
    """Default implementation of ``block_type`` on ``backend``."""
    return _entry(block_type, backend)["default"]


def options_for(backend, block_type):
    """Supported implementations of ``block_type`` on ``backend`` (``{name: device operation}``)."""
    return _entry(block_type, backend)["options"]


def config_default(backend, field_name):
    """Default value of a backend configuration field listed under ``config_field``."""
    for block_type, spec in BACKEND_OPERATIONS.items():
        if spec.get("config_field", {}).get(backend) == field_name:
            return default_for(backend, block_type)
    raise KeyError(f"{field_name!r} is not an operation choice of backend {backend!r}.")


def required_cache(backend, kv_precision):
    """The KV-cache atom ``kv_precision`` (a name, or a ``(key, value)`` pair) needs on ``backend``
    (``None``: any listed cache): each precision atom's own requirement or the backend's
    (``requires_cache`` of its ``kv_precision`` entry)."""
    needed = set()
    for name in kv_precision_pair(kv_precision):
        atom = KV_PRECISIONS.get(name)
        needed.add((atom.requires_cache if atom is not None else None)
                   or _entry("kv_precision", backend).get("requires_cache", {}).get(name))
    needed.discard(None)
    if len(needed) > 1:
        raise ValueError(f"kv_precision={kv_precision!r} needs different caches on the {backend} backend: "
                         f"{sorted(needed)}.")
    return needed.pop() if needed else None


def check_kv_combination(backend, cache_impl, kv_precision):
    """Raise ``ValueError`` if ``kv_precision`` cannot be stored in ``cache_impl`` on ``backend``, or if the
    two precisions of a ``(key, value)`` pair cannot be stored together there."""
    parts = set(kv_precision_pair(kv_precision))
    for pair in _entry("kv_precision", backend).get("incompatible", ()):
        if set(pair) == parts:
            raise ValueError(f"kv_precision={kv_precision!r} is not supported by the {backend} backend: {pair[0]!r} "
                             f"and {pair[1]!r} caches cannot be in one {backend} graph. "
                             "See transformersurgeon/export/operations.py.")
    needed = required_cache(backend, kv_precision)
    if needed not in (None, cache_impl):
        raise ValueError(f"kv_precision={kv_precision!r} needs cache_impl={needed!r} on the {backend} backend "
                         f"(got {cache_impl!r}). See transformersurgeon/export/operations.py.")


def _per_cache_parts(value):
    """The options of a ``per_cache`` choice: one option (both caches) or a ``(key, value)`` pair."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return tuple(value)
    return (value,)


def check_choice(backend, block_type, value):
    """Raise ``ValueError`` if ``value`` is not a supported ``block_type`` implementation on ``backend``
    (a ``(key, value)`` KV-precision pair: if either part is not)."""
    options = options_for(backend, block_type)
    parts = _per_cache_parts(value) if BACKEND_OPERATIONS[block_type].get("per_cache") else (value,)
    for part in parts:
        if part not in options:
            raise ValueError(
                f"{block_type} implementation {part!r} is not supported by the {backend} backend; "
                f"supported: {list(options)} (default {default_for(backend, block_type)!r}). "
                "See transformersurgeon/export/operations.py."
            )


def check_config_choices(backend, config):
    """Validate every configuration field of ``config`` that selects an operation."""
    for block_type, spec in BACKEND_OPERATIONS.items():
        field_name = spec.get("config_field", {}).get(backend)
        if field_name is not None and hasattr(config, field_name):
            check_choice(backend, block_type, getattr(config, field_name))


def check_prefill(backend, config, convert_options):
    """Raise ``ValueError`` if ``config`` asks for a whole-prompt graph (``max_input_len > 1``) with a cache that
    the backend's exporter cannot export that way (``prefill`` of its ``kv_cache`` entry)."""
    if int(getattr(config, "max_input_len", 1) or 1) <= 1:
        return
    allowed = _entry("kv_cache", backend).get("prefill", [])
    cache = convert_options.get("cache_impl")
    if cache not in allowed:
        raise ValueError(f"max_input_len > 1 (prompt prefill) on the {backend} backend requires cache_impl in "
                         f"{allowed} (got {cache!r}). See transformersurgeon/export/operations.py.")


def check_config_requirements(backend, config, convert_options=None):
    """Raise ``ValueError`` if a non-default ``config_field`` choice lacks what its backend entry
    ``requires``: other configuration values, and (with ``convert_options``, resolved) a cache atom."""
    for block_type, spec in BACKEND_OPERATIONS.items():
        field_name = spec.get("config_field", {}).get(backend)
        if field_name is None or not hasattr(config, field_name):
            continue
        entry = _entry(block_type, backend)
        value, requires = getattr(config, field_name), entry.get("requires")
        if not requires or value == entry["default"]:
            continue
        what = f"{type(config).__name__}.{field_name}={value!r}"
        needed = requires.get("config", {})
        if any(getattr(config, name, None) != want for name, want in needed.items()):
            raise ValueError(f"{what} requires " + " and ".join(f"{n}={w!r}" for n, w in needed.items())
                             + ". See transformersurgeon/export/operations.py.")
        caches = requires.get("cache_impl")
        if caches and convert_options is not None and convert_options.get("cache_impl") not in caches:
            raise ValueError(f"{what} requires cache_impl in {caches} (got {convert_options.get('cache_impl')!r}). "
                             "See transformersurgeon/export/operations.py.")


def resolve_convert_options(backend, options=None):
    """Complete the caller's ``convert_options`` with the backend defaults and validate them.

    The older aliases are honoured first: ``use_sdpa=True`` means
    ``attn_impl="sdpa"``; ``rmsnorm_prescale``/``rmsnorm_upcast`` select an
    ``rmsnorm_impl``. When the attention kernel is not given, the backend default
    is used unless it is incompatible with the chosen cache, in which case the
    first compatible supported kernel is taken. Likewise, when the cache is not
    given, a ``kv_precision`` that needs a particular cache on the backend selects it.
    """
    from .convert import resolve_convert_options as resolve_generic

    user = dict(options or {})
    if user.get("use_sdpa") and not user.get("attn_impl"):
        user["attn_impl"] = "sdpa"
    resolved = resolve_generic(user)
    by_option = {spec["convert_option"]: block_type
                 for block_type, spec in BACKEND_OPERATIONS.items() if "convert_option" in spec}
    for option, block_type in by_option.items():
        if option not in user or user[option] is None:
            if option == "rmsnorm_impl" and ("rmsnorm_prescale" in user or "rmsnorm_upcast" in user):
                continue  # already resolved from the boolean flags
            if option != "attn_impl":
                resolved[option] = default_for(backend, block_type)

    precisions = options_for(backend, "kv_precision")
    if user.get("cache_impl") is None and all(p in precisions for p in kv_precision_pair(resolved["kv_precision"])):
        resolved["cache_impl"] = required_cache(backend, resolved["kv_precision"]) or resolved["cache_impl"]
    cache = resolved["cache_impl"]
    if resolved.get("attn_impl") is None:
        candidates = [default_for(backend, "attention_kernel")] + list(options_for(backend, "attention_kernel"))
        required = KV_CACHES[cache].requires_kernel
        for name in candidates:
            if ATTENTION_KERNELS[name].requires_cache in (None, cache) and required in (None, name):
                resolved["attn_impl"] = name
                break

    for option, block_type in by_option.items():
        check_choice(backend, block_type, resolved[option])
    check_kv_combination(backend, resolved["cache_impl"], resolved["kv_precision"])
    return resolved


def check_components(backend, decoder):
    """Validate an already-converted decoder against the backend's supported implementations."""
    check_choice(backend, "kv_cache", decoder.cache_impl)
    check_choice(backend, "batch_dim", bool(getattr(decoder, "add_batch_dim", False)))
    if len(decoder.blocks):
        check_choice(backend, "attention_kernel", decoder.blocks[0].attn.attn_impl)
        precision = canonical_kv_precision(getattr(decoder.blocks[0].attn, "kv_precision", "full"))
        check_choice(backend, "kv_precision", precision)
        check_kv_combination(backend, decoder.cache_impl, precision)


__all__ = [
    "BACKENDS", "BACKEND_OPERATIONS", "default_for", "options_for", "config_default", "required_cache",
    "check_kv_combination", "check_choice", "check_config_choices", "check_config_requirements", "check_prefill", "resolve_convert_options", "check_components",
]
