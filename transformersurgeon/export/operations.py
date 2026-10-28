"""Which device operation implements each block, for each export backend.

``BACKEND_OPERATIONS`` links the selectable atoms of ``blocks/`` (see
``blocks/catalog.py``) and the backend-specific lowering choices to what runs on
the device. It is organised by *block type*; for every backend it gives:

- ``default``: the implementation used when the caller chooses nothing;
- ``options``: every implementation known to work on that backend, each with a
  short description of the resulting device operation.

Each block type also says how the choice reaches the code:

- ``"convert_option"``: a key of ``convert_options`` (the atoms used to build
  the export-ready model, for example ``cache_impl``);
- ``"config_field"``: a field of the backend's export configuration (for
  example the XNNPACK ``quant_path``).

The exporters read this dictionary: missing convert options and configuration
fields take the backend default from here, and a choice that is not listed for
the backend is rejected before any tracing starts.
"""

from ..blocks import ATTENTION_KERNELS, KV_CACHES

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
                "io_scatter": "ONNX ScatterND.",
                "io_concat": "ONNX Where on a one-hot position mask.",
                "io_paged": "Only with a custom translation of tsurgeon::paged_attention "
                            "(the tensorrt backend provides one).",
            }},
            "tensorrt": {"default": "io_inplace", "options": {
                "io_inplace": "TensorScatter executed in place by TensorRT (output aliased to input).",
                "io_scatter": "TensorRT ScatterND.",
                "io_concat": "TensorRT Select.",
                "io_paged": "Paged KV pool of the TensorRT Edge-LLM AttentionPlugin.",
            }},
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


def check_choice(backend, block_type, value):
    """Raise ``ValueError`` if ``value`` is not a supported ``block_type`` implementation on ``backend``."""
    options = options_for(backend, block_type)
    if value not in options:
        raise ValueError(
            f"{block_type} implementation {value!r} is not supported by the {backend} backend; "
            f"supported: {list(options)} (default {default_for(backend, block_type)!r}). "
            "See transformersurgeon/export/operations.py."
        )


def check_config_choices(backend, config):
    """Validate every configuration field of ``config`` that selects an operation."""
    for block_type, spec in BACKEND_OPERATIONS.items():
        field_name = spec.get("config_field", {}).get(backend)
        if field_name is not None and hasattr(config, field_name):
            check_choice(backend, block_type, getattr(config, field_name))


def resolve_convert_options(backend, options=None):
    """Complete the caller's ``convert_options`` with the backend defaults and validate them.

    The older aliases are honoured first: ``use_sdpa=True`` means
    ``attn_impl="sdpa"``; ``rmsnorm_prescale``/``rmsnorm_upcast`` select an
    ``rmsnorm_impl``. When the attention kernel is not given, the backend default
    is used unless it is incompatible with the chosen cache, in which case the
    first compatible supported kernel is taken.
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
    return resolved


def check_components(backend, decoder):
    """Validate an already-converted decoder against the backend's supported implementations."""
    check_choice(backend, "kv_cache", decoder.cache_impl)
    check_choice(backend, "batch_dim", bool(getattr(decoder, "add_batch_dim", False)))
    if len(decoder.blocks):
        check_choice(backend, "attention_kernel", decoder.blocks[0].attn.attn_impl)


__all__ = [
    "BACKENDS", "BACKEND_OPERATIONS", "default_for", "options_for", "config_default", "check_choice",
    "check_config_choices", "resolve_convert_options", "check_components",
]
