"""Catalog of the selectable atoms of ``blocks/``.

``BLOCK_CATALOG`` is the single place that lists every interchangeable piece of
the export-ready transformer, grouped by family. For every atom it gives:

- ``object``: where it is implemented (relative to ``transformersurgeon.blocks``);
- ``summary``: what it does, in one sentence;
- ``implements``: the operation it provides. Atoms with the same ``implements``
  value are interchangeable implementations of the same operation (for example
  three numerically different RMS normalisations);
- ``selected_by``: who chooses it. ``"model"`` means the model dictionary in
  ``models/`` names it (the model *needs* that operation); ``"export"`` means
  the export dictionary in ``export/operations.py`` picks it per backend (the
  device *prefers* that implementation).

Code never hard-codes these choices: ``build_block(family, name, ...)`` looks
the atom up here.
"""

from importlib import import_module


def _atom(obj, summary, implements, selected_by):
    return {"object": obj, "summary": summary, "implements": implements, "selected_by": selected_by}


BLOCK_CATALOG = {
    # Whole stacks of repeated blocks. Model key: "structure".
    "stack": {
        "transformer_decoder": _atom(
            "stacks.decoder.TransformerDecoder",
            "Causal decoder: pre-normalised blocks, rotary positions, KV cache, final normalisation.",
            "transformer_decoder", "model"),
        "transformer_encoder": _atom(
            "stacks.encoder.TransformerEncoder",
            "Bidirectional encoder: pre- or post-normalised blocks without cache.",
            "transformer_encoder", "model"),
    },
    # Attention modules. Model key: "attn_type".
    "attention": {
        "mha_causal": _atom(
            "attention.causal.MHACausal",
            "Causal multi-head (grouped-query) attention over a fixed-size KV cache.",
            "mha_causal", "model"),
        "mha_encoder": _atom(
            "attention.bidirectional.MHAEncoder",
            "Bidirectional multi-head attention without cache.",
            "mha_encoder", "model"),
    },
    # How attention is computed inside the attention module. Export option: "attn_impl".
    "attention_kernel": {
        "manual": _atom(
            "attention.kernels.manual.ManualKernel",
            "Explicit matrix products and softmax; portable; supports masked softmax.",
            "attention_kernel", "export"),
        "sdpa": _atom(
            "attention.kernels.sdpa.SdpaKernel",
            "torch.nn.functional.scaled_dot_product_attention.",
            "attention_kernel", "export"),
        "custom_sdpa": _atom(
            "attention.kernels.custom_sdpa.CustomSdpaKernel",
            "ExecuTorch fused CPU kernel that scans only the filled cache; needs the mutable cache.",
            "attention_kernel", "export"),
        "paged": _atom(
            "attention.kernels.paged.PagedKernel",
            "Fused rotary embedding + paged cache write + attention; needs the io_paged cache.",
            "attention_kernel", "export"),
    },
    # How past keys/values are stored. Export option: "cache_impl".
    "kv_cache": {
        "mutable": _atom(
            "attention.kv_cache.mutable.MutableCache",
            "Internal module state written in place (not a graph input/output).",
            "kv_cache", "export"),
        "io_scatter": _atom(
            "attention.kv_cache.io_scatter.ScatterCache",
            "Graph input/output, updated with a functional scatter.",
            "kv_cache", "export"),
        "io_concat": _atom(
            "attention.kv_cache.io_concat.ConcatCache",
            "Graph input/output, updated with a one-hot select (no scatter); one token per call.",
            "kv_cache", "export"),
        "io_inplace": _atom(
            "attention.kv_cache.io_inplace.InplaceCache",
            "Graph input/output in heads-major layout, updated by an in-place-capable operator; allows prefill.",
            "kv_cache", "export"),
        "io_paged": _atom(
            "attention.kv_cache.io_paged.PagedCache",
            "Paged pool as graph input/output; written and read by the paged kernel.",
            "kv_cache", "export"),
    },
    # Feed-forward blocks. Model key: "mlp_type".
    "mlp": {
        "mlp": _atom("mlp.plain.MLP", "down(act(up(x))).", "mlp", "model"),
        "mlp_gated": _atom("mlp.gated.MLPGated", "down(act(gate(x)) * up(x)).", "mlp_gated", "model"),
    },
    # Normalisations. Model key: "norm_type" names the operation; for RMS
    # normalisation the export option "rmsnorm_impl" picks the implementation.
    "norm": {
        "rmsnorm": _atom(
            "norm.rmsnorm.RMSNorm",
            "Plain RMS normalisation; fusable into one operation by backend compilers.",
            "rmsnorm", "export"),
        "rmsnorm_prescaled": _atom(
            "norm.rmsnorm_prescaled.RMSNormPrescaled",
            "Rescales rows by max |x| first to avoid float16 overflow.",
            "rmsnorm", "export"),
        "rmsnorm_upcast": _atom(
            "norm.rmsnorm_upcast.RMSNormUpcast",
            "Normalises in float32 and casts back (Hugging Face formulation).",
            "rmsnorm", "export"),
        "layernorm": _atom("norm.layernorm.LayerNorm", "Layer normalisation.", "layernorm", "model"),
    },
    # Rotary-embedding frequency tables. Model key: "rope_type" ("auto" = from the HF configuration).
    "rope_frequencies": {
        "default": _atom("positions.frequencies_default.inv_freq_default",
                         "Plain rotary frequencies.", "rope_frequencies", "model"),
        "llama3": _atom("positions.frequencies_band_scaled.inv_freq_band_scaled",
                        "Band-wise scaled frequencies for long contexts (HF rope_type 'llama3').",
                        "rope_frequencies", "model"),
    },
}


def resolve_atom(family, name):
    """Return the class or function registered as ``BLOCK_CATALOG[family][name]``."""
    try:
        entry = BLOCK_CATALOG[family][name]
    except KeyError:
        available = sorted(BLOCK_CATALOG.get(family, {}))
        raise ValueError(f"Unknown {family} atom {name!r}; available: {available}.") from None
    module_path, _, attr = entry["object"].rpartition(".")
    return getattr(import_module(f"{__package__}.{module_path}"), attr)


def build_block(family, name, *args, **kwargs):
    """Instantiate the atom ``BLOCK_CATALOG[family][name]`` with the given arguments."""
    return resolve_atom(family, name)(*args, **kwargs)


def implementations_of(operation, family=None):
    """Names of the atoms (optionally within ``family``) that implement ``operation``."""
    families = [family] if family else list(BLOCK_CATALOG)
    return [name for fam in families for name, entry in BLOCK_CATALOG[fam].items()
            if entry["implements"] == operation]


def norm_atom_name(norm_type, rmsnorm_impl="rmsnorm_prescaled"):
    """Catalog name of the normalisation atom for a model's ``norm_type``.

    The model decides the operation (``"rmsnorm"`` or ``"layernorm"``); for RMS
    normalisation the export option ``rmsnorm_impl`` picks the implementation.
    """
    if norm_type == "rmsnorm":
        if rmsnorm_impl not in implementations_of("rmsnorm", "norm"):
            raise ValueError(f"Unknown rmsnorm_impl {rmsnorm_impl!r}; "
                             f"available: {implementations_of('rmsnorm', 'norm')}.")
        return rmsnorm_impl
    return norm_type


__all__ = ["BLOCK_CATALOG", "resolve_atom", "build_block", "implementations_of", "norm_atom_name"]
