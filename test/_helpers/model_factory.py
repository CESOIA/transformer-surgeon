"""Tiny random-weight compressed models for every supported family.

Instantiating a ``*Compress`` class from a small config exercises the full
framework-specific code path (layer replacement at init, manager construction,
scheme generation, apply/restore, convert) in well under a second and with zero
network I/O. This is the backbone of the parametrized ``e2e`` model-family tests.

Each entry in ``FAMILIES`` is a :class:`FamilySpec` describing how to build the
model and what to compress. VL families build nested vision+text sub-configs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch


@dataclass
class FamilySpec:
    name: str
    build: Callable[[], torch.nn.Module]
    manager: Callable[[torch.nn.Module], object]
    sample_inputs: Callable[[torch.nn.Module], dict]
    # A path substring selecting a single linear layer to LRD/quantize.
    lrd_criteria: object
    # MLP layers that form one coupled prune group (must be pruned together).
    mlp_prune_criteria: object
    is_causal: bool = False
    # Families that cannot yet build a manager / fully run are marked here so the
    # e2e suite can xfail them with a pointer to FRAMEWORK_PROBLEMS.md.
    known_broken: str = ""
    # Structured MLP pruning is only cleanly wired for single-tower text decoders.
    # On dual-tower VL models auto_groups() mixes vision+text coupled groups and the
    # shared mask length no longer matches the text down_proj (FRAMEWORK_PROBLEMS.md #6).
    struct_prune_supported: bool = True


def _import_ts():
    import transformersurgeon as T

    return T


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _qwen2():
    T = _import_ts()
    cfg = T.Qwen2ConfigCompress(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return T.Qwen2ForCausalLMCompress(cfg)


def _llama():
    T = _import_ts()
    cfg = T.LlamaConfigCompress(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return T.LlamaForCausalLMCompress(cfg)


def _bert():
    T = _import_ts()
    cfg = T.BertConfigCompress(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=64, num_labels=2,
    )
    return T.BertForSequenceClassificationCompress(cfg)


def _modernbert():
    T = _import_ts()
    cfg = T.ModernBertConfigCompress(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=64, num_labels=2,
        global_attn_every_n_layers=2, local_attention=32,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=3, sep_token_id=4,
    )
    return T.ModernBertForSequenceClassificationCompress(cfg)


def _distilbert():
    T = _import_ts()
    cfg = T.DistilBertConfigCompress(
        vocab_size=256, dim=64, hidden_dim=128, n_layers=2, n_heads=4,
        max_position_embeddings=64, num_labels=2,
    )
    return T.DistilBertForSequenceClassificationCompress(cfg)


def _vit():
    T = _import_ts()
    cfg = T.ViTConfigCompress(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
        image_size=32, patch_size=16, num_channels=3, num_labels=10,
    )
    return T.ViTForImageClassificationCompress(cfg)


def _make_vl(model_attr, vis_attr, txt_attr, cfg_attr):
    def build():
        T = _import_ts()
        VisC = getattr(T, vis_attr)
        TxtC = getattr(T, txt_attr)
        CfgC = getattr(T, cfg_attr)
        ModelC = getattr(T, model_attr)
        vc = VisC(depth=2, hidden_size=32, embed_dim=32, num_heads=2,
                  intermediate_size=64, out_hidden_size=64)
        tc = TxtC(vocab_size=256, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=64,
                  # Real Qwen2-VL text towers use multimodal RoPE; without an
                  # mrope_section the attention forward raises KeyError.
                  rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]})
        cfg = CfgC(vision_config=vc.to_dict(), text_config=tc.to_dict())
        return ModelC(cfg)

    return build


def _ids(n=8):
    return lambda m: {"input_ids": torch.randint(0, 256, (1, n))}


def _pixels(m):
    return {"pixel_values": torch.randn(1, 3, 32, 32)}


def _mgr(attr):
    def make(model):
        T = _import_ts()
        return getattr(T, attr)(model)

    return make


FAMILIES: Dict[str, FamilySpec] = {
    "qwen2": FamilySpec(
        "qwen2", _qwen2, _mgr("Qwen2CompressionSchemesManager"), _ids(),
        lrd_criteria="self_attn.q_proj",
        mlp_prune_criteria=["mlp.gate_proj", "mlp.up_proj"], is_causal=True,
    ),
    "llama": FamilySpec(
        "llama", _llama, _mgr("LlamaCompressionSchemesManager"), _ids(),
        lrd_criteria="self_attn.q_proj",
        mlp_prune_criteria=["mlp.gate_proj", "mlp.up_proj"], is_causal=True,
    ),
    "bert": FamilySpec(
        "bert", _bert, _mgr("BertCompressionSchemesManager"), _ids(),
        lrd_criteria="attention.self.query",
        mlp_prune_criteria="intermediate.dense",
    ),
    "modernbert": FamilySpec(
        "modernbert", _modernbert, _mgr("ModernBertCompressionSchemesManager"), _ids(),
        lrd_criteria="attn.Wqkv",
        mlp_prune_criteria="mlp.Wi",
        struct_prune_supported=False,
    ),
    "distilbert": FamilySpec(
        "distilbert", _distilbert, _mgr("DistilBertCompressionSchemesManager"), _ids(),
        lrd_criteria="q_lin", mlp_prune_criteria="ffn.lin1",
    ),
    "vit": FamilySpec(
        "vit", _vit, _mgr("ViTCompressionSchemesManager"), _pixels,
        lrd_criteria="attention.attention.query", mlp_prune_criteria="intermediate.dense",
    ),
    "qwen2_vl": FamilySpec(
        "qwen2_vl",
        _make_vl("Qwen2VLForConditionalGenerationCompress", "Qwen2VLVisionConfigCompress",
                 "Qwen2VLTextConfigCompress", "Qwen2VLConfigCompress"),
        _mgr("Qwen2VLCompressionSchemesManager"), _ids(),
        lrd_criteria=[["language_model", "self_attn.q_proj", 0]],
        mlp_prune_criteria=[["language_model", "mlp.gate_proj"],
                            ["language_model", "mlp.up_proj"]],
        struct_prune_supported=False,
    ),
    "qwen2_5_vl": FamilySpec(
        "qwen2_5_vl",
        _make_vl("Qwen2_5_VLForConditionalGenerationCompress", "Qwen2_5_VLVisionConfigCompress",
                 "Qwen2_5_VLTextConfigCompress", "Qwen2_5_VLConfigCompress"),
        _mgr("Qwen2_5_VLCompressionSchemesManager"), _ids(),
        lrd_criteria=[["language_model", "self_attn.q_proj", 0]],
        mlp_prune_criteria=[["language_model", "mlp.gate_proj"],
                            ["language_model", "mlp.up_proj"]],
        struct_prune_supported=False,
    ),
}

ALL_FAMILY_NAMES: List[str] = list(FAMILIES)
CAUSAL_FAMILY_NAMES: List[str] = [n for n, s in FAMILIES.items() if s.is_causal]
