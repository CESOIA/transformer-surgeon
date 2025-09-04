# coding=utf-8
#
# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
#
# This file is based on HuggingFace Transformers Qwen2.5 implementation:
# Source repository: https://github.com/huggingface/transformers
# Source file: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Commit: 1d742644c09928d6d596c56eae2ffcc8e303be6e
# Retrieved on 2025-09-04
#
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2-VL model."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm

# original --->
# from ...activations import ACT2FN
# from ...cache_utils import Cache, DynamicCache
# from ...generation import GenerationMixin
# from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
# from ...modeling_flash_attention_utils import FlashAttentionKwargs
# from ...modeling_layers import GradientCheckpointingLayer
# from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
# from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
# from ...processing_utils import Unpack
# from ...utils import (
#     TransformersKwargs,
#     auto_docstring,
#     can_return_tuple,
#     is_torchdynamo_compiling,
#     logging,
# )
# from ...utils.deprecation import deprecate_kwarg
# from ..qwen2.modeling_qwen2 import (
#     Qwen2RMSNorm,
# )
# from .configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLTextConfig, Qwen2VLVisionConfig
# -------------
from transformers.activations import ACT2FN
from transformers.utils import (
    auto_docstring,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
)
from .configuration_qwen2_vl_c import (
    Qwen2VLConfigCompress,
    Qwen2VLTextConfigCompress,
    Qwen2VLVisionConfigCompress
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLRotaryEmbedding,
    VisionRotaryEmbedding,
    PatchEmbed,
    PatchMerger,
    VisionAttention,
    Qwen2VLVisionBlock,
    Qwen2MLP,
    Qwen2VLAttention,
    Qwen2VLDecoderLayer,
    Qwen2VLPreTrainedModel,
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLTextModel,
    Qwen2VLModel,
    Qwen2VLForConditionalGeneration,
)

from ..utils import (
    LinearLRD,
    get_validated_dict_value,
)
# <--- CESOIA modifications

logger = logging.get_logger(__name__)

# original --->
# @dataclass
# @auto_docstring(
#     custom_intro="""
#     Base class for Llava outputs, with hidden states and attentions.
#     """
# )
# class Qwen2VLModelOutputWithPast(ModelOutput):
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# @dataclass
# @auto_docstring(
#     custom_intro="""
#     Base class for Qwen2VL causal language model (or autoregressive) outputs.
#     """
# )
# class Qwen2VLCausalLMOutputWithPast(ModelOutput):
# [...]
# -------------
# <--- CESOIA modifications
    
# original --->
# class Qwen2VLRotaryEmbedding(nn.Module):
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# Copied from transformers.models.llama.modeling_llama.rotate_half
# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)
# -------------
# <--- CESOIA modifications

# original --->
# def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
#     [...]
# -------------
# <--- CESOIA modifications

# original --->
# def apply_rotary_pos_emb_vision(
#     q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     [...]
# -------------
# <--- CESOIA modifications

# original --->
# class VisionRotaryEmbedding(nn.Module):
#     [...]
# -------------
# <--- CESOIA modifications

# original --->
# class PatchEmbed(nn.Module):
#     [...]
# -------------
# <--- CESOIA modifications

# NOTE: this could potentially be compressed given the presence of two linear layers
# original --->
# class PatchMerger(nn.Module):
#     def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
#         super().__init__()
#         self.hidden_size = context_dim * (spatial_merge_size**2)
#         self.ln_q = LayerNorm(context_dim, eps=1e-6)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.GELU(),
#             nn.Linear(self.hidden_size, dim),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
#         return x
# -------------
# <--- CESOIA modifications

# original --->
# class VisionMlp(nn.Module):
# def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = ACT2FN[hidden_act]
#         self.fc2 = nn.Linear(hidden_dim, dim)
# -------------
class VisionMlpCompress(nn.Module):
    def __init__(self, config, dim: int, hidden_dim: int, hidden_act: str, layer_idx: int) -> None:
        super().__init__()
        self.rank_fc1 = get_validated_dict_value(config.lrd_rank_lists, "mlp_up", layer_idx, default="full", min_value=1)
        self.rank_fc2 = get_validated_dict_value(config.lrd_rank_lists, "mlp_down", layer_idx, default="full", min_value=1)
        self.fc1 = LinearLRD(dim, hidden_dim, lrd_rank=self.rank_fc1)
        self.act = ACT2FN[hidden_act]
        self.fc2 = LinearLRD(hidden_dim, dim, lrd_rank=self.rank_fc2)
# <------------- CESOIA modifications

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

# original --->
# Copied from transformers.models.llama.modeling_llama.repeat_kv
# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attention_mask: Optional[torch.Tensor],
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs,
# ):
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# class VisionAttention(nn.Module):
#     def __init__(self, config: Qwen2VLVisionConfig) -> None:
#         super().__init__()
# -------------
class VisionAttentionCompress(VisionAttention):
    def __init__(self, config: Qwen2VLVisionConfigCompress, layer_idx: int) -> None:
        super(VisionAttention, self).__init__()
# <--- CESOIA modifications
        self.dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
# original --->
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
# -------------
        rank_qkv = get_validated_dict_value(config.lrd_rank_lists, "sa_qkv", index=layer_idx, default="full", min_value=1)
        rank_proj = get_validated_dict_value(config.lrd_rank_lists, "sa_out", index=layer_idx, default="full", min_value=1)
        self.qkv = LinearLRD(self.dim, self.dim * 3, bias=True, lrd_rank=rank_qkv)
        self.proj = LinearLRD(self.dim, self.dim, bias=True, lrd_rank=rank_proj)
# <--- CESOIA modifications
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

# original --->
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     cu_seqlens: torch.Tensor,
    #     rotary_pos_emb: Optional[torch.Tensor] = None,
    #     position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2VLVisionBlock(GradientCheckpointingLayer):
#     def __init__(self, config, attn_implementation: str = "sdpa") -> None:
#         super().__init__()
# -------------
class Qwen2VLVisionBlockCompress(Qwen2VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa", layer_idx: int = 0) -> None:
        super(Qwen2VLVisionBlock, self).__init__()
# <--- CESOIA modifications

        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionAttentionCompress(config=config, layer_idx=layer_idx)
# original --->
        # self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)
# -------------
        self.mlp = VisionMlpCompress(config, dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act, layer_idx=layer_idx)

# original --->
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     cu_seqlens: torch.Tensor,
    #     rotary_pos_emb: Optional[torch.Tensor] = None,
    #     position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     [...]
# -------------
# <--- CESOIA modifications

# original --->
# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
# class Qwen2MLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
# -------------
class Qwen2MLPCompress(Qwen2MLP):
    def __init__(self, config, layer_idx: int):
        super(Qwen2MLP, self).__init__()
# <--- CESOIA modifications
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
# original --->
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
# -------------
        self.rank_gate = get_validated_dict_value(config.lrd_rank_lists, "mlp_gate", layer_idx, default="full", min_value=1)
        self.rank_up = get_validated_dict_value(config.lrd_rank_lists, "mlp_up", layer_idx, default="full", min_value=1)
        self.rank_down = get_validated_dict_value(config.lrd_rank_lists, "mlp_down", layer_idx, default="full", min_value=1)
        self.gate_proj = LinearLRD(self.hidden_size, self.intermediate_size, bias=False, lrd_rank=self.rank_gate)
        self.up_proj = LinearLRD(self.hidden_size, self.intermediate_size, bias=False, lrd_rank=self.rank_up)
        self.down_proj = LinearLRD(self.intermediate_size, self.hidden_size, bias=False, lrd_rank=self.rank_down)
# <------------- CESOIA modifications
        self.act_fn = ACT2FN[config.hidden_act]

# original --->
    # def forward(self, x):
    #     down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #     return down_proj
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2VLAttention(nn.Module):
#     """
#     Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
#     and "Generating Long Sequences with Sparse Transformers".
#     """

#     def __init__(self, config: Qwen2VLTextConfig, layer_idx: Optional[int] = None):
#         super().__init__()
# -------------
class Qwen2VLAttentionCompress(Qwen2VLAttention):
    def __init__(self, config: Qwen2VLTextConfigCompress, layer_idx: int = 0):
        super(Qwen2VLAttention, self).__init__()
# <--- CESOIA modifications

        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
# original --->
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
# -------------
        self.rank_q = get_validated_dict_value(config.lrd_rank_lists, "sa_q", layer_idx, default="full", min_value=1)
        self.rank_k = get_validated_dict_value(config.lrd_rank_lists, "sa_k", layer_idx, default="full", min_value=1)
        self.rank_v = get_validated_dict_value(config.lrd_rank_lists, "sa_v", layer_idx, default="full", min_value=1)
        self.rank_o = get_validated_dict_value(config.lrd_rank_lists, "sa_out", layer_idx, default="full", min_value=1)
        self.q_proj = LinearLRD(self.hidden_size, self.num_heads * self.head_dim, bias=True, lrd_rank=self.rank_q)
        self.k_proj = LinearLRD(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, lrd_rank=self.rank_k)
        self.v_proj = LinearLRD(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, lrd_rank=self.rank_v)
        self.o_proj = LinearLRD(self.num_heads * self.head_dim, self.hidden_size, bias=False, lrd_rank=self.rank_o)
# <--- CESOIA modifications
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

# original --->
    # @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     output_attentions: bool = False,
    #     use_cache: bool = False,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    #     [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2VLDecoderLayer(GradientCheckpointingLayer):
#     def __init__(self, config: Qwen2VLTextConfig, layer_idx: int):
#         super().__init__()
# -------------
class Qwen2VLDecoderLayerCompress(Qwen2VLDecoderLayer):
    def __init__(self, config: Qwen2VLTextConfigCompress, layer_idx: int):
        super(Qwen2VLDecoderLayer, self).__init__()
# <--- CESOIA modifications
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
# original --->
        # self.self_attn = Qwen2VLAttention(config, layer_idx)

        # self.mlp = Qwen2MLP(config)
# -------------
        self.self_attn = Qwen2VLAttentionCompress(config, layer_idx)

        self.mlp = Qwen2MLPCompress(config, layer_idx)
# <--- CESOIA modifications
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

# original --->
    # @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[tuple[torch.Tensor]] = None,
    #     output_attentions: Optional[bool] = False,
    #     use_cache: Optional[bool] = False,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
    #     [...]
# -------------
# <--- CESOIA modifications
        
# original --->
# @auto_docstring
# class Qwen2VLPreTrainedModel(PreTrainedModel):
#     config: Qwen2VLConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
#     _skip_keys_device_placement = "past_key_values"
#     _supports_flash_attn = True
#     _supports_sdpa = True

#     _can_compile_fullgraph = True
#     _supports_attention_backend = True
# -------------
class Qwen2VLPreTrainedModelCompress(Qwen2VLPreTrainedModel):
    config: Qwen2VLConfigCompress
    _no_split_modules = ["Qwen2VLDecoderLayerCompress", "Qwen2VLVisionBlockCompress"]
# <--- CESOIA modifications

# original --->
# @auto_docstring
# class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
#     config: Qwen2VLVisionConfig
#     _no_split_modules = ["Qwen2VLVisionBlock"]

#     def __init__(self, config) -> None:
#         super().__init__(config)
# -------------
class Qwen2VisionTransformerPretrainedModelCompress(Qwen2VisionTransformerPretrainedModel, Qwen2VLPreTrainedModelCompress):
    config: Qwen2VLVisionConfigCompress
    _no_split_modules = ["Qwen2VLVisionBlockCompress"]

    def __init__(self, config) -> None:
        super(Qwen2VLPreTrainedModelCompress, self).__init__(config)
# <--- CESOIA modifications
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

# original --->
        # self.blocks = nn.ModuleList([Qwen2VLVisionBlock(config) for _ in range(config.depth)])
# -------------
        self.blocks = nn.ModuleList([Qwen2VLVisionBlockCompress(config, layer_idx) for layer_idx in range(config.depth)])
# <--- CESOIA modifications
        self.norm = LayerNorm(config.embed_dim, eps=1e-6)
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )
        self.gradient_checkpointing = False

# original --->
    # def get_dtype(self) -> torch.dtype:
    #     return self.blocks[0].mlp.fc2.weight.dtype

    # def get_device(self) -> torch.device:
    #     return self.blocks[0].mlp.fc2.weight.device

    # def rot_pos_emb(self, grid_thw):
    #     [...]
        
    # @auto_docstring
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     grid_thw: torch.Tensor,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     [...]
# -------------
# <--- CESOIA modifications

# original --->
# @auto_docstring
# class Qwen2VLTextModel(Qwen2VLPreTrainedModel):
#     config: Qwen2VLTextConfig

#     def __init__(self, config: Qwen2VLTextConfig):
#         super().__init__(config)
# -------------
class Qwen2VLTextModelCompress(Qwen2VLTextModel, Qwen2VLPreTrainedModelCompress):
    config: Qwen2VLTextConfigCompress

    def __init__(self, config: Qwen2VLTextConfigCompress):
        super(Qwen2VLTextModel, self).__init__(config)
# <--- CESOIA modifications
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
# original --->
        # self.layers = nn.ModuleList(
        #     [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        # )
# -------------
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayerCompress(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
# <--- CESOIA modifications
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

# original --->
    # @auto_docstring
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> Union[tuple, BaseModelOutputWithPast]:
    #     [...]
# -------------
# <--- CESOIA modifications

# original --->
# @auto_docstring
# class Qwen2VLModel(Qwen2VLPreTrainedModel):
#     base_model_prefix = ""
#     _checkpoint_conversion_mapping = {"^model": "language_model"}
#     # Reference: fix gemma3 grad acc #37208
#     accepts_loss_kwargs = False

#     def __init__(self, config: Qwen2VLConfig):
#         super().__init__(config)
#         self.visual = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
#         self.language_model = Qwen2VLTextModel._from_config(config.text_config)
# -------------
class Qwen2VLModelCompress(Qwen2VLModel, Qwen2VLPreTrainedModelCompress):
    def __init__(self, config: Qwen2VLConfigCompress):
        super(Qwen2VLPreTrainedModelCompress, self).__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModelCompress._from_config(config.vision_config)
        self.language_model = Qwen2VLTextModelCompress._from_config(config.text_config)
# <--- CESOIA modifications
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

# original --->
    # def get_input_embeddings(self):
    #     return self.language_model.get_input_embeddings()

    # def set_input_embeddings(self, value):
    #     self.language_model.set_input_embeddings(value)

    # def set_decoder(self, decoder):
    #     self.language_model = decoder

    # def get_decoder(self):
    #     return self.language_model

    # def get_rope_index(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     [...]

    # def get_video_features(
    #     self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    # ):
    #     [...]

    # def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
    #     [...]

    # def get_placeholder_mask(
    #     self,
    #     input_ids: torch.LongTensor,
    #     inputs_embeds: torch.FloatTensor,
    #     image_features: torch.FloatTensor = None,
    #     video_features: torch.FloatTensor = None,
    # ):
    #     [...]

    # @auto_docstring
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_values_videos: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     rope_deltas: Optional[torch.LongTensor] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs: Unpack[TransformersKwargs],
    # ) -> Union[tuple, Qwen2VLModelOutputWithPast]:
    #     [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
#     _checkpoint_conversion_mapping = {
#         "^visual": "model.visual",
#         r"^model(?!\.(language_model|visual))": "model.language_model",
#     }
#     _tied_weights_keys = ["lm_head.weight"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = Qwen2VLModel(config)
# -------------
class Qwen2VLForConditionalGenerationCompress(Qwen2VLForConditionalGeneration, Qwen2VLPreTrainedModelCompress):
    config_class = Qwen2VLConfigCompress
    def __init__(self, config):
        super(Qwen2VLForConditionalGeneration, self).__init__(config)
        self.model = Qwen2VLModelCompress(config)
# <--- CESOIA modifications
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

# original --->
    # def get_input_embeddings(self):
    #     return self.model.get_input_embeddings()

    # def set_input_embeddings(self, value):
    #     self.model.set_input_embeddings(value)

    # def set_decoder(self, decoder):
    #     self.model.set_decoder(decoder)

    # def get_decoder(self):
    #     return self.model.get_decoder()

    # def get_video_features(
    #     self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    # ):
    #     return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    # def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
    #     return self.model.get_image_features(pixel_values, image_grid_thw)

    # # Make modules available through conditional class for BC
    # @property
    # def language_model(self):
    #     return self.model.language_model

    # @property
    # def visual(self):
    #     return self.model.visual

    # @can_return_tuple
    # @auto_docstring
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_values_videos: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     rope_deltas: Optional[torch.LongTensor] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs: Unpack[TransformersKwargs],
    # ) -> Union[tuple, Qwen2VLCausalLMOutputWithPast]:
    #     [...]

    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     cache_position=None,
    #     position_ids=None,
    #     use_cache=True,
    #     pixel_values=None,
    #     pixel_values_videos=None,
    #     image_grid_thw=None,
    #     video_grid_thw=None,
    #     **kwargs,
    # ):
    #     [...]

    # def _get_image_nums_and_video_nums(
    #     self,
    #     input_ids: Optional[torch.LongTensor],
    #     inputs_embeds: Optional[torch.Tensor] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     [...]

    # def _expand_inputs_for_generation(
    #     self,
    #     expand_size: int = 1,
    #     is_encoder_decoder: bool = False,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     **model_kwargs,
    # ) -> tuple[torch.LongTensor, dict[str, Any]]:
    #     [...]
# -------------
# <--- CESOIA modifications


# original --->
# __all__ = ["Qwen2VLForConditionalGeneration", "Qwen2VLModel", "Qwen2VLPreTrainedModel", "Qwen2VLTextModel"]
# -------------
__all__ = ["Qwen2VLForConditionalGenerationCompress", "Qwen2VLModelCompress", "Qwen2VLPreTrainedModelCompress", "Qwen2VLTextModelCompress"]
# <--- CESOIA modifications