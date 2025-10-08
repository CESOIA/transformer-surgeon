# coding=utf-8
#
# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
#
# This file is based on HuggingFace Transformers Qwen2.5 implementation:
# Source repository: https://github.com/huggingface/transformers
# Source file: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# Commit: 1d742644c09928d6d596c56eae2ffcc8e303be6e
# Retrieved on 2025-09-03
#
# Copyright 2025 The Qwen team and the HuggingFace Inc. team. All rights reserved.
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

# original --->
# -------------
FORCE_ORIGINAL_LAYERS = False  # for debugging purposes
# <--- CESOIA modifications

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
# from ...utils.deprecation import deprecate_kwarg
# from ..qwen2.modeling_qwen2 import Qwen2RMSNorm
# from .configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig
# -------------
from transformers.activations import ACT2FN
# original --->
# from transformers.utils import auto_docstring, logging
# -------------
from transformers.utils import logging
# <--- CESOIA modifications
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from .configuration_qwen2_5_vl_c import (
    Qwen2_5_VLConfigCompress,
    Qwen2_5_VLTextConfigCompress,
    Qwen2_5_VLVisionConfigCompress
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLPatchMerger,
    # rotate_half,
    # apply_rotary_pos_emb_vision,
    # repeat_kv,
    # eager_attention_forward,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLRotaryEmbedding,
    # apply_multimodal_rotary_pos_emb,
    Qwen2_5_VLAttention,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
)

from ...utils import get_validated_dict_value
from ...layers import LinearCompressed

# <--- CESOIA modifications

logger = logging.get_logger(__name__)

# original --->
# class Qwen2_5_VLMLP(nn.Module):
#     def __init__(self, config, bias: bool = False):
# -------------
class Qwen2_5_VLMLPCompress(nn.Module):
    def __init__(self, config, bias: bool = False, path: str = None) -> None:
# <--- CESOIA modifications
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
# original --->
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
# -------------
        self.gate_rank = get_validated_dict_value(config.lrd_ranks, path+".gate_proj", default="full", min_value=1)
        self.up_rank = get_validated_dict_value(config.lrd_ranks, path+".up_proj", default="full", min_value=1)
        self.down_rank = get_validated_dict_value(config.lrd_ranks, path+".down_proj", default="full", min_value=1)

        if FORCE_ORIGINAL_LAYERS:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        else:
            self.gate_proj = LinearCompressed(self.hidden_size, self.intermediate_size, bias=bias, lrd_rank=self.gate_rank)
            self.up_proj = LinearCompressed(self.hidden_size, self.intermediate_size, bias=bias, lrd_rank=self.up_rank)
            self.down_proj = LinearCompressed(self.intermediate_size, self.hidden_size, bias=bias, lrd_rank=self.down_rank)
# <--- CESOIA modifications
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

# original --->
# class Qwen2_5_VisionPatchEmbed(nn.Module):
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2_5_VisionRotaryEmbedding(nn.Module):
# [...]
# -------------
# <--- CESOIA modifications

# NOTE: the next class contains linear layers that could be potentially compressed
# original --->
# class Qwen2_5_VLPatchMerger(nn.Module):
#     def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
#         super().__init__()
#         self.hidden_size = context_dim * (spatial_merge_size**2)
#         self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
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
# def rotate_half(x):
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# def apply_rotary_pos_emb_vision(
# q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
# [...]
# -------------
# <--- CESOIA modifications

# original --->
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
# class Qwen2_5_VLVisionAttention(nn.Module):
#     def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
#         super().__init__()
#         self.dim = config.hidden_size
#         self.num_heads = config.num_heads
#         self.head_dim = self.dim // self.num_heads
#         self.num_key_value_groups = 1  # needed for eager attention
#         self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
#         self.proj = nn.Linear(self.dim, self.dim)
#         self.scaling = self.head_dim**-0.5
#         self.config = config
#         self.attention_dropout = 0.0
#         self.is_causal = False
# -------------
class Qwen2_5_VLVisionAttentionCompress(Qwen2_5_VLVisionAttention):
    def __init__(self, config: Qwen2_5_VLVisionConfigCompress, path: str) -> None:
        super(Qwen2_5_VLVisionAttention, self).__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv_rank = get_validated_dict_value(config.lrd_ranks, path+".qkv", default="full", min_value=1)
        self.proj_rank = get_validated_dict_value(config.lrd_ranks, path+".proj", default="full", min_value=1)
        if FORCE_ORIGINAL_LAYERS:
            self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
            self.proj = nn.Linear(self.dim, self.dim)
        else:
            self.qkv = LinearCompressed(self.dim, self.dim * 3, bias=True, lrd_rank=self.qkv_rank)
            self.proj = LinearCompressed(self.dim, self.dim, lrd_rank=self.proj_rank)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False
# <--- CESOIA modifications

# original --->
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,-
    #     cu_seqlens: torch.Tensor,
    #     rotary_pos_emb: Optional[torch.Tensor] = None,
    #     position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    #     **kwargs,
    # ) -> torch.Tensor:
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer):
#     def __init__(self, config, attn_implementation: str = "sdpa") -> None:
#         super().__init__()
#         self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
#         self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
#         self.attn = Qwen2_5_VLVisionAttention(config=config)
#         self.mlp = Qwen2_5_VLMLP(config, bias=True)
# -------------
class Qwen2_5_VLVisionBlockCompress(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa", path: str = None) -> None:
        super(Qwen2_5_VLVisionBlock, self).__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttentionCompress(config=config, path=path+".attn")
        self.mlp = Qwen2_5_VLMLPCompress(config, bias=True, path=path+".mlp")
# <--- CESOIA modifications

# original --->
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     cu_seqlens: torch.Tensor,
    #     rotary_pos_emb: Optional[torch.Tensor] = None,
    #     position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    #     **kwargs,
    # ) -> torch.Tensor:
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
# @auto_docstring
# class Qwen2_5_VLPreTrainedModel(PreTrainedModel):
#     config: Qwen2_5_VLConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
#     _skip_keys_device_placement = "past_key_values"
#     _supports_flash_attn = True
#     _supports_sdpa = True

#     _can_compile_fullgraph = True
#     _supports_attention_backend = True
# -------------
class Qwen2_5_VLPreTrainedModelCompress(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5_VLConfigCompress
    _no_split_modules = ["Qwen2_5_VLDecoderLayerCompress", "Qwen2_5_VLVisionBlockCompress"]
# <--- CESOIA modifications

# original --->
# class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
#     config: Qwen2_5_VLVisionConfig
#     _no_split_modules = ["Qwen2_5_VLVisionBlock"]
# 
# def __init__(self, config, *inputs, **kwargs) -> None:
# -------------
class Qwen2_5_VisionTransformerPretrainedModelCompress(Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLPreTrainedModelCompress):
    config: Qwen2_5_VLVisionConfigCompress
    _no_split_modules = ["Qwen2_5_VLVisionBlockCompress"]

    def __init__(self, config, path: str, *inputs, **kwargs) -> None:
# <--- CESOIA modifications
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        # original --->
        # self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        # -------------
        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlockCompress(config, path=path+f".blocks.{i}") for i in range(config.depth)])
        # <--- CESOIA modifications

        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

# original --->
    # def rot_pos_emb(self, grid_thw):
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
    # def get_window_index(self, grid_thw):
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
    # def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
# @dataclass
# @auto_docstring(
#     custom_intro="""
#     Base class for Llava outputs, with hidden states and attentions.
#     """
# )
# class Qwen2_5_VLModelOutputWithPast(ModelOutput):
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2_5_VLRotaryEmbedding(nn.Module):
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2MLP(nn.Module):
#    def __init__(self, config):
# -------------
class Qwen2MLPCompress(nn.Module):
    def __init__(self, config, path: str):
# <--- CESOIA modifications
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
# original --->
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
# -------------
        self.gate_rank = get_validated_dict_value(config.lrd_ranks, path+".gate_proj", default="full", min_value=1)
        self.up_rank = get_validated_dict_value(config.lrd_ranks, path+".up_proj", default="full", min_value=1)
        self.down_rank = get_validated_dict_value(config.lrd_ranks, path+".down_proj", default="full", min_value=1)
        if FORCE_ORIGINAL_LAYERS:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        else:
            self.gate_proj = LinearCompressed(self.hidden_size, self.intermediate_size, bias=False, lrd_rank=self.gate_rank)
            self.up_proj = LinearCompressed(self.hidden_size, self.intermediate_size, bias=False, lrd_rank=self.up_rank)
            self.down_proj = LinearCompressed(self.intermediate_size, self.hidden_size, bias=False, lrd_rank=self.down_rank)
# <--- CESOIA modifications
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# original --->
# def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
# [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2_5_VLAttention(nn.Module):
#     """
#     Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
#     and "Generating Long Sequences with Sparse Transformers".
#     """

#     def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: Optional[int] = None):
#         super().__init__()
# -------------
class Qwen2_5_VLAttentionCompress(Qwen2_5_VLAttention):
    def __init__(self, config: Qwen2_5_VLTextConfigCompress, layer_idx: Optional[int] = None, path: str = None):
        super(Qwen2_5_VLAttention, self).__init__()
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
        self.q_rank = get_validated_dict_value(config.lrd_ranks, path+".q_proj", default="full", min_value=1)
        self.k_rank = get_validated_dict_value(config.lrd_ranks, path+".k_proj", default="full", min_value=1)
        self.v_rank = get_validated_dict_value(config.lrd_ranks, path+".v_proj", default="full", min_value=1)
        self.o_rank = get_validated_dict_value(config.lrd_ranks, path+".o_proj", default="full", min_value=1)
        if FORCE_ORIGINAL_LAYERS:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        else:
            self.q_proj = LinearCompressed(self.hidden_size, self.num_heads * self.head_dim, bias=True, lrd_rank=self.q_rank)
            self.k_proj = LinearCompressed(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, lrd_rank=self.k_rank)
            self.v_proj = LinearCompressed(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, lrd_rank=self.v_rank)
            self.o_proj = LinearCompressed(self.num_heads * self.head_dim, self.hidden_size, bias=False, lrd_rank=self.o_rank)
# <--- CESOIA modifications
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
# <--- CESOIA modifications

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
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
# class Qwen2_5_VLDecoderLayer(GradientCheckpointingLayer):
#     def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: int):
#         super().__init__()
# -------------
class Qwen2_5_VLDecoderLayerCompress(Qwen2_5_VLDecoderLayer):
    def __init__(self, config: Qwen2_5_VLTextConfigCompress, layer_idx: int, path: str):
        super(Qwen2_5_VLDecoderLayer, self).__init__()
# <--- CESOIA modifications

        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
# original --->
        # self.self_attn = Qwen2_5_VLAttention(config, layer_idx)

        # self.mlp = Qwen2MLP(config)
# -------------
        self.self_attn = Qwen2_5_VLAttentionCompress(config, layer_idx, path=path+".self_attn")

        self.mlp = Qwen2MLPCompress(config, path=path+".mlp")
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
# -------------
# <--- CESOIA modifications
    
# original --->
# @auto_docstring
# class Qwen2_5_VLTextModel(Qwen2_5_VLPreTrainedModel):
#     config: Qwen2_5_VLTextConfig

#     def __init__(self, config: Qwen2_5_VLTextConfig):
#         super().__init__(config)
# -------------
class Qwen2_5_VLTextModelCompress(Qwen2_5_VLTextModel, Qwen2_5_VLPreTrainedModelCompress):
    config: Qwen2_5_VLTextConfigCompress

    def __init__(self, config: Qwen2_5_VLTextConfigCompress, path: str):
        super(Qwen2_5_VLTextModel, self).__init__(config)
# <--- CESOIA modifications

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

# original --->
        # self.layers = nn.ModuleList(
        #     [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        # )
# -------------
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayerCompress(config, layer_idx, path=path+f".layers.{layer_idx}") for layer_idx in range(config.num_hidden_layers)]
        )
# <--- CESOIA modifications

        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
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
# -------------
# <--- CESOIA modifications

# original --->
# @auto_docstring
# class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
# -------------
class Qwen2_5_VLModelCompress(Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModelCompress):
# <--- CESOIA modifications

    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

# original --->
    # config: Qwen2_5_VLConfig
    # _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
# -------------
    config: Qwen2_5_VLConfigCompress
    _no_split_modules = ["Qwen2_5_VLDecoderLayerCompress", "Qwen2_5_VLVisionBlockCompress"]
# <--- CESOIA modifications

    def __init__(self, config):
# original --->
        # super().__init__(config)
        # self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        # self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
# -------------
        super(Qwen2_5_VLModel, self).__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModelCompress._from_config(config.vision_config, path="model.visual")
        self.language_model = Qwen2_5_VLTextModelCompress._from_config(config.text_config, path="model.language_model")
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
    #     second_per_grid_ts: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
    # def get_video_features(
    #     self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    # ):
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
    # def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
    # def get_placeholder_mask(
    #     self,
    #     input_ids: torch.LongTensor,
    #     inputs_embeds: torch.FloatTensor,
    #     image_features: torch.FloatTensor = None,
    #     video_features: torch.FloatTensor = None,
    # ):
# -------------
# <--- CESOIA modifications

# original --->
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
    #     second_per_grid_ts: Optional[torch.Tensor] = None,
    #     **kwargs: Unpack[TransformersKwargs],
    # ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
# @dataclass
# @auto_docstring(
#     custom_intro="""
#     Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.
#     """
# )
# class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
# [...]
# -------------
# <--- CESOIA modifications

# NOTE the output layer could be potentially compressed if we take into account the vocabulary we need for a specific task
# original --->
# class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
# _checkpoint_conversion_mapping = {
#         "^visual": "model.visual",
#         r"^model(?!\.(language_model|visual))": "model.language_model",
#     }
#     _tied_weights_keys = ["lm_head.weight"]
#     # Reference: fix gemma3 grad acc #37208
#     accepts_loss_kwargs = False
# -------------
class Qwen2_5_VLForConditionalGenerationCompress(Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLPreTrainedModelCompress):
    config_class = Qwen2_5_VLConfigCompress
# <--- CESOIA modifications

    def __init__(self, config):
# original --->
        # super().__init__(config)
        # self.model = Qwen2_5_VLModel(config)
        # self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
# -------------
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        self.model = Qwen2_5_VLModelCompress(config)
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
    #     second_per_grid_ts: Optional[torch.Tensor] = None,
    #     logits_to_keep: Union[int, torch.Tensor] = 0,
    #     **kwargs: Unpack[TransformersKwargs],
    # ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
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
    #     second_per_grid_ts=None,
    #     **kwargs,
    # ):
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
    # def _get_image_nums_and_video_nums(
    #     self,
    #     input_ids: Optional[torch.LongTensor],
    #     inputs_embeds: Optional[torch.Tensor] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
    # def _expand_inputs_for_generation(
    #     self,
    #     expand_size: int = 1,
    #     is_encoder_decoder: bool = False,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     **model_kwargs,
    # ) -> tuple[torch.LongTensor, dict[str, Any]]:
    # [...]
# -------------
# <--- CESOIA modifications

# original --->
# __all__ = ["Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLModel", "Qwen2_5_VLPreTrainedModel", "Qwen2_5_VLTextModel"]
# -------------
__all__ = ["Qwen2_5_VLForConditionalGenerationCompress", "Qwen2_5_VLModelCompress", "Qwen2_5_VLPreTrainedModelCompress", "Qwen2_5_VLTextModelCompress"]
# <--- CESOIA modifications
