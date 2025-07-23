# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
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
"""Qwen2VL model configuration"""

# from ...configuration_utils import PretrainedConfig, layer_type_validation
# from ...modeling_rope_utils import rope_config_validation
# from ...utils import logging
from transformers import Qwen2VLVisionConfig, Qwen2VLTextConfig, Qwen2VLConfig
from transformers.utils import logging



logger = logging.get_logger(__name__)


class Qwen2VLVisionConfigCompress(Qwen2VLVisionConfig):
    # model_type = "qwen2_vl"
    # base_config_key = "vision_config"

    def __init__(
        self,
        # depth=32,
        # embed_dim=1280,
        # hidden_size=3584,
        # hidden_act="quick_gelu",
        # mlp_ratio=4,
        # num_heads=16,
        # in_channels=3,
        # patch_size=14,
        # spatial_merge_size=2,
        # temporal_patch_size=2,
        # initializer_range=0.02,
        pruning_list=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.depth = depth
        # self.embed_dim = embed_dim
        # self.hidden_size = hidden_size
        # self.hidden_act = hidden_act
        # self.mlp_ratio = mlp_ratio
        # self.num_heads = num_heads
        # self.in_channels = in_channels
        # self.patch_size = patch_size
        # self.spatial_merge_size = spatial_merge_size
        # self.temporal_patch_size = temporal_patch_size
        # self.initializer_range = initializer_range
        if self.pruning_list is not None:
            self.wq_pruning_list = pruning_list.get("wq", None)
            self.wk_pruning_list = pruning_list.get("wk", None)
            self.wv_pruning_list = pruning_list.get("wv", None)
            self.wo_pruning_list = pruning_list.get("wo", None)
            self.wmlpi_pruning_list = pruning_list.get("wmlpi", None)
            self.wmlpo_pruning_list = pruning_list.get("wmlpo", None)
        ''' pruning_list is a list of float values representing the structured pruning ratios
            for the respective layers in the vision tower.
            wq, wk, wv, wo, wmlpi, wmlpo correspond to the weights of the query, key, value, output in the attention block and
            intermediate, and output layers of the multi-layer perceptron in the vision tower.
            When the value in the list is None or 0, it means that the corresponding weight will not be pruned.
            When the value is a float between 0 and 1, it represents the pruning ratio for that weight.
            When the value is True or 1.0, it means that the entire block will be removed (block pruning).
        '''


class Qwen2VLTextConfigCompress(Qwen2VLTextConfig):
    # model_type = "qwen2_vl_text"
    # base_config_key = "text_config"
    # keys_to_ignore_at_inference = ["past_key_values"]
    # # Default tensor parallel plan for base model `Qwen2VL`
    # base_model_tp_plan = {
    #     "layers.*.self_attn.q_proj": "colwise",
    #     "layers.*.self_attn.k_proj": "colwise",
    #     "layers.*.self_attn.v_proj": "colwise",
    #     "layers.*.self_attn.o_proj": "rowwise",
    #     "layers.*.mlp.gate_proj": "colwise",
    #     "layers.*.mlp.up_proj": "colwise",
    #     "layers.*.mlp.down_proj": "rowwise",
    # }
    # base_model_pp_plan = {
    #     "embed_tokens": (["input_ids"], ["inputs_embeds"]),
    #     "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    #     "norm": (["hidden_states"], ["hidden_states"]),
    # }

    def __init__(
        self,
        # vocab_size=152064,
        # hidden_size=8192,
        # intermediate_size=29568,
        # num_hidden_layers=80,
        # num_attention_heads=64,
        # num_key_value_heads=8,
        # hidden_act="silu",
        # max_position_embeddings=32768,
        # initializer_range=0.02,
        # rms_norm_eps=1e-05,
        # use_cache=True,
        # tie_word_embeddings=False,
        # rope_theta=1000000.0,
        # use_sliding_window=False,
        # sliding_window=4096,
        # max_window_layers=80,
        # layer_types=None,
        # attention_dropout=0.0,
        # rope_scaling=None,
        # image_token_id=None,
        # video_token_id=None,
        pruning_list=None,
        **kwargs,
    ):
        # self.vocab_size = vocab_size
        # self.max_position_embeddings = max_position_embeddings
        # self.hidden_size = hidden_size
        # self.intermediate_size = intermediate_size
        # self.num_hidden_layers = num_hidden_layers
        # self.num_attention_heads = num_attention_heads
        # self.use_sliding_window = use_sliding_window
        # self.sliding_window = sliding_window if self.use_sliding_window else None
        # self.max_window_layers = max_window_layers

        # # for backward compatibility
        # if num_key_value_heads is None:
        #     num_key_value_heads = num_attention_heads

        # self.num_key_value_heads = num_key_value_heads
        # self.hidden_act = hidden_act
        # self.initializer_range = initializer_range
        # self.rms_norm_eps = rms_norm_eps
        # self.use_cache = use_cache
        # self.rope_theta = rope_theta
        # self.attention_dropout = attention_dropout
        # self.rope_scaling = rope_scaling

        # self.layer_types = layer_types
        # if self.layer_types is None:
        #     self.layer_types = [
        #         "sliding_attention"
        #         if self.sliding_window is not None and i >= self.max_window_layers
        #         else "full_attention"
        #         for i in range(self.num_hidden_layers)
        #     ]
        # layer_type_validation(self.layer_types)

        # # Validate the correctness of rotary position embeddings parameters
        # # BC: if there is a 'type' field, move it to 'rope_type'.
        # # and change type from 'mrope' to 'default' because `mrope` does default RoPE calculations
        # # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # # TODO: @raushan update config in the hub
        # if self.rope_scaling is not None and "type" in self.rope_scaling:
        #     if self.rope_scaling["type"] == "mrope":
        #         self.rope_scaling["type"] = "default"
        #     self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        # rope_config_validation(self, ignore_keys={"mrope_section"})
        # self.image_token_id = image_token_id
        # self.video_token_id = video_token_id

        super().__init__(**kwargs)
        if self.pruning_list is not None:
            self.wq_pruning_list = pruning_list.get("wq", None)
            self.wk_pruning_list = pruning_list.get("wk", None)
            self.wv_pruning_list = pruning_list.get("wv", None)
            self.wo_pruning_list = pruning_list.get("wo", None)
            self.wmlpi_pruning_list = pruning_list.get("wmlpi", None)
            self.wmlpo_pruning_list = pruning_list.get("wmlpo", None)


class Qwen2VLConfigCompress(Qwen2VLConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2VLModel`]. It is used to instantiate a
    Qwen2-VL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-VL-7B-Instruct [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2_5_VLTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen2_5_VLVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.

    ```python
    >>> from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig

    >>> # Initializing a Qwen2_5_VL style configuration
    >>> configuration = Qwen2_5_VLConfig()

    >>> # Initializing a model from the Qwen2-VL-7B style configuration
    >>> model = Qwen2_5_VLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2_vl"
    sub_configs = {"vision_config": Qwen2VLVisionConfigCompress, "text_config": Qwen2VLTextConfigCompress}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # For BC use all kwargs to init `TextConfig`
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super(Qwen2VLConfig, self).__init__(**kwargs)


__all__ = ["Qwen2VLConfigCompress", "Qwen2VLTextConfigCompress"]