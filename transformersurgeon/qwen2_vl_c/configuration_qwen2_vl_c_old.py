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
"""Qwen2VL model configuration - patched for compression algorithms."""

### -----------------------------------
### CESOIA PATCH
### -----------------------------------

from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLVisionConfig,
    Qwen2VLTextConfig,
    Qwen2VLConfig,
)

### -----------------------------------

from transformers.utils import logging

logger = logging.get_logger(__name__)

### -----------------------------------
### CESOIA PATCH
### -----------------------------------

''' pruning_ratio_lists and lrd_rank_lists are dictionaries containing lists of pruning ratios and ranks for each layer.
    The keys of the dictionaries are:
        - "sa_qkv": for self-attention query, key, and value weights (vision model only)
        - "sa_q": for self-attention query weights (text model only)
        - "sa_k": for self-attention key weights (text model only)
        - "sa_v": for self-attention value weights (text model only)
        - "sa_out": for self-attention output weights (lrd only)
        - "mlp_gate": for MLP gate weights # ONLY FOR TEXT CONFIG
        - "mlp_up": for MLP up weights
        - "mlp_down": for MLP down weights (lrd only)
    General-type keys can be used to prune multiple layer-types in one go:
        - "sa": apply to all the self-attention layers equally
        - "mlp": apply to all the mlp layers equally
        - "all": apply to all the layers equally

    Because of skip connections, the pruning ratio for all the layers whose output connects to skip connections should be the same.
    For this purpose, use config
        pruning_ratio_skip_connections

    For pruning_ratio_lists:
        - When the value in the list is None or 0, it means that the corresponding weight will not be pruned.
        - When the value is a float between 0 and 1, it represents the pruning ratio for that weight.
        - When the value is True or 1.0, it means that the entire block will be removed (block pruning).
    For lrd_rank_lists:
        - When the value in the list is None or 0, it means that the corresponding weight will not be compressed.
        - When the value is a positive integer, it represents the rank for that weight.
'''

'''
    Large-size modules left out from compression:
        - PatchMerger for the vision module
        - Tokenizer for the text module
'''

class Qwen2VLVisionConfigCompress(Qwen2VLVisionConfig):
    def __init__(
        self,
        pruning_ratio_lists=None,
        pruning_ratio_skip_connections=None,
        lrd_rank_lists=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        total_blocks = self.depth
        hidden_size = self.embed_dim

        # Check all the values in the dict; if they are scalar, convert them in lists with all equal values
        if pruning_ratio_lists is not None:
            for key, value in pruning_ratio_lists.items():
                if isinstance(value, (int, bool, float)):
                    pruning_ratio_lists[key] = [value] * self.depth

        if lrd_rank_lists is not None:
            for key, value in lrd_rank_lists.items():
                if isinstance(value, (int, bool, float)):
                    lrd_rank_lists[key] = [value] * self.depth

        # Assign pruning ratio for the skip connections - must be equal for all the layers connecting to skip connections
        self.pruning_ratio_skip_connections = pruning_ratio_skip_connections

        # # Define default values for pruning_ratio_lists if not provided
        # self.pruning_ratio_lists = {
        #     "sa_qkv": [0.0] * total_blocks,
        #     "mlp_up": [0.0] * total_blocks,
        # }

        # Convert scalar values in pruning_ratio_lists to lists with equal values
        if pruning_ratio_lists is not None:
            for key, value in pruning_ratio_lists.items():
                if isinstance(value, (int, bool, float)):
                    pruning_ratio_lists[key] = [value] * total_blocks

        # Update pruning_ratio_lists with the provided values
        if pruning_ratio_lists is not None:
            # Use general-type keys to substitute the keys for each layer type
            if pruning_ratio_lists.get("all") is not None:
                self.pruning_ratio_lists["sa_qkv"] = pruning_ratio_lists["all"]
                self.pruning_ratio_lists["mlp_up"] = pruning_ratio_lists["all"]
            if pruning_ratio_lists.get("sa") is not None:
                self.pruning_ratio_lists["sa_qkv"] = pruning_ratio_lists["sa"]
            if pruning_ratio_lists.get("mlp") is not None:
                self.pruning_ratio_lists["mlp_up"] = pruning_ratio_lists["mlp"]

            # Substitute the keys for each layer type if they are defined
            self.pruning_ratio_lists.update(pruning_ratio_lists)

        # # Define default values for lrd_rank_lists if not provided
        # self.lrd_rank_lists = {
        #     "sa_qkv": ["full"] * total_blocks,
        #     "sa_out": ["full"] * total_blocks,
        #     "mlp_up": ["full"] * total_blocks,
        #     "mlp_down": ["full"] * total_blocks,
        # }

        # Convert scalar values in lrd_rank_lists to lists with equal values
        if lrd_rank_lists is not None:
            for key, value in lrd_rank_lists.items():
                if isinstance(value, (int, bool, float)):
                    lrd_rank_lists[key] = [value] * total_blocks

        # Update lrd_rank_lists with the provided values
        if lrd_rank_lists is not None:
            # Use general-type keys to substitute the keys for each layer type
            if lrd_rank_lists.get("all") is not None:
                self.lrd_rank_lists["sa_qkv"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["sa_out"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["mlp_up"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["mlp_down"] = lrd_rank_lists["all"]
            if lrd_rank_lists.get("sa") is not None:
                self.lrd_rank_lists["sa_qkv"] = lrd_rank_lists["sa"]
                self.lrd_rank_lists["sa_out"] = lrd_rank_lists["sa"]
            if lrd_rank_lists.get("mlp") is not None:
                self.lrd_rank_lists["mlp_up"] = lrd_rank_lists["mlp"]
                self.lrd_rank_lists["mlp_down"] = lrd_rank_lists["mlp"]

            # Substitute the keys for each layer type if they are defined
            self.lrd_rank_lists.update(lrd_rank_lists)

        # Infer embed dimension of skip connections
        if self.pruning_ratio_skip_connections is not None:
            # Check value of the pruning ratio for skip connections
            if not (0.0 <= self.pruning_ratio_skip_connections <= 1.0):
                raise ValueError(
                    f"pruning_ratio_skip_connections must be between 0.0 and 1.0, but got {self.pruning_ratio_skip_connections}."
                )
            # Calculate the pruned embed dimension based on the pruning ratio for skip connections
            self.pruned_embed_dim = hidden_size - int(hidden_size * self.pruning_ratio_skip_connections)
        else:
            # If no pruning ratio is defined, use the original embed dimension
            self.pruned_embed_dim = hidden_size

        # Infer embed dimension from the pruning dict
        self.pruned_dim_lists = {}
        for key, value in self.pruning_ratio_lists.items():
            if value is not None and isinstance(value, list):
                pruned_dim_list = []
                for i, v in enumerate(value):
                    # Check if value is valid
                    if not (0.0 <= v <= 1.0):
                        raise ValueError("All pruning ratios must be between 0.0 and 1.0.")
                    # Calculate the pruned dimension
                    if key == "mlp_up":
                        embed_dim = hidden_size * self.mlp_ratio
                    else:
                        embed_dim = hidden_size
                    pruned_dim = embed_dim - int(embed_dim * v)
                    # Ensure that the size of the attention heads is divisible by the number of heads
                    if key == "sa_qkv":
                        pruned_dim = (pruned_dim // self.num_heads) * self.num_heads
                    pruned_dim_list.append(pruned_dim)
                self.pruned_dim_lists[key] = pruned_dim_list

class Qwen2VLTextConfigCompress(Qwen2VLTextConfig):
    def __init__(
        self,
        pruning_ratio_lists=None,
        pruning_ratio_skip_connections=None,
        lrd_rank_lists=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        total_blocks = self.num_hidden_layers
        hidden_size = self.hidden_size

        # Assign pruning ratio for the skip connections - must be equal for all the layers connecting to skip connections
        self.pruning_ratio_skip_connections = pruning_ratio_skip_connections

        # Define default values for pruning_ratio_lists if not provided
        # self.pruning_ratio_lists = {
        #     "sa_qkv": [0.0] * total_blocks,
        #     "mlp_gate": [0.0] * total_blocks,
        #     "mlp_up": [0.0] * total_blocks,        
        # }

        # Convert scalar values in pruning_ratio_lists to lists with equal values
        if pruning_ratio_lists is not None:
            for key, value in pruning_ratio_lists.items():
                if isinstance(value, (int, bool, float)):
                    pruning_ratio_lists[key] = [value] * total_blocks

        # Update pruning_ratio_lists with the provided values
        if pruning_ratio_lists is not None:
            # Use general-type keys to substitute the keys for each layer type
            if pruning_ratio_lists.get("all") is not None:
                self.pruning_ratio_lists["sa_qkv"] = pruning_ratio_lists["all"]
                self.pruning_ratio_lists["mlp_gate"] = pruning_ratio_lists["all"]
                self.pruning_ratio_lists["mlp_up"] = pruning_ratio_lists["all"]
            if pruning_ratio_lists.get("sa") is not None:
                self.pruning_ratio_lists["sa_q"] = pruning_ratio_lists["sa"]
                self.pruning_ratio_lists["sa_k"] = pruning_ratio_lists["sa"]
                self.pruning_ratio_lists["sa_v"] = pruning_ratio_lists["sa"]
            if pruning_ratio_lists.get("mlp") is not None:
                self.pruning_ratio_lists["mlp_gate"] = pruning_ratio_lists["mlp"]
                self.pruning_ratio_lists["mlp_up"] = pruning_ratio_lists["mlp"]

            # Substitute the keys for each layer type if they are defined
            self.pruning_lists.update(pruning_ratio_lists)

        # Define default values for lrd_rank_lists if not provided
        # self.lrd_rank_lists = {
        #     "sa_q": ["full"] * total_blocks,
        #     "sa_k": ["full"] * total_blocks,
        #     "sa_v": ["full"] * total_blocks,
        #     "sa_out": ["full"] * total_blocks,
        #     "mlp_gate": ["full"] * total_blocks,
        #     "mlp_up": ["full"] * total_blocks,
        #     "mlp_down": ["full"] * total_blocks,
        # }

        # Convert scalar values in lrd_rank_lists to lists with equal values
        if lrd_rank_lists is not None:
            for key, value in lrd_rank_lists.items():
                if isinstance(value, (int, bool, float)):
                    lrd_rank_lists[key] = [value] * total_blocks

        # Update lrd_rank_lists with provided values
        if lrd_rank_lists is not None:
            # Use general-type keys to substitute the keys for each layer type
            if lrd_rank_lists.get("all") is not None:
                self.lrd_rank_lists["sa_q"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["sa_k"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["sa_v"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["sa_out"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["mlp_gate"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["mlp_up"] = lrd_rank_lists["all"]
                self.lrd_rank_lists["mlp_down"] = lrd_rank_lists["all"]
            if lrd_rank_lists.get("sa") is not None:
                self.lrd_rank_lists["sa_q"] = lrd_rank_lists["sa"]
                self.lrd_rank_lists["sa_k"] = lrd_rank_lists["sa"]
                self.lrd_rank_lists["sa_v"] = lrd_rank_lists["sa"]
                self.lrd_rank_lists["sa_out"] = lrd_rank_lists["sa"]
            if lrd_rank_lists.get("sa_qkv") is not None:
                self.lrd_rank_lists["sa_q"] = lrd_rank_lists["sa_qkv"]
                self.lrd_rank_lists["sa_k"] = lrd_rank_lists["sa_qkv"]
                self.lrd_rank_lists["sa_v"] = lrd_rank_lists["sa_qkv"]
            if lrd_rank_lists.get("mlp") is not None:
                self.lrd_rank_lists["mlp_gate"] = lrd_rank_lists["mlp"]
                self.lrd_rank_lists["mlp_up"] = lrd_rank_lists["mlp"]
                self.lrd_rank_lists["mlp_down"] = lrd_rank_lists["mlp"]

            # Substitute the keys for each layer type if they are defined
            self.lrd_rank_lists.update(lrd_rank_lists)       

        if self.pruning_ratio_skip_connections is not None:
            # Check value of the pruning ratio for skip connections
            if not (0.0 <= self.pruning_ratio_skip_connections <= 1.0):
                raise ValueError(
                    f"pruning_ratio_skip_connections must be between 0.0 and 1.0, but got {self.pruning_ratio_skip_connections}."
                )
            # Calculate the pruned embed dimension based on the pruning ratio for skip connections
            self.pruned_hidden_size = hidden_size - int(hidden_size * self.pruning_ratio_skip_connections)
        else:
            # If no pruning ratio is defined, use the original embed dimension
            self.pruned_hidden_size = self.hidden_size

        # Infer embed dimension from the pruning dict
        self.pruned_dim_lists = {}
        for key, value in self.pruning_ratio_lists.items():
            if value is not None and isinstance(value, list):
                pruned_dim_list = []
                for i, v in enumerate(value):
                    # Check if value is valid
                    if not (0.0 <= v <= 1.0):
                        raise ValueError("All pruning ratios must be between 0.0 and 1.0.")
                    # Calculate the pruned dimension
                    if key == "mlp_up":
                        embed_dim = self.intermediate_size
                    else:
                        embed_dim = self.hidden_size
                    pruned_dim = hidden_size - int(hidden_size * v)
                    # Ensure that the size of the attention heads is divisible by the number of heads
                    if key == "sa_q" or key == "sa_k" or key == "sa_v":
                        pruned_dim = (pruned_dim // self.num_heads) * self.num_heads
                    pruned_dim_list.append(pruned_dim)
                self.pruned_dim_lists[key] = pruned_dim_list


### -----------------------------------


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

    ### -----------------------------------
    # CESOIA PATCH: substitute the original vision and text config classes with the compressed versions
    ### -----------------------------------

    sub_configs = {"vision_config": Qwen2VLVisionConfigCompress, "text_config": Qwen2VLTextConfigCompress}

    ### -----------------------------------

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