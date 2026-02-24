# coding=utf-8
# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""DistilBERT model configuration"""

from transformers.models.distilbert.configuration_distilbert import DistilBertConfig

from ...utils.configuration import init_compression_config

from .indexing_distilbert_c import DISTILBERT_C_INDEXING as INDEXING


class DistilBertConfigCompress(DistilBertConfig):
    model_type = "distilbert"

    def __init__(
        self,
        pruning_ratios=None,
        lrd_ranks=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        init_compression_config(
            config_instance=self,
            indexing=INDEXING["distilbert"],
            pruning_ratios=pruning_ratios,
            lrd_ranks=lrd_ranks,
        )


__all__ = ["DistilBertConfigCompress"]
