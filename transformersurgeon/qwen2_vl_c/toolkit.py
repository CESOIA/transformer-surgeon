import torch
from ..utils import CompressionScheme

class CompressionSchemesManager:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.schemes = self._generate_schemes()

    def _generate_schemes(self):
        VISION_BLOCKS_NUM = 32
        TEXT_BLOCKS_NUM = 28

        config = self.config
        
        # Generate compression schemes for vision blocks
        vision_path_dicts = []
        key_list = ["sa_qkv", "sa_out", "mlp_up", "mlp_down"]
        path_list = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
        # output_path_lists = [
        # ]
        for i in range(VISION_BLOCKS_NUM):
            tmp_dict = {}
            for key, path in zip(key_list, path_list):
                # Check the existence of the keys
                if key in config.vision_config.pruning_ratio_lists:
                    pruning_ratio = config.vision_config.pruning_ratio_lists[key][i]
                else:
                    pruning_ratio = 0.0
                if key in config.vision_config.lrd_rank_lists:
                    lrd_rank = config.vision_config.lrd_rank_lists[key][i]
                else:
                    lrd_rank = "full"

                # Create CompressionScheme instance
                tmp_dict[key] = CompressionScheme(
                    path=f"model.visual.blocks.{i}.{path}",
                    pruning_ratio=pruning_ratio,
                    lrd_rank=lrd_rank,
                    is_qkv_concatenated=(key == "sa_qkv"),
                    module=self.model,
                )
            vision_path_dicts.append(tmp_dict)

        # Generate compression schemes for text blocks
        text_path_dicts = []
        key_list = ["sa_q", "sa_k", "sa_v", "sa_out", "mlp_gate", "mlp_up", "mlp_down"]
        path_list = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        for i in range(TEXT_BLOCKS_NUM):
            tmp_dict = {}
            for key, path in zip(key_list, path_list):
                # Check the existence of the keys
                if key in config.text_config.pruning_ratio_lists:
                    pruning_ratio = config.text_config.pruning_ratio_lists[key][i]
                else:
                    pruning_ratio = 0.0
                if key in config.text_config.lrd_rank_lists:
                    lrd_rank = config.text_config.lrd_rank_lists[key][i]
                else:
                    lrd_rank = "full"

                # Create CompressionScheme instance
                tmp_dict[key] = CompressionScheme(
                    path=f"model.language_model.layers.{i}.{path}",
                    pruning_ratio=pruning_ratio,
                    lrd_rank=lrd_rank,
                    module=self.model,
                )
            text_path_dicts.append(tmp_dict)

        return {
            "vision": vision_path_dicts,
            "text": text_path_dicts,
        }

    def __iter__(self):
        """
        Yields all CompressionScheme objects from the nested vision/text dictionaries.
        """
        for block_dicts in self.schemes.values():
            for block in block_dicts:
                for scheme in block.values():
                    # Ensure the scheme is an instance of CompressionScheme
                    if isinstance(scheme, CompressionScheme):
                        yield scheme
                    else:
                        raise TypeError(f"Expected CompressionScheme, got {type(scheme)}")
                    
    def __repr__(self):
        string = ""
        string += "  Pruning Ratio   LRD Rank   QKV Concatenated   Path\n"
        string += "|"+"-"*80 + "\n"
        for i, scheme in enumerate(self):
            if i % 2 == 1:
                string += f"| {scheme.pruning_ratio:<14}| {scheme.lrd_rank:<9}| {str(scheme.is_qkv_concatenated):<17}| {scheme.path}\n"
            else:
                string += f"  {scheme.pruning_ratio:<15} {scheme.lrd_rank:<10} {str(scheme.is_qkv_concatenated):<18} {scheme.path}\n"
        return string
                    
    def _set_model(self, model):
        """
        Sets the model for each CompressionScheme in the manager.
        """       
        # Set the model for each compression scheme
        for scheme in self:
            scheme.model = model