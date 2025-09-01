import torch
from .compression import CompressionScheme

class CompressionSchemesManager:
    """
    Generic compression schemes manager that can be used across different models.
    Model-specific configurations should be provided through model_config parameter.
    """
    
    def __init__(self, config, model, model_config):
        """
        Initialize the compression manager.
        
        Args:
            config: The main configuration object
            model: The model to apply compression to
            model_config: Model-specific configuration containing:
                - block_configs: List of block configurations, each containing:
                    - name: Name of the block type (e.g., 'vision', 'text')
                    - num_blocks: Number of blocks
                    - key_list: List of compression keys
                    - path_list: List of corresponding module paths
                    - path_template: Template for generating full paths
                    - config_attr: Attribute name in main config for this block type
        """
        self.config = config
        self.model = model
        self.model_config = model_config
        self.schemes = self._generate_schemes()

    def apply_all(self, hard=False, verbose=False):
        """
        Applies all compression schemes to their respective modules in the model.
        """
        for scheme in self:
            scheme.apply(hard=hard, verbose=verbose)

    def restore_all(self, verbose=False):
        """
        Restores all modules to their original state by removing pruning and LRD.
        """
        for scheme in self:
            scheme.restore(verbose=verbose)

    def _generate_schemes(self):
        """
        Generate compression schemes based on model configuration.
        """
        all_schemes = {}
        
        for block_config in self.model_config['block_configs']:
            block_name = block_config['name']
            num_blocks = block_config['num_blocks']
            key_list = block_config['key_list']
            path_list = block_config['path_list']
            path_template = block_config['path_template']
            config_attr = block_config['config_attr']
            
            # Get the specific config for this block type
            block_specific_config = getattr(self.config, config_attr)
            
            block_schemes = []
            for i in range(num_blocks):
                tmp_dict = {}
                for key, path in zip(key_list, path_list):
                    # Check the existence of the keys
                    if hasattr(block_specific_config, 'pruning_ratio_lists') and key in block_specific_config.pruning_ratio_lists:
                        pruning_ratio = block_specific_config.pruning_ratio_lists[key][i]
                    else:
                        pruning_ratio = 0.0
                        
                    if hasattr(block_specific_config, 'lrd_rank_lists') and key in block_specific_config.lrd_rank_lists:
                        lrd_rank = block_specific_config.lrd_rank_lists[key][i]
                    else:
                        lrd_rank = "full"

                    # Create CompressionScheme instance
                    full_path = path_template.format(block_index=i, path=path)
                    
                    # Check if this is a QKV concatenated layer
                    is_qkv_concatenated = block_config.get('qkv_keys', [])
                    is_qkv = key in is_qkv_concatenated if is_qkv_concatenated else False
                    
                    tmp_dict[key] = CompressionScheme(
                        path=full_path,
                        pruning_ratio=pruning_ratio,
                        lrd_rank=lrd_rank,
                        is_qkv_concatenated=is_qkv,
                        module=self.model,
                    )
                block_schemes.append(tmp_dict)
            
            all_schemes[block_name] = block_schemes
        
        return all_schemes

    def __iter__(self):
        """
        Yields all CompressionScheme objects from the nested dictionaries.
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

__all__ = ["CompressionSchemesManager"]