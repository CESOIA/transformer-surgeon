"""
manager.py

Provides the CompressionSchemesManager class for managing multiple compression schemes in transformer models.
"""

import torch
from typing import Dict, List, Any
from .compression import CompressionScheme

class CompressionSchemesManager:
    """
    Manages multiple compression schemes for transformer models.

    Args:
        config (Dict[str, Any]): Configuration dictionary for compression. You can use model.config.to_dict() to get this.
        model (torch.nn.Module): The model to be compressed.
        indexing (List[Dict[str, Any]]): List of block-specific configuration dictionaries.

    Methods:
        init_vcon: Initialize VCON blocks for selected modules.
        cancel_vcon: Remove VCON blocks and restore original modules.
        set_vcon_beta: Set the beta parameter for VCON blocks.
        freeze_uncompressed_vcon: Freeze original blocks in VCON.
        apply: Apply compression schemes to the model.
        restore: Restore the model to its original state.
        init_vcon_all: Initialize VCON for all modules.
        cancel_vcon_all: Cancel VCON for all modules.
        set_vcon_beta_all: Set beta for all VCON blocks.
        freeze_uncompressed_vcon_all: Freeze all uncompressed VCON blocks.
        apply_all: Apply all compression schemes.
        restore_all: Restore all modules.
    """
    
    def __init__(self,
                 config:Dict[str, Any],
                 model:torch.nn.Module,
                 indexing:List[Dict[str, Any]]):
        """
        Initialize the compression manager.
        
        Args:
            config: The main configuration object
            model: The model to apply compression to
            indexing: Model-specific indexing containing:
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
        self.indexing = indexing
        self.schemes = self._generate_schemes()

    def init_vcon(self, criteria=None, verbose=False):
        """
        Initializes VCON blocks for filtered modules

        Args:
            criteria: List of criteria to filter modules (by name or block_id)
            verbose: If True, prints information about the initialization process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.init_vcon(verbose=verbose)

    def cancel_vcon(self, criteria=None, keep_block_b=True, verbose=False):
        """
        Cancels VCON blocks for filtered modules, keeping either block_a or block_b

        Args:
            criteria: List of criteria to filter modules (by name or block_id)
            keep_block_b: If True, keeps the compressed block (block_b); otherwise keeps the original block (block_a)
            verbose: If True, prints information about the cancellation process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.cancel_vcon(keep_block_b=keep_block_b, verbose=verbose)

    def set_vcon_beta(self, beta: float, criteria=None, verbose=False):
        """
        Sets the beta value for filtered VCON-initialized blocks

        Args:
            beta: The beta value to set (0 <= beta <= 1)
            criteria: List of criteria to filter modules (by name or block_id)
            verbose: If True, prints information about the beta setting process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            if scheme.vcon_initialized:
                scheme.set_vcon_beta(beta, verbose=verbose)

    def freeze_uncompressed_vcon(self, criteria=None, verbose=False):
        """
        Freezes uncompressed blocks in filtered VCON-initialized modules

        Args:
            criteria: List of criteria to filter modules (by name or block_id)
            verbose: If True, prints information about the freezing process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            if scheme.vcon_initialized:
                scheme.freeze_uncompressed_block(verbose=verbose)

    def apply(self, criteria=None, hard=False, verbose=False):
        """
        Applies filtered compression schemes to their respective modules in the model.

        Args:
            criteria: List of criteria to filter modules (by name or block_id)
            hard: If True, applies hard compression (non-reversible); if False, applies soft compression (reversible)
            verbose: If True, prints information about the application process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.apply(hard=hard, verbose=verbose)

    def restore(self, criteria=None, verbose=False):
        """
        Restores filtered modules to their original state by removing pruning and LRD.

        Args:
            criteria: List of criteria to filter modules (by name or block_id)
            verbose: If True, prints information about the restoration process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.restore(verbose=verbose)

    # aliases for "all" criteria
    def init_vcon_all(self, verbose=False):
        """
        Alias for init_vcon with criteria set to "all"
        """
        self.init_vcon(criteria=["all"], verbose=verbose)

    def cancel_vcon_all(self, keep_block_b=True, verbose=False):
        """
        Alias for cancel_vcon with criteria set to "all"
        """
        self.cancel_vcon(criteria=["all"], keep_block_b=keep_block_b, verbose=verbose)

    def set_vcon_beta_all(self, beta: float, verbose=False):
        """
        Alias for set_vcon_beta with criteria set to "all"
        """
        self.set_vcon_beta(beta, criteria=["all"], verbose=verbose)

    def freeze_uncompressed_vcon_all(self, verbose=False):
        """
        Alias for freeze_uncompressed_vcon with criteria set to "all"
        """
        self.freeze_uncompressed_vcon(criteria=["all"], verbose=verbose)

    def apply_all(self, hard=False, verbose=False):
        """
        Alias for apply with criteria set to "all"
        """
        self.apply(criteria=["all"], hard=hard, verbose=verbose)
        
    def restore_all(self, verbose=False):
        """
        Alias for restore with criteria set to "all"
        """
        self.restore(criteria=["all"], verbose=verbose)

    def _generate_schemes(self):
        """
        Generate compression schemes based on model configuration and indexing.
        Returns:
            Dict[str, List[Dict[str, CompressionScheme]]]: Nested dictionary of compression schemes organized by block type and block index.
        """
        all_schemes = {}
        
        for block_name, block_indexing in self.indexing.items():
            config_attr = block_indexing.get('config_attr', None)
            num_blocks_attr = block_indexing['num_blocks_attr']
            key_list = block_indexing['key_list']
            path_list = block_indexing['path_list']
            path_template = block_indexing['path_template']
            config_attr = block_indexing['config_attr']

            # Get the specific config for this block type
            if config_attr is None:
                block_specific_config = self.config
            else:
                block_specific_config = self.config[config_attr]

            # Get blocks number
            num_blocks = block_specific_config[num_blocks_attr]

            block_schemes = []
            for i in range(num_blocks):
                tmp_dict = {}
                for key, path in zip(key_list, path_list):
                    # Check the existence of the keys
                    pruning_ratio_lists = block_specific_config.get('pruning_ratio_lists', {})
                    if key in pruning_ratio_lists:
                        pruning_ratio = pruning_ratio_lists[key][i]
                    else:
                        pruning_ratio = 0.0
                        
                    lrd_rank_lists = block_specific_config.get('lrd_rank_lists', {})
                    if key in lrd_rank_lists:
                        lrd_rank = lrd_rank_lists[key][i]
                    else:
                        lrd_rank = "full"

                    # Create CompressionScheme instance
                    full_path = path_template.format(block_index=i, path=path)
                    
                    # Check if this is a QKV concatenated layer
                    is_qkv_concatenated = block_indexing.get('qkv_keys', [])
                    is_qkv = key in is_qkv_concatenated if is_qkv_concatenated else False
                    
                    tmp_dict[key] = CompressionScheme(
                        name=key,
                        block_id=i,
                        path=full_path,
                        pruning_ratio=pruning_ratio,
                        lrd_rank=lrd_rank,
                        is_qkv_concatenated=is_qkv,
                        model=self.model,
                    )
                block_schemes.append(tmp_dict)
            
            all_schemes[block_name] = block_schemes
        
        return all_schemes
    
    def iter_filtered(self, criteria:list=None):
        """
        Yields CompressionScheme objects filtered by name and/or block_id.
        
        Args:
            criteria (list): List of criteria to filter schemes. If even one of the criteria is not met, the scheme is skipped. Can include:
                - int: Block ID to match
                - str: Substring to match in the scheme name or path
                - "all": Matches all schemes
        """
        if type(criteria) != list:
            criteria = [criteria]
        for scheme in self:
            select = True
            # Verify if all criteria are met
            for crit in criteria:
                if crit is None:
                    continue
                elif crit in ["all", "ALL"]:
                    continue
                elif isinstance(crit, int):
                    if scheme.block_id != crit:
                        select = False
                        break
                elif isinstance(crit, str):
                    if crit not in scheme.name and crit not in scheme.path:
                        select = False
                        break
            if select:
                yield scheme

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
        string += "  Name      Block id  Pruning Ratio   LRD Rank   QKV Concatenated   Path\n"
        string += "|"+"-"*100 + "\n"
        for i, scheme in enumerate(self):
            if i % 2 == 1:
                string += f"| {scheme.name:<9}| {scheme.block_id:<10}| {scheme.pruning_ratio:<14}| {scheme.lrd_rank:<9}| {str(scheme.is_qkv_concatenated):<17}| {scheme.path:<50}|\n"
            else:
                string += f"  {scheme.name:<9}  {scheme.block_id:<10}  {scheme.pruning_ratio:<14}  {scheme.lrd_rank:<9}  {str(scheme.is_qkv_concatenated):<17}  {scheme.path:<50} \n"
        return string
                    
    def _set_model(self, model):
        """
        Sets the model for each CompressionScheme in the manager.
        Args:
            model (torch.nn.Module): The model to set for each compression scheme.
        """       
        # Set the model for each compression scheme
        for scheme in self:
            scheme.model = model

__all__ = ["CompressionSchemesManager"]