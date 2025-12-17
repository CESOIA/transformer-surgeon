# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
manager.py

Provides the CompressionSchemesManager class for managing multiple compression schemes in transformer models.
"""

import torch
from transformers import PretrainedConfig
from typing import Dict, List, Any, Union
import warnings
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
                 model: torch.nn.Module,
                 indexing: List[Dict[str, Any]]
                 ):
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
        self.model = model
        try:
            self.config = model.config
        except AttributeError:
            raise ValueError("The provided model does not have a 'config' attribute. Please provide a model with a valid configuration.")
        self.indexing = indexing
        self.schemes = self._generate_schemes()

    def set_pruning_ratio(self, ratio: float, criteria=None, verbose=False):
        """
        Sets the pruning ratio for filtered modules.

        Args:
            ratio: The pruning ratio to set (0.0 to 1.0)
            criteria: List of criteria to filter modules (by name or block_id)
        """
        for scheme in self.iter_filtered(criteria=criteria):
            if scheme.soft_applied or scheme.hard_applied:
                raise RuntimeError(f"Cannot set pruning ratio for {scheme.path} because compression has already been applied.")
            scheme.pruning_ratio = ratio
            if verbose:
                print(f"Set pruning ratio of {ratio} for {scheme.path}")
            
    def set_pruning_mode(self, mode: str, criteria=None, verbose=False):
        """
        Sets the pruning mode for filtered modules.

        Args:
            mode: The pruning mode to set ('structured' or 'unstructured')
            criteria: List of criteria to filter modules (by name or block_id)
        """
        for scheme in self.iter_filtered(criteria=criteria):
            if scheme.soft_applied or scheme.hard_applied:
                raise RuntimeError(f"Cannot set pruning mode for {scheme.path} because compression has already been applied.")
            scheme.pruning_mode = mode
            if verbose:
                print(f"Set pruning mode of {mode} for {scheme.path}")

    def set_lrd_rank(self, rank: Union[int, str], criteria=None, verbose=False):
        """
        Sets the LRD rank for filtered modules.

        Args:
            rank: The LRD rank to set (int or "full")
            criteria: List of criteria to filter modules (by name or block_id)
        """
        for scheme in self.iter_filtered(criteria=criteria):
            if scheme.soft_applied or scheme.hard_applied:
                raise RuntimeError(f"Cannot set LRD rank for {scheme.path} because compression has already been applied.")
            scheme.lrd_rank = rank
            if verbose:
                print(f"Set LRD rank of {rank} for {scheme.path}")

    def init_vcon(self, criteria=None, verbose=False, variant=False):
        """
        Initializes VCON blocks for filtered modules

        Args:
            criteria: List of criteria to filter modules (by name or block_id)
            verbose: If True, prints information about the initialization process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.init_vcon(verbose=verbose, variant=variant)

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
                scheme.freeze_uncompressed_vcon(verbose=verbose)

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
    def set_pruning_ratio_all(self, ratio: float, verbose=False):
        """
        Alias for set_pruning_ratio with criteria set to "all"
        """
        self.set_pruning_ratio(ratio, criteria=["all"], verbose=verbose)

    def set_pruning_mode_all(self, mode: str, verbose=False):
        """
        Alias for set_pruning_mode with criteria set to "all"
        """
        self.set_pruning_mode(mode, criteria=["all"], verbose=verbose)

    def set_lrd_rank_all(self, rank: Union[int, str], verbose=False):
        """
        Alias for set_lrd_rank with criteria set to "all"
        """
        self.set_lrd_rank(rank, criteria=["all"], verbose=verbose)

    def init_vcon_all(self, verbose=False, variant=False):
        """
        Alias for init_vcon with criteria set to "all"
        """
        self.init_vcon(criteria=["all"], verbose=verbose, variant=variant)

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

        config = self.config.to_dict() if isinstance(self.config, PretrainedConfig) else self.config
        
        for block_name, block_indexing in self.indexing.items():
            config_attr = block_indexing.get('config_attr', None)
            num_blocks_attr = block_indexing['num_blocks_attr']
            path_list = block_indexing['path_list']
            path_template = block_indexing['path_template']
            config_attr = block_indexing['config_attr']

            # Get the specific config for this block type
            if config_attr == '':
                block_specific_config = config
            else:
                block_specific_config = config[config_attr]

            # Get blocks number
            num_blocks = block_specific_config[num_blocks_attr]

            tmp_dict = {}
            for i in range(num_blocks):
                for path in path_list:
                    # Create CompressionScheme instance
                    full_path = path_template.format(block_index=i, path=path)

                    # Get pruning ratio and LRD rank from config
                    pruning_ratio = block_specific_config.get('pruning_ratios', {}).get(full_path, 0.0)
                    pruning_mode = block_specific_config.get('pruning_modes', {}).get(full_path, "structured")
                    lrd_rank = block_specific_config.get('lrd_ranks', {}).get(full_path, "full")
                    tmp_dict[full_path] = CompressionScheme(
                        name=path,
                        block_id=i,
                        path=full_path,
                        pruning_ratio=pruning_ratio,
                        pruning_mode=pruning_mode,
                        lrd_rank=lrd_rank,
                        model=self.model,
                    )
            
            all_schemes[block_name] = tmp_dict
        
        return all_schemes
    
    def print_filtered(self, criteria:list=None):
        """
        Prints CompressionScheme objects filtered by name and/or block_id.

        Args:
            criteria (list): List of criteria to filter schemes. If even one of the criteria is not met, the scheme is skipped.
        """
        for scheme in self.iter_filtered(criteria=criteria):
            print(scheme)
    
    def iter_filtered(self, criteria:Union[list, int, str]=None):
        """
        Yields CompressionScheme objects filtered by name and/or block_id.

        Args:
            criteria (list): List of criteria to filter schemes. If even one of the criteria is met, the scheme is kept. Can include:
                - int: Block ID to match
                - str: Substring to match in the scheme name or path
                - "all": Matches all schemes
                - list: A list of criteria, where all of them must be met (AND logic within the list)
        """
        if type(criteria) != list:
            criteria = [criteria]
        for scheme in self:
            select = False
            # Verify if all criteria are met
            for or_crit in criteria: # Use AND logic for criteria
                if or_crit in ["all", "ALL", "All"]:
                    select = True
                    break
                elif isinstance(or_crit, int):
                    if scheme.block_id == or_crit:
                        select = True
                        break
                elif isinstance(or_crit, str):
                    if or_crit in scheme.name or or_crit in scheme.path:
                        select = True
                        break
                elif isinstance(or_crit, list): # If a list is provided, use AND logic within the list
                    tmp_select = True
                    for and_crit in or_crit:
                        if and_crit is None:
                            tmp_select = False
                            break
                        elif and_crit in ["all", "ALL", "All"]:
                            continue
                        elif isinstance(and_crit, int):
                            if scheme.block_id != and_crit:
                                tmp_select = False
                                break
                        elif isinstance(and_crit, str):
                            if and_crit not in scheme.name and and_crit not in scheme.path:
                                tmp_select = False
                                break
                    if tmp_select:
                        select = True
                        break
            
            if select:
                yield scheme

    def update_config(self, verbose=False):
        """
        Updates the model's configuration object with the current pruning ratios and LRD ranks from all CompressionScheme objects.
        Modifications are made in-place.
        
        Returns:
            The updated configuration object.
        """
        config_names = [block['config_attr'] for block in self.indexing.values()]

        total_updates = 0
        for cname, block_dicts in zip(config_names, self.schemes.values()):
            for scheme in block_dicts.values():
                if scheme.hard_applied:
                    if cname == '':
                        self.config.pruning_ratios[scheme.path] = scheme.pruning_ratio
                        self.config.pruning_modes[scheme.path] = scheme.pruning_mode
                        self.config.lrd_ranks[scheme.path] = scheme.lrd_rank
                    else:
                        getattr(self.config, cname).pruning_ratios[scheme.path] = scheme.pruning_ratio
                        getattr(self.config, cname).pruning_modes[scheme.path] = scheme.pruning_mode
                        getattr(self.config, cname).lrd_ranks[scheme.path] = scheme.lrd_rank
                    if verbose:
                        print(f"Updated config for {scheme.path}:"
                              f"pruning_ratio={scheme.pruning_ratio},"
                              f"pruning_mode={scheme.pruning_mode},"
                              f"lrd_rank={scheme.lrd_rank}")
                    total_updates += 1
        if total_updates == 0:
            warnings.warn("No compression has been applied to configuration. Check if compression was applied in hard mode.")

        return self.config

    def __iter__(self):
        """
        Yields all CompressionScheme objects from the nested dictionaries.
        """
        for block_dicts in self.schemes.values():
            for scheme in block_dicts.values():
                # Ensure the scheme is an instance of CompressionScheme
                if isinstance(scheme, CompressionScheme):
                    yield scheme
                else:
                    raise TypeError(f"Expected CompressionScheme, got {type(scheme)}")
                    
    def __repr__(self):
        string = ""
        string += "  Prune Rat.   LRD Rank   Soft       Hard       Path\n"
        string += "|"+"-"*100 + "\n"
        for i, scheme in enumerate(self):
            if i % 2 == 1:
                string += f"| {scheme.pruning_ratio:<10} | {scheme.lrd_rank:<10} | {scheme.soft_applied:<10} | {scheme.hard_applied:<10} | {scheme.path:<60}|\n"
            else:
                string += f"  {scheme.pruning_ratio:<10}   {scheme.lrd_rank:<10}   {scheme.soft_applied:<10}   {scheme.hard_applied:<10}   {scheme.path:<60} \n"
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