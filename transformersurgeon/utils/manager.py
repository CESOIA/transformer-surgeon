# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
manager.py

Provides the CompressionSchemesManager class for managing multiple compression schemes in transformer models.
"""

import logging
import torch
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy

logger = logging.getLogger(__name__)
from ..calibration import run_compression_calibration
from .scheme import CompressionScheme
from .utils import flatten_index_paths
from .cascade import apply_cascade

class CompressionSchemesManager:
    """
    Manages multiple compression schemes across different modules of a transformer model.
    
    Core features:
    - Setting compression properties (rank, ratio, method) on filtered modules.
    - Initializing and managing VCON (Virtual Compression On-demand Network) blocks.
    - Applying compression schemes with optional calibration for activation-aware methods.
    - Restoring original model state.
    
        Calibration:
        - Uses a single, method-agnostic calibration pass to collect shared activation
            statistics for all compressors that require calibration.
        - Each compressor reports needed summary names when calibration is required.
        - Calibration data is managed by the manager and consumed by the
            run_compression_calibration utility.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 indexing: List[Dict[str, Any]]
                 ):
        """
        Initialize the compression manager.
        
        Args:
            model: The model to apply compression to
            indexing: Model-specific indexing
        """
        self.model = model
        self.calibration_data = None
        self.calibration_loss_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None
        self.calibration_mode = "standard"
        self.calibration_groups: List[List[Union[int, str, list]]] = []
        try:
            self.config = model.config
            assert isinstance(self.config, PretrainedConfig), "Model config is not an instance of PretrainedConfig. Please provide a model with a valid Hugging Face configuration."
        except AttributeError:
            raise ValueError("The provided model does not have a 'config' attribute. Please provide a model with a valid configuration.")
        self.indexing = indexing
        self.calibration_groups = self._get_calibration_groups_from_indexing()
        self.schemes = self._generate_schemes()

    def _get_calibration_groups_from_indexing(self):
        groups = []
        for _, block_indexing in self.indexing.items():
            block_groups = block_indexing.get("calibration_groups", [])

            # Structured format by subblock:
            #   {
            #     'self_attn': [['q_proj', 'k_proj', 'v_proj'], ['o_proj']],
            #     'mlp': [['down_proj'], ['gate_proj', 'up_proj']],
            #   }
            if isinstance(block_groups, dict):
                for subblock_name, subblock_groups in block_groups.items():
                    if not isinstance(subblock_groups, list):
                        raise TypeError(
                            "Each 'calibration_groups' dict value must be a list of layer groups. "
                            f"Invalid value type for key '{subblock_name}': {type(subblock_groups)}."
                        )

                    for group_idx, group_layers in enumerate(subblock_groups):
                        if not isinstance(group_layers, list):
                            raise TypeError(
                                "Each layer group in 'calibration_groups' dict must be a list of layer names. "
                                f"Invalid group at key '{subblock_name}' index {group_idx}: {type(group_layers)}."
                            )

                        normalized_group = []
                        for layer_name in group_layers:
                            layer_name = str(layer_name)
                            if "." in layer_name or not str(subblock_name):
                                normalized_group.append(layer_name)
                            else:
                                normalized_group.append(f"{subblock_name}.{layer_name}")

                        groups.append(normalized_group)
                continue

            raise TypeError(
                "Indexing field 'calibration_groups' must be a list or a dict. "
                f"Got {type(block_groups)}."
            )
        return groups

    def set(self, compression, property, value, criteria=None, verbose=False):
        """
        Generic setter for compression properties based on criteria.

        Args:
            compression: The type of compression to set (e.g., 'pruning', 'lrd', 'quantization')
            property: The name of the property to set (e.g., 'ratio', 'rank')
            value: The value to set for the specified property
            criteria: List of criteria to filter modules (by name or block_id). No criteria = all
            verbose: If True, prints information about the setting process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.set(compression, property, value, verbose=verbose)

    def init_vcon(self, criteria=None, verbose=False):
        """
        Initializes VCON blocks for filtered modules.

        Args:
            criteria: List of criteria to filter modules (by name or block_id). No criteria = all
            verbose: If True, prints information about the initialization process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.init_vcon(verbose=verbose)

    def cancel_vcon(self, keep_block_b=True, criteria=None, verbose=False):
        """
        Cancels VCON blocks for filtered modules, keeping either block_a or block_b

        Args:
            keep_block_b: If True, keeps the compressed block (block_b); otherwise keeps the original block (block_a)
            criteria: List of criteria to filter modules (by name or block_id). No criteria = all
            verbose: If True, prints information about the cancellation process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.cancel_vcon(keep_block_b=keep_block_b, verbose=verbose)

    def set_vcon_beta(self, beta: float, criteria=None, verbose=False):
        """
        Sets the beta value for filtered VCON-initialized blocks

        Args:
            beta: The beta value to set (0 <= beta <= 1)
            criteria: List of criteria to filter modules (by name or block_id). No criteria = all
            verbose: If True, prints information about the beta setting process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            if scheme.vcon_initialized:
                scheme.set_vcon_beta(beta, verbose=verbose)

    def freeze_uncompressed_vcon(self, criteria=None, verbose=False):
        """
        Freezes uncompressed blocks in filtered VCON-initialized modules

        Args:
            criteria: List of criteria to filter modules (by name or block_id). No criteria = all
            verbose: If True, prints information about the freezing process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            if scheme.vcon_initialized:
                scheme.freeze_uncompressed_vcon(verbose=verbose)

    def set_calibration_data(self, calibration_data):
        """
        Stores a calibration dataset/dataloader.

        Args:
            calibration_data: A torch.utils.data.DataLoader that yields
                             framework-compatible batches (mapping kwargs,
                             (data, label), positional tuples/lists, or single
                             input objects).
        """
        if not isinstance(calibration_data, DataLoader):
            raise TypeError(
                "calibration_data must be a torch.utils.data.DataLoader. "
                f"Got {type(calibration_data)}."
            )
        self.calibration_data = calibration_data

    def set_calibration_loss(self, loss_fn: Callable[[Any, Any], torch.Tensor]):
        """
        Stores a user-defined loss function used during calibration backward passes.

        The callable is invoked as `loss_fn(model_output, target)` for each batch.
        Calibration does not infer or extract loss from model outputs.
        """
        if not callable(loss_fn):
            raise TypeError(f"calibration loss function must be callable. Got {type(loss_fn)}.")
        self.calibration_loss_fn = loss_fn

    def set_calibration_mode(
        self,
        mode: str = "standard",
    ):
        """
        Configure calibration target scheduling.

        Args:
            mode:
                - "standard": single-stage calibration over selected schemes.
                - "cascade": multi-stage calibration using indexing-provided
                  calibration_groups.
        """
        if not isinstance(mode, str):
            raise TypeError(f"calibration mode must be a string. Got {type(mode)}.")

        mode = mode.lower()
        if mode not in ["standard", "cascade"]:
            raise ValueError(
                f"Unsupported calibration mode '{mode}'. Supported modes are: ['standard', 'cascade']."
            )

        self.calibration_mode = mode
        self.calibration_groups = self._get_calibration_groups_from_indexing()

    def run_calibration(
        self,
        shifted_model: Optional[torch.nn.Module] = None,
        criteria=None,
        max_batches: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        offload_to_cpu: bool = True,
        verbose: bool = False,
        show_progress: bool = True,
        summary_dump_dir: Optional[str] = None,
        summary_dump_names: Optional[tuple] = None,
    ):
        """
        Run a single calibration pass to collect shared activation statistics.
        
        Args:
            shifted_model: Optional second model used by shifted/cross summaries.
            criteria: Filter criteria for target schemes.
            max_batches: Maximum calibration batches to process.
            device: Device to move batches to.
            offload_to_cpu: If True, accumulate calibration tensors on CPU.
            verbose: If True, print progress.
            show_progress: If True, show calibration batch progress.
            Note: For backward-based calibration summaries, set a loss callback
                  first with set_calibration_loss(...).
        
        Returns:
            Number of schemes calibrated.
        """
        target_stages = self.collect_calibration_target_stages(criteria=criteria)

        return run_compression_calibration(
            model=self.model,
            calibration_data=self.calibration_data,
            target_stages=target_stages,
            shifted_model=shifted_model,
            loss_fn=self.calibration_loss_fn,
            max_batches=max_batches,
            device=device,
            offload_to_cpu=offload_to_cpu,
            verbose=verbose,
            show_progress=show_progress,
            summary_dump_dir=summary_dump_dir,
            summary_dump_names=summary_dump_names,
        )

    def clear_calibration(self, criteria=None):
        """
        Clear stored calibration stats (optionally filtered by criteria).
        """
        schemes = list(self.iter_filtered(criteria=criteria))
        for scheme in schemes:
            # Keep the same dictionary object and clear all summary keys generically.
            scheme.calibration_data.clear()
        return len(schemes)

    def apply(
        self,
        hard=False,
        shifted_model: Optional[torch.nn.Module] = None,
        criteria=None,
        verbose=False,
        max_batches: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        offload_to_cpu: bool = True,
        show_progress: bool = True,
        summary_dump_dir: Optional[str] = None,
        summary_dump_names: Optional[tuple] = None,
    ):
        """
        Applies filtered compression schemes to their respective modules in the model.

        Args:
            hard: If True, applies hard compression (non-reversible); if False, applies soft compression (reversible)
            shifted_model: Optional second model used by shifted/cross summaries.
            criteria: List of criteria to filter modules (by name or block_id). No criteria = all
            verbose: If True, prints information about the application process
            max_batches: Optional cap on calibration batches.
            device: Device where calibration batches should be moved before forward.
            offload_to_cpu: If True, calibration tensors are accumulated on CPU.
            show_progress: If True, show calibration batch progress.
            Note: If selected compressors require backward-based calibration,
                  set both calibration data and calibration loss first.
        """
        if self.calibration_mode == "cascade":
            apply_cascade(
                manager=self,
                hard=hard,
                criteria=criteria,
                verbose=verbose,
                max_batches=max_batches,
                device=device,
                offload_to_cpu=offload_to_cpu,
                show_progress=show_progress,
                summary_dump_dir=summary_dump_dir,
                summary_dump_names=summary_dump_names,
            )
            return

        calibration_targets = self._collect_calibration_targets(criteria=criteria)

        if calibration_targets:
            if self.calibration_data is None:
                missing_stats = [scheme.path for scheme, _ in calibration_targets]
                raise ValueError(
                    "Calibration-required compression needs calibration data before apply. "
                    "Call manager.set_calibration_data(...) before manager.apply(...) "
                    f"for: {missing_stats}"
                )
            self.run_calibration(
                shifted_model=shifted_model,
                criteria=criteria,
                max_batches=max_batches,
                device=device,
                offload_to_cpu=offload_to_cpu,
                verbose=verbose,
                show_progress=show_progress,
                summary_dump_dir=summary_dump_dir,
                summary_dump_names=summary_dump_names,
            )

        for scheme in self.iter_filtered(criteria=criteria):
            scheme.apply(hard=hard, verbose=verbose)

    def _collect_calibration_targets(self, criteria=None):
        stages = self.collect_calibration_target_stages(criteria=criteria)
        merged = {}
        order = []
        for stage in stages:
            for scheme, summary_names in stage:
                if scheme not in merged:
                    merged[scheme] = []
                    order.append(scheme)
                for summary_name in summary_names:
                    if summary_name not in merged[scheme]:
                        merged[scheme].append(summary_name)

        return [(scheme, tuple(merged[scheme])) for scheme in order]

    def _collect_calibration_targets_for_schemes(self, schemes):
        targets = []
        for scheme in schemes:
            required_summaries = []

            for compressor in scheme.compressors.values():
                if not compressor._to_compress():
                    continue

                compressor.set_calibration_store(scheme.calibration_data)

                needs_calibration_fn = getattr(compressor, "needs_calibration", None)
                if not callable(needs_calibration_fn):
                    continue
                needs_calibration = needs_calibration_fn()
                if not needs_calibration:
                    continue

                summary_names = needs_calibration
                if summary_names is True:
                    raise ValueError(
                        f"Compressor {type(compressor).__name__} needs calibration but did not provide summary names."
                    )

                if isinstance(summary_names, str):
                    summary_names = (summary_names,)
                elif not isinstance(summary_names, (tuple, list, set)):
                    raise TypeError(
                        "needs_calibration must return False/None or a string/sequence of summary names."
                    )

                compressor_summary_names = []
                for summary_name in summary_names:
                    if summary_name not in compressor_summary_names:
                        compressor_summary_names.append(summary_name)

                if len(compressor_summary_names) == 0:
                    raise ValueError(
                        f"Compressor {type(compressor).__name__} needs calibration but returned no summaries."
                    )

                for summary_name in compressor_summary_names:
                    if summary_name not in required_summaries and summary_name not in scheme.calibration_data:
                        required_summaries.append(summary_name)

            if len(required_summaries) > 0:
                targets.append((scheme, tuple(required_summaries)))

        return targets

    def collect_calibration_target_stages(self, criteria=None):
        """
        Return a list of calibration stages.

        Each stage is a list of tuples: (scheme, summary_names).
        """
        selected_schemes = list(self.iter_filtered(criteria=criteria))
        if len(selected_schemes) == 0:
            return []

        if self.calibration_mode == "standard":
            stage = self._collect_calibration_targets_for_schemes(selected_schemes)
            return [stage] if len(stage) > 0 else []

        if len(self.calibration_groups) == 0:
            stage = self._collect_calibration_targets_for_schemes(selected_schemes)
            return [stage] if len(stage) > 0 else []

        selected_set = set(selected_schemes)

        grouped_scheme_stages = []
        grouped_set = set()
        for group in self.calibration_groups:
            group_schemes = []
            seen_in_group = set()
            for scheme in self.iter_filtered(criteria=group):
                if scheme not in selected_set:
                    continue
                if scheme in seen_in_group:
                    continue
                seen_in_group.add(scheme)
                group_schemes.append(scheme)
                grouped_set.add(scheme)

            if len(group_schemes) > 0:
                grouped_scheme_stages.append(group_schemes)

        ungrouped = [scheme for scheme in selected_schemes if scheme not in grouped_set]

        stages = []
        for group_schemes in grouped_scheme_stages:
            stage = self._collect_calibration_targets_for_schemes(group_schemes)
            if len(stage) > 0:
                stages.append(stage)

        for scheme in ungrouped:
            stage = self._collect_calibration_targets_for_schemes([scheme])
            if len(stage) > 0:
                stages.append(stage)

        return stages

    def save_state(self, path: str) -> None:
        """Persist per-scheme calibration data to *path* (torch.save format).

        Saves the computed activation statistics (min/max ranges, covariances,
        etc.) for every scheme so they can be restored later without re-running
        the calibration forward passes.  Call this BEFORE prepare_for_save().

        Typical save workflow::

            manager.save_state("checkpoint/compression_state.pt")
            manager.prepare_for_save()
            model.save_pretrained("checkpoint")
        """
        state = {scheme.path: dict(scheme.calibration_data) for scheme in self}
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Restore per-scheme calibration data previously saved with save_state().

        After this call, manager.apply() will find the required summaries
        already populated in scheme.calibration_data and skip the calibration
        forward pass entirely — no DataLoader is needed.

        Typical reload workflow::

            model = Qwen2ForCausalLMCompress.from_pretrained("checkpoint")
            manager = Qwen2CompressionSchemesManager(model)
            manager.load_state("checkpoint/compression_state.pt")
            manager.apply(hard=True, criteria=["mlp"])
        """
        state = torch.load(path, weights_only=False)
        for scheme in self:
            saved = state.get(scheme.path)
            if saved:
                scheme.calibration_data.update(saved)

    def prepare_for_save(self, criteria=None) -> None:
        """Strip runtime quantization artifacts from module state dicts.

        Removes activation-quant hooks, registered buffers (_act_quant_scale,
        _act_out_quant_scale, _torchao_scale, …) and dequantizes any torchao
        AffineQuantizedTensor weights to plain float so that save_pretrained
        can serialize them.  The compression config is preserved in model.config
        and round-trips through save_pretrained/from_pretrained automatically.

        Unlike restore(), this works after hard (non-reversible) quantization
        and does NOT reset the compressor config.

        Recommended export workflow::

            manager.apply(hard=True)                              # finalize compression
            manager.save_state("checkpoint/compression_state.pt") # persist calibration
            manager.prepare_for_save()                            # strip runtime artifacts
            model.save_pretrained("checkpoint")

        Note on pruning: ``weight_mask`` buffers are kept in the state dict by
        default so they round-trip through save/load. Call
        ``manager.remove_masks()`` before saving if they are not needed.
        """
        from ..compression.quantization import Quantizer
        import torch.nn as nn

        soft_only = [
            scheme.path for scheme in self.iter_filtered(criteria=criteria)
            if scheme.soft_applied and not scheme.hard_applied
        ]
        if soft_only:
            logger.warning(
                "Saving with soft compression (not hard) on: %s. "
                "For quantization, call manager.apply(hard=True) first to use "
                "torchao acceleration. For pruning, hard and soft are currently "
                "equivalent (in-place zeroing).",
                soft_only,
            )

        for scheme in self.iter_filtered(criteria=criteria):
            module = scheme.get_compression_module()

            # Dequantize torchao AffineQuantizedTensor weights to plain float.
            # save_pretrained cannot serialize subclassed tensors.
            if isinstance(module, nn.Linear):
                w = module.weight
                if type(w) is not torch.Tensor and hasattr(w, 'dequantize'):
                    module.weight = nn.Parameter(w.dequantize(), requires_grad=False)

            for compressor in scheme.compressors.values():
                if isinstance(compressor, Quantizer):
                    compressor.strip_runtime_state(module)

    def restore(self, topology=False, criteria=None, verbose=False):
        """
        Restores filtered modules to their original state by removing pruning and LRD.

        Args:
            topology: If True, restores only the original topology (e.g., original weight shapes); if False, restores the original weights and parameters as well.
            criteria: List of criteria to filter modules (by name or block_id). No criteria = all
            verbose: If True, prints information about the restoration process
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.restore(topology=topology, verbose=verbose)

    def reapply_masks(self, criteria=None):
        """Re-apply pruning masks to all filtered schemes after an optimizer step.

        Call this inside the fine-tuning loop immediately after
        ``optimizer.step()`` to prevent mask drift when training with soft
        pruning (STE style). Pruned weights receive non-zero gradients during
        the backward pass and drift away from zero; this method zeroes them
        back using the stored ``weight_mask`` buffers.

        Example::

            for batch in dataloader:
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                manager.reapply_masks()

        Args:
            criteria: Optional filter (same format as ``apply()``).
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.reapply_masks()

    def remove_masks(self, criteria=None):
        """Remove ``weight_mask`` buffers from all filtered scheme modules.

        Use before recomputing compression with different settings, or before
        saving a deployment model where the mask buffers are not needed.

        Args:
            criteria: Optional filter (same format as ``apply()``).
        """
        for scheme in self.iter_filtered(criteria=criteria):
            scheme.remove_masks()

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
            path_list = flatten_index_paths(block_indexing['path_list'])
            path_template = block_indexing['path_template']
            config_attr = block_indexing['config_attr'] # added by babisant88: it's done twice.

            # Get the specific config for this block type
            if config_attr == '':
                block_specific_config = self.config
            else:
                block_specific_config = getattr(self.config, config_attr, None)
            assert block_specific_config is not None, f"Config attribute '{config_attr}' not found in the model configuration for block '{block_name}'. Please check the indexing configuration."

            # Get blocks number
            num_blocks = getattr(block_specific_config, num_blocks_attr, None)
            assert num_blocks is not None, f"Number of blocks attribute '{num_blocks_attr}' not found in the model configuration for block '{block_name}'. Please check the indexing configuration."

            tmp_dict = {}
            for i in range(num_blocks):
                for path in path_list:
                    # Create CompressionScheme instance
                    full_path = path_template.format(block_index=i, path=path)

                    # Get pruning ratio and LRD rank from config
                    compression_config = getattr(block_specific_config, 'compression_config', {})
                    compression_config = compression_config.setdefault(full_path, {})
                    tmp_dict[full_path] = CompressionScheme(
                        name=path,
                        block_id=i,
                        path=full_path,
                        compression_config=compression_config,
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
                - None: Matches all schemes
                - list: A list of criteria, where all of them must be met (AND logic within the list)
        """
        if criteria is None:
            criteria = ["all"]
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
                
    def __len__(self):
        """
        Returns the total number of CompressionScheme objects managed.
        """
        return sum(len(block_dicts) for block_dicts in self.schemes.values())

    def __repr__(self):
        """
        Returns a string representation of the CompressionSchemesManager, including the number of schemes and their paths.
        """
        string = f"CompressionSchemesManager with {len(self)} schemes:\n"
        for scheme in self:
            string += scheme.__repr__() + "\n"
        # Remove the last newline character for cleaner formatting
        string = string.rstrip("\n")
        # add intendation for better readability
        string = string.replace("\n", "\n  ")
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
