# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
manager.py

Provides the CompressionSchemesManager class for managing multiple compression schemes in transformer models.
"""

from collections.abc import Mapping
import torch
from transformers import PretrainedConfig
from typing import Any, Callable, Dict, List, Optional, Union
import warnings
from .scheme import CompressionScheme

class CompressionSchemesManager:
    """
    Class for managing multiple compression schemes across different modules of a transformer model. It allows setting properties, initializing VCON blocks, applying compression, and restoring the original model state based on flexible filtering criteria.
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
        try:
            self.config = model.config
            assert isinstance(self.config, PretrainedConfig), "Model config is not an instance of PretrainedConfig. Please provide a model with a valid Hugging Face configuration."
        except AttributeError:
            raise ValueError("The provided model does not have a 'config' attribute. Please provide a model with a valid configuration.")
        self.indexing = indexing
        self.schemes = self._generate_schemes()

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
        Stores a calibration dataset/dataloader on the manager.

        The stored data can later be consumed by calibrate_lrd().
        """
        self.calibration_data = calibration_data

    def calibrate_lrd(
        self,
        calibration_data=None,
        criteria=None,
        max_batches: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        forward_fn: Optional[Callable] = None,
        offload_to_cpu: bool = True,
        verbose: bool = False,
    ):
        """
        Runs calibration data once through the model and accumulates activation
        covariance for all filtered LRD compressors using method "svd_llm_v2".

        Args:
            calibration_data: Iterable of model inputs. If None, uses data
                previously provided via set_calibration_data().
            criteria: Optional manager criteria to filter target modules.
            max_batches: Optional cap on the number of calibration batches.
            device: Device where model inputs should be moved before forward.
                Defaults to the model's first parameter/buffer device.
            forward_fn: Optional callable with signature
                forward_fn(model, batch). Use this when the batch structure
                needs custom model invocation.
            offload_to_cpu: If True, activation covariance is accumulated on CPU.
            verbose: If True, prints calibration progress information.

        Returns:
            int: Number of LRD compressors calibrated.
        """
        if calibration_data is None:
            calibration_data = self.calibration_data
        else:
            self.calibration_data = calibration_data

        if calibration_data is None:
            raise ValueError("Calibration data is required to calibrate LRD with method 'svd_llm_v2'.")

        targets = self._lrd_svd_llm_v2_targets(criteria=criteria)
        if len(targets) == 0:
            if verbose:
                print("No LRD compressors with method 'svd_llm_v2' found for calibration.")
            return 0

        for _, compressor in targets:
            compressor.clear_covariance()
            compressor._covariance_sum = None
            compressor._covariance_tokens = 0

        handles = []
        for scheme, compressor in targets:
            module = scheme.get_module()
            handles.append(module.register_forward_hook(self._make_lrd_calibration_hook(compressor, offload_to_cpu)))

        if device is None:
            device = self._infer_model_device()
        device = torch.device(device)

        was_training = self.model.training
        num_batches = 0
        try:
            self.model.eval()
            with torch.no_grad():
                for batch_id, batch in enumerate(self._calibration_iter(calibration_data)):
                    if max_batches is not None and batch_id >= max_batches:
                        break
                    batch = self._move_to_device(batch, device)
                    self._run_calibration_batch(batch, forward_fn=forward_fn)
                    num_batches += 1
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                self.model.train()

        if num_batches == 0:
            raise ValueError("Calibration data must contain at least one batch.")

        for scheme, compressor in targets:
            if compressor._covariance_tokens == 0:
                raise RuntimeError(f"No calibration activations were collected for module {scheme.path}.")
            compressor.set_covariance(compressor._covariance_sum / compressor._covariance_tokens)
            del compressor._covariance_sum
            del compressor._covariance_tokens

        if verbose:
            print(f"Calibrated {len(targets)} LRD compressors over {num_batches} batches.")

        return len(targets)

    def clear_lrd_calibration(self, criteria=None):
        """
        Clears stored LRD activation covariance for filtered modules.
        """
        for _, compressor in self._lrd_svd_llm_v2_targets(criteria=criteria):
            compressor.clear_covariance()

    def apply(self, hard=False, criteria=None, verbose=False, calibration_data=None, calibration_kwargs=None):
        """
        Applies filtered compression schemes to their respective modules in the model.

        Args:
            hard: If True, applies hard compression (non-reversible); if False, applies soft compression (reversible)
            criteria: List of criteria to filter modules (by name or block_id). No criteria = all
            verbose: If True, prints information about the application process
            calibration_data: Optional calibration dataset/dataloader. When
                provided, LRD compressors using method "svd_llm_v2" are
                calibrated before compression is applied.
            calibration_kwargs: Optional dict forwarded to calibrate_lrd().
        """
        if calibration_data is not None:
            calibration_kwargs = calibration_kwargs or {}
            self.calibrate_lrd(
                calibration_data=calibration_data,
                criteria=criteria,
                verbose=verbose,
                **calibration_kwargs,
            )

        for scheme in self.iter_filtered(criteria=criteria):
            scheme.apply(hard=hard, verbose=verbose)

    def _lrd_svd_llm_v2_targets(self, criteria=None):
        targets = []
        for scheme in self.iter_filtered(criteria=criteria):
            compressor = scheme.compressors.get("lrd", None)
            if compressor is None or not compressor._to_compress():
                continue
            method = getattr(compressor, "method", compressor.config.get("method", "svd"))
            if method == "svd_llm_v2":
                targets.append((scheme, compressor))
        return targets

    def _make_lrd_calibration_hook(self, compressor, offload_to_cpu: bool):
        def hook(module, inputs, output):
            if len(inputs) == 0:
                raise ValueError("Cannot collect LRD calibration data from a module without input activations.")
            activation = inputs[0].detach()
            in_features = getattr(module, "in_features", activation.size(-1))
            if activation.size(-1) != in_features:
                raise ValueError(
                    f"Calibration activation last dimension is {activation.size(-1)}, expected {in_features}."
                )
            if activation.dim() == 1:
                activation = activation.reshape(1, -1)
            else:
                activation = activation.reshape(-1, activation.size(-1))
            if offload_to_cpu:
                activation = activation.cpu()
            activation = activation.float()
            covariance_sum = activation.transpose(0, 1) @ activation
            if compressor._covariance_sum is None:
                compressor._covariance_sum = covariance_sum
            else:
                compressor._covariance_sum.add_(covariance_sum)
            compressor._covariance_tokens += activation.size(0)
        return hook

    def _calibration_iter(self, calibration_data):
        if isinstance(calibration_data, torch.Tensor) or isinstance(calibration_data, Mapping):
            return (calibration_data,)
        if isinstance(calibration_data, tuple) and all(isinstance(item, torch.Tensor) for item in calibration_data):
            return (calibration_data,)
        return calibration_data

    def _infer_model_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            pass
        try:
            return next(self.model.buffers()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_to_device(self, value, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: self._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item, device) for item in value]
        if hasattr(value, "to"):
            try:
                return value.to(device)
            except TypeError:
                pass
        return value

    def _run_calibration_batch(self, batch, forward_fn: Optional[Callable] = None):
        if forward_fn is not None:
            return forward_fn(self.model, batch)
        if isinstance(batch, Mapping):
            return self.model(**batch)
        if isinstance(batch, (tuple, list)):
            return self.model(*batch)
        return self.model(batch)

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
            path_list = block_indexing['path_list']
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
