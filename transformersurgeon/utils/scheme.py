"""
compression.py

Provides the CompressionScheme class for managing structured pruning and low-rank decomposition of transformer blocks.
"""
import copy
import inspect
import torch
from typing import Union
from ..blocks import LinearCompressed
from ..blocks import VCONBlock
from ..compression import (
    COMPRESSOR_DICT,
    COMPRESSION_REGISTRY,
) 
from .utils import get_submodule

# PROBLEM: when pruning a layer, the next layer should also be adjusted accordingly
# but this is not easy for skip connections, e.g., residual connections in transformers
# Maybe it is possible to do so by storing the masks of the previouse layers and use torch.scatter (that might be supported in executorch) to insert zeros where needed

class CompressionScheme:
    """
    Represents a compression scheme for a transformer block.

    Args:
        name (str): Name of the layer inside the block.
        block_id (int): Block identifier.
        path (str): Path to the block in the model.
        pruning_ratio (float): Ratio for structured pruning.
        pruning_mode (str): Mode for pruning ('structured' or 'unstructured').
        lrd_rank (Union[int, str]): Rank for low-rank decomposition. Use "full" for no decomposition.
        model (torch.nn.Module, optional): Reference to the model.

    Attributes:
        hard_applied (bool): Flags if compression is non-reversible.
        soft_applied (bool): Flags if compression has been performed.
        vcon_initialized (bool): Flags if VCONBlock has been initialized.

    Methods:
        get_module: Returns the module at the specified path.
        set_module: Sets a new module at the specified path.
        _is_to_compress: Checks if compression should be applied.
        _module_copy: Returns a copy of the module.
        init_vcon: Initializes VCON block.
        cancel_vcon: Cancels VCON block.
        set_vcon_beta: Sets beta for VCON block.
        freeze_uncompressed_vcon: Freezes uncompressed VCON block.
        apply: Applies compression (pruning and/or LRD).
        restore: Restores original module.
        _unstructured_maks: Generates a mask for unstructured pruning.
        _structured_mask: Generates a mask for structured pruning.
        _low_rank_decomposition: Performs low-rank decomposition.
    """

    def __init__(
            self,
            name,
            block_id,
            path,
            # output_paths,
            compression_config=None,
            model=None,
            ):
        self.name = name
        self.block_id = block_id
        self.path = path
        self.model = model
        self.hard_applied = False # this flags the compression as non-reversible when True
        self.soft_applied = False # this flags the compression as already-peformed: do not overwrite/reinitialize with hard application
        self.vcon_initialized = False # this flags if the VCONBlock has been initialized

        # Generate compressor list based on the provided configuration
        self.compressors = {}
        # Extract path-specific compression configuration, if provided
        if compression_config is not None:
            for comp_name, comp_config in compression_config.items():
                # Instantiate correct compressor with the provided configuration
                compressor = COMPRESSOR_DICT.get(comp_name, None)
                if compressor is None:
                    raise ValueError(f"Unsupported compression type '{comp_name}' in configuration.")
                self.compressors[comp_name] = compressor(**comp_config)

        # Check if the module exists in the model
        if self.model is not None:
            self.get_module()

        # Backup objects
        self._weight_original = None # This will store the original weights for restoration if needed

    def get_module(self):
        """
        Returns the module connected to the compression scheme.
        """
        # Check if model has been provided
        if not hasattr(self, 'model'):
            raise ValueError("Model is not set. Please set the model before getting the module.")

        return get_submodule(self.model, self.path)

    def set(self, compression_name, compression_property, value, verbose=False):
        """
        Sets a specific property of a compression type in the compression scheme.

        Args:
            compression_name (str): The name of the compression type (e.g., 'pruning', 'lrd').
            compression_property (str): The name of the property to set (e.g., 'ratio' for pruning).
            value: The new value to set for the specified property.
        """
        if verbose:
            print(f"Setting {compression_property} of {compression_name} to {value} for module {self.path}.")

        # Check validity of the property value using COMPRESSION_REGISTRY
        if compression_name not in COMPRESSION_REGISTRY:
            raise ValueError(f"Unsupported compression type '{compression_name}' in configuration.")
        if compression_property not in COMPRESSION_REGISTRY[compression_name]:
            raise ValueError(f"Unsupported property '{compression_property}' for compression type '{compression_name}' in configuration.")
        validator = COMPRESSION_REGISTRY[compression_name][compression_property]['validator']
        if validator is not None:
            validator(value) # This will raise an error if the value is invalid

        if self.soft_applied or self.hard_applied:
            raise RuntimeError(f"Cannot set {compression_property} for {compression_name} because compression has already been applied.")

        # Generate compressor if necessary
        if compression_name not in self.compressors:
            compressor_class = COMPRESSOR_DICT.get(compression_name, None)
            if compressor_class is None:
                raise ValueError(f"Unsupported compression type '{compression_name}' in configuration.")
            self.compressors[compression_name] = compressor_class(self.get_module)

        # Set the specified property of the compressor
        compressor = self.compressors[compression_name]
        if hasattr(compressor, compression_property):
            setattr(compressor, compression_property, value)
        else:
            raise ValueError(f"Compressor '{compression_name}' does not have property '{compression_property}'.")
    
    def set_module(self, new_module):
        """
        Sets a new module at the position specified by the compression scheme.

        Args:
            new_module (torch.nn.Module): The new module to set.
        """
        # Check if model has been provided
        if not hasattr(self, 'model'):
            raise ValueError("Model is not set. Please set the model before setting the module.")
        
        # Check if init_vcon has been called
        if self.vcon_initialized:
            raise ValueError("Cannot set module when VCONBlock is initialized. Please cancel VCON first.")

        split_path = self.path.split('.')
        # Traverse the model iteratively to find the parent module
        tmp_module = self.model
        for path_piece in split_path[:-1]:
            tmp_module = getattr(tmp_module, path_piece, None)

            if tmp_module is None:
                raise ValueError(f"Module at path '{self.path}' not found in the model.")
        
        # Set the new module at the specified path
        setattr(tmp_module, split_path[-1], new_module)
    
    def _is_to_compress(self):
        """
        Checks if compression has been set for the module, or if the module is compatible with compression.
        Compatible modules:
            - LinearCompressed

        Returns:
            bool: True if compression should be applied, False otherwise.
        """
        # Navigate through the compressors to check if any compression is configured for this module
        to_compress = False
        for comp_name, compressor in self.compressors.items():
            to_compress = to_compress or compressor._to_compress()

        return to_compress and self._is_compatible()
    
    def _is_compatible(self):
        """
        Checks if the module is compatible with compression.
        Compatible modules:
            - LinearCompressed

        Returns:
            bool: True if the module is compatible, False otherwise.
        """
        compatible = False
        compatible = compatible or self.vcon_initialized # if VCON is initialized, compatible by definition
        compatible = compatible or (type(self.get_module()) is LinearCompressed)
        
        return compatible
    
    def _module_copy(self, module):
        """
        Returns a hard copy of the module.

        Args:
            module (torch.nn.Module): Module to copy.

        Returns:
            torch.nn.Module: Copied module.
        """
        # Get the __init__ signature
        sig = inspect.signature(type(module).__init__)
        # Build kwargs from module attributes
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if hasattr(module, name):
                # check on bias, must be bool
                if name == 'bias':
                    kwargs["bias"] = True if module.bias is not None else False
                else:
                    kwargs[name] = getattr(module, name)
        # Create a new instance of the same class
        module_copy = type(module)(**kwargs)
        for name, param in module.named_parameters(recurse=False):
            getattr(module_copy, name).data = param.data.clone()
        for name, buf in module.named_buffers(recurse=False):
            setattr(module_copy, name, buf.clone())
        # Copy other attributes except parameters, buffers, and methods
        for attr, value in module.__dict__.items():
            if (
                not attr.startswith('_')
                and not isinstance(value, torch.nn.Parameter)
                and not isinstance(value, torch.Tensor)
                and not callable(value)
            ):
                try:
                    setattr(module_copy, attr, copy.deepcopy(value))
                except Exception:
                    pass
        return module_copy

    def init_vcon(self, verbose=False):
        """
        Initializes a VCONBlock for the current module by duplicating it.
        This method checks if compression is required and has not yet been applied.
        If so, it creates a copy of the current module and instantiates a VCONBlock
        with the original and copied modules as its components. The module is then
        replaced with the VCONBlock. If `verbose` is True, a message indicating
        successful initialization is printed. The method sets the `vcon_initialized`
        flag to True upon completion.

        Raises:
            ValueError: If compression has already been applied (either soft or hard).
            
        Args:
            verbose (bool, optional): If True, prints a message upon successful initialization.
        """   
        # check if the model is compatible with compression
        if not self._is_compatible():
            return # nothing to do
        
        # check if compression has already been applied
        if self.soft_applied or self.hard_applied:
            raise ValueError("Cannot initialize VCONBlock after compression has been applied.")

        module = self.get_module()
        module_copy = self._module_copy(module)
        vcon_block = VCONBlock(block_a=module, block_b=module_copy)
        self.set_module(vcon_block)
        if verbose:
            print(f"Initialized VCONBlock at {self.path}")
        self.vcon_initialized = True

    def cancel_vcon(self, keep_block_b=True, verbose=False):
        """
        Cancels and removes the VCONBlock from the module, retaining either `block_a` or `block_b`.
        This method replaces the current module containing a VCONBlock with either its `block_a` or `block_b` submodule,
        depending on the value of `keep_block_b`. It also updates the internal state to reflect that the VCONBlock
        is no longer initialized.

        Args:
            keep_block_b (bool, optional): If True, retains `block_b` after cancellation; otherwise, retains `block_a`.
                Defaults to True.
            verbose (bool, optional): If True, prints a message indicating which block was kept and the path of the module.
                Defaults to False.

        Raises:
            ValueError: If the VCONBlock is not initialized and cancellation is attempted.

        Side Effects:
            - Replaces the current module with the selected block.
            - Updates `self.vcon_initialized` to False.
            - Optionally prints a message if `verbose` is True.
        """
 
        if not self.vcon_initialized:
            # Nothing to do
            return
        
        self.vcon_initialized = False

        module = self.get_module()
        if keep_block_b:
            module = module.block_b
        else:
            module = module.block_a
        
        self.set_module(module)

        if verbose:
            kept = "block_b" if keep_block_b else "block_a"
            print(f"Cancelled VCONBlock at {self.path}, kept {kept}.")  

    def set_vcon_beta(self, beta: float, verbose=False):
        """
        Set the beta value for the VCONBlock module.
        This method updates the beta parameter of the VCONBlock, which controls the contribution
        of each block in the model. The VCONBlock must be initialized before calling this method.

        Args:
            beta (float): The new beta value to set for the VCONBlock.
            verbose (bool, optional): If True, prints a message indicating the beta value has been set. Defaults to False.

        Raises:
            ValueError: If the VCONBlock is not initialized.

        Example:
            >>> obj.set_vcon_beta(0.5, verbose=True)
            Set VCONBlock beta to 0.5 at <path>.
        """

        if not self.vcon_initialized:
            raise ValueError("VCONBlock is not initialized, cannot set beta.")
        
        module = self.get_module() # get the VCONBlock
        module.set_beta(beta)
        if verbose:
            print(f"Set VCONBlock beta to {beta} at {self.path}.")

    def freeze_uncompressed_vcon(self, verbose=False):
        """
        Freezes the parameters of the `block_a` submodule within a VCONBlock to prevent them from being updated during training.
        This method sets `requires_grad` to `False` for all parameters in `block_a`, effectively freezing them.
        It is typically used when you want to exclude the uncompressed part of the VCONBlock from optimization.

        Args:
            verbose (bool, optional): If True, prints a message indicating that the parameters have been frozen. Default is False.

        Raises:
            ValueError: If the VCONBlock has not been initialized (`self.vcon_initialized` is False).
        """
        if not self.vcon_initialized:
            raise ValueError("VCONBlock is not initialized, cannot freeze uncompressed block.")
        
        module = self.get_module() # get the VCONBlock
        for param in module.block_a.parameters():
            param.requires_grad = False
        if verbose:
            print(f"Froze parameters of block_a in VCONBlock at {self.path}.")
        
    def apply(self, hard=False, verbose=False):
        """
        Applies magnitude-based structured pruning and Low Rank Decomposition (LRD) to the target module.

        VCONBlock Handling:
            - If the module is wrapped in a VCONBlock, compression is applied only to the second block (`block_b`).

        Parameters:
            hard (bool): If True, applies irreversible (hard) changes to the module. If False, applies reversible (soft) changes.
            verbose (bool): If True, prints detailed information about the application process.

        Returns:
            None
        """       

        if not self._is_to_compress():
            return # nothing to apply

        if verbose:
            print(f"Applying ({"HARD (non-reversible)" if hard else "soft"}) compression scheme:\n{self}.")
            if self.soft_applied:
                if hard:
                    print(f"  ! Compression already hard applied, making the changes permanent (HARD mode)")
                else:
                    print(f"  ! Compression already soft applied, skipping re-application.")

        if self.hard_applied or (self.soft_applied and not hard):
            return # Nothing to do
        
        # Get the module to compress
        module = self.get_module()
        if self.vcon_initialized:
            module = module.block_b # apply to block_b only
        
        if not hard and not self.soft_applied: # Backup original weights if not already done
            self._weight_original = module.weight.data.clone() # Store original weights for restoration if needed
        else: # Make changes permanent
            self._weight_original = None

        # Perform compression for each compressor in the scheme
        for cname, compressor in self.compressors.items():
            if verbose:
                print(f"Applying compression '{cname}' for module {self.path}")
            compressor.apply(module=module, hard=hard, soft_applied=self.soft_applied)
                    
        # Flag as applied
        self.soft_applied = True
        self.hard_applied = hard

    def restore(self, topology=False, verbose=False):
        """
        Restores the original topology of the model.
        If the module is wrapped in a VCONBlock, restoration is applied only to the second block (`block_b`).
        The function reverses any soft-applied changes, such as pruning masks and LRD, but raises an error if hard-applied (non-reversible) changes have been made.

        Args:
            topology (bool, optional): If True, restores only the original topology of the model, while compression is functionally maintained. Defaults to False.
            verbose (bool, optional): If True, prints information about the restoration process. Defaults to False.

        Raises:
            ValueError: If the module has been hard applied and cannot be restored.
        """
        if not self._is_to_compress() or not self.soft_applied:
            return # nothing to restore

        if self.hard_applied:
            raise ValueError("Cannot restore a module that has been hard applied (non-reversible changes).")
        
        module = self.get_module()
        if self.vcon_initialized:
            module = module.block_b # apply to block_b only
        
        # For each compression type, if it has been applied, call the corresponding restore function
        for comp_name, compressor in self.compressors.items():
            if verbose:
                print(f"Restoring topology for compression '{comp_name}' for module {self.path}.")
            compressor.restore(module) # Call the restore function of the compressor to reverse the changes

        if not topology: # Restore original weights before compression
            # Raise error if original weights are missing
            if self._weight_original is None:
                raise ValueError("Original weights not found for restoration. Cannot restore the module.")
            if verbose:
                print(f"Restoring parameters for module {self.path}.")
            with torch.no_grad():
                module.weight.copy_(self._weight_original)

        # Clean up the backup of original weights
        self._weight_original = None
        # Reset the soft application flag
        self.soft_applied = False 

    def __repr__(self):
        string = f"CompressionScheme(name={self.name}, block_id={self.block_id}, path={self.path}, applied=(soft={self.soft_applied}, hard={self.hard_applied}))\n"
        for comp_name, compressor in self.compressors.items():
            if compressor._to_compress(): # Only include compressors that are set to be applied
                string += comp_name + ": " + compressor.__repr__() + "\n"
        # Remove the last newline character for cleaner formatting
        string = string.rstrip("\n")
        # Add indentation for better readability
        string = string.replace("\n", "\n  ")
        return string


__all__ = ["CompressionScheme"]
