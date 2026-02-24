"""
compression.py

Provides the CompressionScheme class for managing structured pruning and low-rank decomposition of transformer blocks.
"""
import copy
import inspect
import torch
from typing import Union
from ..layers import LinearCompressed
from ..layers import VCONBlock
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
        _unstructured_pruning: Performs unstructured pruning.
        _structured_pruning: Performs structured pruning.
        _low_rank_decomposition: Performs low-rank decomposition.
    """

    def __init__(self,
                 name,
                 block_id,
                 path,
                 # output_paths,
                 pruning_ratio=0.0,
                 pruning_mode='structured',
                 lrd_rank="full",
                 bits=None,
                 model=None,
                 ):
        self.name = name
        self.block_id = block_id
        self.path = path
        self.pruning_ratio = pruning_ratio
        self.pruning_mode = pruning_mode
        # self.output_paths = output_paths # blocks after this layer, input should be pruned accordingly
        self.lrd_rank = lrd_rank
        self.bits=bits
        self.model = model
        self.hard_applied = False # this flags the compression as non-reversible when True
        self.soft_applied = False # this flags the compression as already-peformed: do not overwrite/reinitialize with hard application
        self.vcon_initialized = False # this flags if the VCONBlock has been initialized

        if model is not None:
            # Check if the module exists in the model
            module = self.get_module()
            # Check if the module has already been hard-compressed (low-rank only)
            if hasattr(module, 'lrd_rank') and module.lrd_rank != "full":
                self.hard_applied = True

    def get_module(self):
        """
        Returns the module connected to the compression scheme.
        """
        # Check if model has been provided
        if not hasattr(self, 'model'):
            raise ValueError("Model is not set. Please set the model before getting the module.")

        return get_submodule(self.model, self.path)
    
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
        compression_set = False
        compression_set = compression_set or self.pruning_ratio > 0
        compression_set = compression_set or (self.lrd_rank and self.lrd_rank != "full")
        compression_set = compression_set or (self.bits and self.bits < 32)

        return compression_set and self._is_compatible()
    
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
            setattr(module_copy, name, torch.nn.Parameter(param.data.clone()))
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
            raise ValueError("VCONBlock is not initialized, cannot cancel.")
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

        Pruning:
            - If `pruning_ratio` > 0, performs structured pruning on the module's weights.
            - If `hard` is True, pruned rows are permanently removed from the module's weights and biases.
            - If `hard` is False, a prune mask is set for soft pruning (reversible).
            - Pruning is only applied once unless `hard` is specified after a soft application.

        Low Rank Decomposition (LRD):
            - If `lrd_rank` is set and not "full", applies LRD to the module's weights.
            - If `hard` is False, stores the original weight matrix for possible restoration.
            - If `hard` is True, removes the stored original weight matrix.
            - LRD is only applied once unless `hard` is specified after a soft application.

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
        
        module = self.get_module()
        if self.vcon_initialized:
            module = module.block_b # apply to block_b only
        
        if verbose:
            print(f"Applying ({'hard' if hard else 'soft'}) compression scheme {self} to module {self.path}.")
            if self.soft_applied:
                if hard:
                    print(f"Compression already hard applied, making the changes permanent (HARD mode)")
                else:
                    print(f"Compression already soft applied, skipping re-application.")
            else:
                print(f"Applying {'HARD (non-reversible) ' if hard else ''}compression to module {self.path}.")

        if self.hard_applied: # already hard applied, nothing to do
            return
        
        if not hard: # Backup original weights if not already done
            if not hasattr(module, '_weight_original'):
                module._weight_original = module.weight.data.clone()
        else: # Make changes permanent
            if hasattr(module, '_weight_original'):
                del module._weight_original
        
        # Apply pruning
        if self.pruning_ratio > 0:
            # Prune module
            if not self.soft_applied:
                if self.pruning_mode == 'structured':
                    prune_mask, kept = self._structured_pruning(module.weight.data, norm=2, pruning_ratio=self.pruning_ratio)
                    self.prune_mask = prune_mask.to(torch.int8) # store for later use
                    self.kept = kept # store for later use
                elif self.pruning_mode == 'unstructured':
                    prune_mask = self._unstructured_pruning(module.weight.data, criterion="magnitude", pruning_ratio=self.pruning_ratio)
                else:
                    raise ValueError(f"Unsupported pruning mode '{self.pruning_mode}' for module {self.path}.")
                
                if not hard:
                    # Apply soft pruning by zeroing out the pruned rows/weights
                    if self.pruning_mode == 'structured':
                        module.weight.data = module.weight.data*prune_mask.unsqueeze(1).to(torch.float32)
                    else:
                        module.weight.data = module.weight.data*prune_mask.to(torch.float32)
            
            if hard:
                if self.pruning_mode == 'unstructured':
                    raise ValueError("Hard pruning is not supported for unstructured pruning.")
                
                # Apply hard pruning by removing the pruned rows
                module.weight.data = module.weight.data[self.prune_mask]
                module.bias.data = module.bias.data[self.prune_mask] if module.bias is not None else None
                module.out_features = self.kept
                del self.prune_mask                

        # Apply Low Rank Decomposition (LRD)
        if self.lrd_rank and self.lrd_rank != "full":

            if not self.soft_applied: # Apply LRD if decomposition not yet applied
                # module.weight.data shape: (out_features, in_features)
                US_r, V_r = self._low_rank_decomposition(module.weight.data, self.lrd_rank)
                module.weight = torch.nn.Parameter(US_r) # shape (out_features, lrd_rank)
                module.weight_2 = torch.nn.Parameter(V_r) # shape (lrd_rank, in_features)
                # module.bias.data = module.bias.data*0.0 # TEMP
                # module.bias1 = torch.nn.Parameter(torch.zeros(self.lrd_rank, device=module.weight.device, dtype=module.weight.dtype)) # TEMP
                module.set_lrd_rank(self.lrd_rank)

        # Apply quantization
        if self.bits and self.bits < 32:            
            if not self.soft_applied:
                # Replace weight with quantized version
                W = self._quantization(module.weight.data, self.bits)
                module.weight = torch.nn.Parameter(W)  

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
        if self.hard_applied:
            raise ValueError("Cannot restore a module that has been hard applied (non-reversible changes).")
        
        module = self.get_module()
        if self.vcon_initialized:
            module = module.block_b # apply to block_b only
        
        if topology:
            # Restore original topology while maintaining compression
            if hasattr(module, 'weight_2'):
                if verbose:
                    print(f"Restoring original topology for low-rank decomposed module {self.path}.")
                # Rebuild low-rank weight
                W = module.weight.data @ module.weight_2.data
                module.weight = torch.nn.Parameter(W)
                del module.weight_2
                module.set_lrd_rank("full")
            if hasattr(self, 'prune_mask'):
                if verbose:
                    print(f"Deleting prune mask for module {self.path} while maintaining pruning (topology restoration).")
                del self.prune_mask
            if hasattr(self, 'kept'):
                del self.kept
            if hasattr(module, '_weight_original'):
                del module._weight_original
        else:
            # Restore original weights and topology
            if hasattr(module, '_weight_original'):
                if verbose:
                    print(f"Restoring module {self.path} to its original state.")
                module.weight = torch.nn.Parameter(module._weight_original)
                del module._weight_original
                module.set_lrd_rank("full")

        self.soft_applied = False # Reset the soft application flag

    def __repr__(self):
        return (f"CompressionScheme(name={self.name}, block_id={self.block_id}, "
                f"path={self.path}, pruning_ratio={self.pruning_ratio}, pruning_mode={self.pruning_mode}, "
                f"lrd_rank={self.lrd_rank}, "
                f"module={self.get_module().__class__.__name__})")
    
    def _unstructured_pruning(self, weight, criterion="magnitude", pruning_ratio=0.0) -> torch.Tensor:
        """
        Returns a mask for unstructured pruning based on the specified criterion.
        This function generates a binary mask that can be applied to the weight tensor
        to zero out individual weights (unstructured pruning) based on their magnitude.

        Args:
            weight (torch.Tensor): The weight tensor to be pruned.
            criterion (str): The criterion to use for calculating the importance of weights. Default is "magnitude".
            pruning_ratio (float): The ratio of weights to prune (between 0 and 1). Default is 0.0 (no pruning).

        Returns:
            torch.Tensor: A binary mask with the same shape as the weight tensor.
        """        
        if pruning_ratio >= 1.0: # Prune all weights
            return torch.zeros_like(weight, dtype=torch.bool, device=weight.device)
        
        # Determine the number of weights to prune
        num_weights_to_prune = int(pruning_ratio * weight.numel())
        
        if pruning_ratio <= 0.0 or num_weights_to_prune == 0: # No pruning, return a mask of all ones
            return torch.ones_like(weight, dtype=torch.bool, device=weight.device)
        
        # Calculate importance scores based on the criterion
        if criterion == "magnitude":
            scores = torch.abs(weight)
        else:
            raise ValueError(f"Unsupported criterion '{criterion}' for unstructured pruning.")
        
        # Generate the pruning mask
        threshold = torch.topk(scores.reshape(-1), num_weights_to_prune, largest=False, sorted=False).values.max()
        mask = scores > threshold
        
        return mask.view_as(weight)

    def _structured_pruning(self, weight, norm=2, pruning_ratio=0.0) -> torch.Tensor:
        """
        Returns a mask for structured pruning based on the specified norm.
        This function generates a binary mask that can be applied to the weight tensor
        to zero out entire rows (structured pruning) based on their L2 norm.

        Args:
            weight (torch.Tensor): The weight tensor to be pruned.
            norm (int): The norm to use for calculating the importance of rows. Default is 2 (L2 norm).
            pruning_ratio (float): The ratio of rows to prune (between 0 and 1). Default is 0.0 (no pruning).

        Returns:
            torch.Tensor: A binary mask vector with the same number of elements as the rows of the weight tensor.
            kept (int): Number of rows kept after pruning.
        """
        current_rows = weight.size(0)
        
        if pruning_ratio >= 1.0: # Prune all rows
            return torch.zeros(weight.size(0), dtype=torch.bool, device=weight.device), 0
        
        # Determine the number of rows to prune
        num_rows_to_prune = int(pruning_ratio * current_rows)
        kept = current_rows - num_rows_to_prune
        
        if pruning_ratio <= 0.0 or num_rows_to_prune == 0: # Prune no rows
            return torch.ones(weight.size(0), dtype=torch.bool, device=weight.device), weight.size(0)
        
        # Generate scores
        scores = torch.norm(weight, p=norm, dim=1) # Calculate the norm of each row

        # Generate pruning mask
        threshold = torch.topk(scores, num_rows_to_prune, largest=False, sorted=False).values.max()
        mask = scores > threshold
        
        return mask, kept

    def _low_rank_decomposition(self, weight, rank: int) -> torch.Tensor:
        """
        Performs low-rank decomposition on the given weight matrix using SVD.

        Args:
            weight (torch.Tensor): The weight matrix to be decomposed.
            rank (int): The target rank for the decomposition.

        Returns:
            torch.Tensor: The first matrix of the low-rank decomposition.
            torch.Tensor: The second matrix of the low-rank decomposition.
        """
        if rank >= min(weight.size()):
            # No decomposition possible, launch error
            raise ValueError(f"Rank {rank} must be less than the minimum dimension of the weight matrix ({weight.size(0), weight.size(1)}).")
        
        # Perform SVD
        weight_f32 = weight.float() # Convert to float32 for SVD computation
        U, S, Vh = torch.linalg.svd(weight_f32, full_matrices=False)

        # Keep only the top 'rank' components
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        # Reconstruct the low-rank approximation
        US_r_out = (U_r * S_r.unsqueeze(0)).to(weight.dtype)
        V_r_out = Vh_r.to(weight.dtype)
        return US_r_out.contiguous(), V_r_out.contiguous()
    
    def _quantization(self, weight, qbits) -> torch.Tensor:
        """
        Performs Quntization on the given weight matrix using uniform quantization.

        Args:
            weight (torch.Tensor): The weight matrix to be quantized.
            qbits (int): The number of bits for quantization.

        Returns:
            torch.Tensor: The quantized weight matrix.
        """        
        q_levels = 2 ** qbits
        w_min, w_max = weight.min(), weight.max()
        w_norm = (weight - w_min) / (w_max - w_min + 1e-8)
        w_q = torch.round(w_norm * (q_levels - 1)) / (q_levels - 1)
        w_q = w_q * (w_max - w_min) + w_min

        return w_q

__all__ = ["CompressionScheme"]
