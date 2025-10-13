import copy
import inspect
import torch
import torch.nn.utils.prune as prune
"""
compression.py

Provides the CompressionScheme class for managing structured pruning and low-rank decomposition of transformer blocks.
"""

from ..layers.VCONBlock import VCONBlock

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
        lrd_rank (Union[int, str]): Rank for low-rank decomposition. Use "full" for no decomposition.
        is_qkv_concatenated (bool, optional): Whether QKV layers are concatenated in the model definition.
        model (torch.nn.Module, optional): Reference to the model.

    Methods:
        get_module: Returns the module at the specified path.
        set_module: Sets a new module at the specified path.
        _is_to_compress: Checks if compression should be applied.
        _module_copy: Returns a copy of the module.
        init_vcon: Initializes VCON block.
        cancel_vcon: Cancels VCON block.
        set_vcon_beta: Sets beta for VCON block.
        freeze_uncompressed_vcon: Freezes uncompressed VCON block.
        apply: Applies compression.
        restore: Restores original module.
        _structured_pruning: Performs structured pruning.
        _low_rank_decomposition: Performs low-rank decomposition.
    """

    def __init__(self,
                 name,
                 block_id,
                 path,
                 pruning_ratio, #output_paths, 
                 lrd_rank,
                 is_qkv_concatenated=False,
                 model=None,
                 ):
        self.name = name
        self.block_id = block_id
        self.path = path
        self.pruning_ratio = pruning_ratio
        # self.output_paths = output_paths # blocks after this layer, input should be pruned accordingly
        self.lrd_rank = lrd_rank
        self.is_qkv_concatenated = is_qkv_concatenated
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

        split_path = self.path.split('.')
        # Traverse the model iteratively to find the module
        tmp_module = self.model
        for path_piece in split_path:
            tmp_module = getattr(tmp_module, path_piece, None)

            if tmp_module is None:
                raise ValueError(f"Module at path '{self.path}' not found in the model.")

        return tmp_module
    
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
        Checks if compression has been set for the module.

        Returns:
            bool: True if compression should be applied, False otherwise.
        """
        return (self.pruning_ratio > 0) or (self.lrd_rank and self.lrd_rank != "full")
    
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
        # check if there is compression to be applied
        if self._is_to_compress():
            # check if compression is yet to be applied
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
        
        module = self.get_module()
        if keep_block_b:
            module = module.block_b
        else:
            module = module.block_a
        
        self.set_module(module)

        if verbose:
            kept = "block_b" if keep_block_b else "block_a"
            print(f"Cancelled VCONBlock at {self.path}, kept {kept}.")  
        self.vcon_initialized = False

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
        
        # Apply magnitude-based structured pruning
        if self.pruning_ratio > 0:
            if verbose:
                if self.soft_applied:
                    if hard:
                        print(f"Pruning already hard applied to module {self.path}, making the changes permanent (HARD mode)")
                    else:
                        print(f"Pruning already applied to module {self.path}, skipping re-application.")
                else:
                    print(f"Applying {"HARD (non-reversible) "*hard}pruning with pruning ratio {self.pruning_ratio} to module {self.path}.")

            # Prune module
            if self.soft_applied:
                prune_mask = self.module.prune_mask
            else:
                prune_mask, kept = self._structured_pruning(module, norm=2, pruning_ratio=self.pruning_ratio)
            
            if hard:
                # Apply hard pruning by removing the pruned rows
                module.weight.data = module.weight.data[prune_mask]
                module.bias.data = module.bias.data[prune_mask] if module.bias is not None else None
                module.out_features = kept
            else:
                # Apply soft pruning by setting the prune mask
                module.set_prune_mask(prune_mask)

            self.soft_applied = True # Flag the application as soft, do not overwrite/reinitialize

        # Apply Low Rank Decomposition (LRD)
        if self.lrd_rank and self.lrd_rank != "full":
            if verbose:
                if self.soft_applied:
                    if hard:
                        print(f"LRD already hard applied to module {self.path}, making the changes permanent (HARD mode)")
                    else:
                        print(f"LRD already applied to module {self.path}, skipping re-application.")
                else:
                    print(f"Applying {"HARD (non-reversible) "*hard}LRD with rank {self.lrd_rank} to module {self.path}.")

            if not hard:
                # Store original weight matrix to allow restoring/vanishing contributions
                module.weight_original = torch.nn.Parameter(module.weight.data.clone())
            else:
                if hasattr(module, 'weight_original'):
                    del module.weight_original
            
            if not self.soft_applied:
                # Replace the original weight with the decomposed weights
                W1, W2 = self._low_rank_decomposition(module, self.lrd_rank)
                module.weight = torch.nn.Parameter(W1)
                module.weight_2 = torch.nn.Parameter(W2.t())
                module.set_lrd_rank(self.lrd_rank)

            self.soft_applied = True # Flag the application as soft, do not overwrite/reinitialize
            self.hard_applied = hard # Flag the application as hard; changes cannot be reverted    

    def restore(self, verbose=False):
        """
        Restores the original state of the module by removing pruning and Low-Rank Decomposition (LRD) modifications.
        If the module is wrapped in a VCONBlock, restoration is applied only to the second block (`block_b`).
        The function reverses any soft-applied changes, such as pruning masks and LRD, but raises an error if hard-applied (non-reversible) changes have been made.

        Args:
            verbose (bool, optional): If True, prints information about the restoration process. Defaults to False.

        Args:
            verbose (bool, optional): If True, prints information about the restoration process. Defaults to False.

        Raises:
            ValueError: If the module has been hard applied and cannot be restored.
        """
        if self.hard_applied:
            raise ValueError("Cannot restore a module that has been hard applied (non-reversible changes).")
        
        module = self.get_module()
        if self.vcon_initialized:
            module = module.block_b # apply to block_b only
        
        if hasattr(module, 'weight_original') or module.prune_mask is not None:
            if verbose:
                print(f"Restoring module {self.path} to its original state.")
        
        # Restore original weights if LRD was applied
        if hasattr(module, 'weight_original'):
            module.weight = torch.nn.Parameter(module.weight_original.data.clone())
            del module.weight_original
            module.set_lrd_rank("full")
        
        # Remove pruning by deleting the prune mask
        module.reset_prune_mask()

        self.soft_applied = False # Reset the soft application flag

    def __repr__(self):
        return (f"CompressionScheme(name={self.name}, block_id={self.block_id}, "
                f"path={self.path}, pruning_ratio={self.pruning_ratio}, "
                f"lrd_rank={self.lrd_rank}, is_qkv_concatenated={self.is_qkv_concatenated}, "
                f"module={self.get_module().__class__.__name__})")

    def _structured_pruning(self, module, norm=2, pruning_ratio=0.0) -> torch.Tensor:
        """
        Returns a mask for structured pruning based on the specified norm.
        This function generates a binary mask that can be applied to the weight tensor
        to zero out entire rows (structured pruning) based on their L2 norm.

        Args:
            module (torch.nn.Module): The module containing the weight tensor to be pruned.
            norm (int): The norm to use for calculating the importance of rows. Default is 2 (L2 norm).
            pruning_ratio (float): The ratio of rows to prune (between 0 and 1). Default is 0.0 (no pruning).

        Returns:
            torch.Tensor: A binary mask vector with the same number of elements as the rows of the weight tensor.
            kept (int): Number of rows kept after pruning.
        """
        weight = module.weight.data

        if pruning_ratio <= 0.0:
            # No pruning, return a mask of all ones
            return torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)
        
        if pruning_ratio >= 1.0:
            # Full pruning, return a mask of all zeros
            return torch.zeros(weight.size(0), dtype=torch.bool, device=weight.device)
        
        # Calculate the norm of each row
        if module.lrd_rank == "full":
            row_norms = torch.norm(weight, p=norm, dim=1)
        else:
            # For low-rank decomposed weights, only consider the first part of the weight matrix
            row_norms = torch.norm(weight[:module.out_features, :], p=norm, dim=1)
        
        # Determine the number of rows to prune
        num_rows_to_prune = int(pruning_ratio * module.out_features)
        kept = module.out_features - num_rows_to_prune
        
        if num_rows_to_prune == 0:
            return torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)
        
        # Get the indices of the rows with the smallest norms
        _, prune_indices = torch.topk(row_norms, num_rows_to_prune, largest=False)
        
        # Create a mask with ones and set the prune indices to zero
        mask = torch.ones(weight.size(0), dtype=torch.bool, device=weight.device)
        mask[prune_indices] = False
        
        return mask, kept

    def _low_rank_decomposition(self, module, rank: int) -> torch.Tensor:
        """
        Performs low-rank decomposition on the given weight matrix using SVD.

        Args:
            weight (torch.Tensor): The weight matrix to be decomposed.
            rank (int): The target rank for the decomposition.

        Returns:
            torch.Tensor: The first matrix of the low-rank decomposition.
            torch.Tensor: The second matrix of the low-rank decomposition.
        """
        weight = module.weight.data

        if rank >= min(weight.size()):
            # No decomposition possible, launch error
            raise ValueError(f"Rank {rank} must be less than the minimum dimension of the weight matrix ({weight.size(0), weight.size(1)}).")
        
        if rank == "full":
            # No decomposition needed, return original weight and identity
            return weight, torch.eye(weight.size(1), device=weight.device)
        
        # Perform SVD
        if weight.dtype != torch.float32: # Convert to float32 for SVD computation
            weight_f32 = weight.float()
        U, S, Vh = torch.linalg.svd(weight_f32, full_matrices=False)
        if weight.dtype != torch.float32: # Convert back to original dtype
            U = U.to(weight.dtype)
            S = S.to(weight.dtype)
            Vh = Vh.to(weight.dtype)
        # Keep only the top 'rank' components
        U_r = U[:, :rank]
        S_r = torch.diag(S[:rank])
        Vh_r = Vh[:rank, :]
        # Reconstruct the low-rank approximation
        W1 = U_r @ S_r
        W2 = Vh_r
        return W1, W2

__all__ = ["CompressionScheme"]