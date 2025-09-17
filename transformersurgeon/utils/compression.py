import copy
import inspect
import torch
import torch.nn.utils.prune as prune
from .VCONBlock import VCONBlock

# PROBLEM: when pruning a layer, the next layer should also be adjusted accordingly
# but this is not easy for skip connections, e.g., residual connections in transformers
# Maybe it is possible to do so by storing the masks of the previouse layers and use torch.scatter (that might be supported in executorch) to insert zeros where needed

class CompressionScheme:
    def __init__(self,
                 path,
                 pruning_ratio, #output_paths, 
                 lrd_rank,
                 is_qkv_concatenated=False,
                 model=None):
        self.path = path
        self.pruning_ratio = pruning_ratio
        # self.output_paths = output_paths # blocks after this layer, input should be pruned accordingly
        self.lrd_rank = lrd_rank
        self.is_qkv_concatenated = is_qkv_concatenated
        self.model = model

        self.hard_applied = False # this flags the compression as non-reversible
        self.soft_applied = False # this flags the compression as already-peformed: do not overwrite/reinitialize
        self.vcon_initialized = False # this flags if the VCONBlock has been initialized

    def get_module(self):
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
        return (self.pruning_ratio > 0) or (self.lrd_rank and self.lrd_rank != "full")
    
    def _module_copy(self, module):
        # Get the __init__ signature
        sig = inspect.signature(type(module).__init__)
        # Build kwargs from module attributes
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if hasattr(module, name):
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
        Duplicate module and instantiate a VCONBlock
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
        Removes the VCONBlock and keeps either block_a or block_b.
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

    def set_vcon_beta(self, beta: float):
        """
        Sets the beta value of the VCONBlock to control the contribution of each block.
        """   
        if not self.vcon_initialized:
            raise ValueError("VCONBlock is not initialized, cannot set beta.")
        
        module = self.get_module() # get the VCONBlock
        module.set_beta(beta)
        
    def apply(self, hard=False, verbose=False):
        """       
        Applies pruning and LRD to the module specified by the path.
        If the module is wrapped in a VCONBlock, this is applied only to the second block (block_b).
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
                newW = torch.concat((W1, W2.t()), dim=0)
                module.weight.data = newW
                module.set_lrd_rank(self.lrd_rank)

            self.soft_applied = True # Flag the application as soft, do not overwrite/reinitialize
            self.hard_applied = hard # Flag the application as hard; changes cannot be reverted    

    def restore(self, verbose=False):
        """
        Restores the original module by removing pruning and LRD.
        If the module is wrapped in a VCONBlock, this is applied only to the second block (block_b).
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
        return (f"CompressionScheme(path={self.path}, pruning_ratio={self.pruning_ratio}, "
                f"lrd_rank={self.lrd_rank}, is_qkv_concatenated={self.is_qkv_concatenated}, "
                f"module={self.module.__class__.__name__ if self.module else None})")

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
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        # Keep only the top 'rank' components
        U_r = U[:, :rank]
        S_r = torch.diag(S[:rank])
        Vh_r = Vh[:rank, :]
        # Reconstruct the low-rank approximation
        W1 = U_r @ S_r
        W2 = Vh_r
        return W1, W2

__all__ = ["CompressionScheme"]