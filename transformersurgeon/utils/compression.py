import torch
import torch.nn.utils.prune as prune

# PROBLEM: when pruning a layer, the next layer should also be adjusted accordingly
# but this is not easy for skip connections, e.g., residual connections in transformers
# Maybe it is possible to do so by storing the masks of the previouse layers and use torch.scatter (that might be supported in executorch) to insert zeros where needed

class CompressionScheme:
    def __init__(self,
                 path,
                 pruning_ratio, #output_paths, 
                 lrd_rank,
                 is_qkv_concatenated=False,
                 module=None):
        self.path = path
        self.pruning_ratio = pruning_ratio
        # self.output_paths = output_paths # blocks after this layer, input should be pruned accordingly
        self.lrd_rank = lrd_rank
        self.is_qkv_concatenated = is_qkv_concatenated
        self.module = module

    def get_module(self):
        # Check if model has been provided
        if not hasattr(self, 'module'):
            raise ValueError("Model is not set. Please set the model before getting the module.")

        split_path = self.path.split('.')
        # Traverse the model iteratively to find the module
        tmp_module = self.module
        for path_piece in split_path:
            tmp_module = getattr(tmp_module, path_piece, None)
        return tmp_module
    
    def apply(self, hard=False): # WIP
        """       
        Applies pruning and LRD to the module specified by the path.
        """
        module = self.get_module()
        if module is None:
            raise ValueError(f"Module at path '{self.path}' not found in the model.")

        print(f"Applying pruning with ratio {self.pruning_ratio} and LRD rank {self.lrd_rank} to module {self.path}.")

        # Apply magnitude-based structured pruning
        if self.pruning_ratio > 0:
            # Prune module
            prune_mask, kept = self._structured_pruning(module, norm=2, pruning_ratio=self.pruning_ratio)
            if hard:
                # Apply hard pruning by removing the pruned rows
                module.weight.data = module.weight.data[prune_mask]
                module.bias.data = module.bias.data[prune_mask] if module.bias is not None else None
                module.out_features = kept
            else:
                # Apply soft pruning by setting the prune mask
                module.set_prune_mask(prune_mask)


    def restore(self): # WIP
        """
        Restores the original module by removing pruning and LRD.
        """
        # WIP
        print(f"Restoring module {self.path} to its original state.")

    def __repr__(self):
        return (f"CompressionScheme(path={self.path}, pruning_ratio={self.pruning_ratio}, "
                f"lrd_rank={self.lrd_rank}, is_qkv_concatenated={self.is_qkv_concatenated}, "
                f"module={self.module.__class__.__name__ if self.module else None})")

    def _structured_pruning(module, norm=2, pruning_ratio=0.0) -> torch.Tensor:
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

    def _low_rank_decomposition(weight: torch.Tensor, rank: int) -> torch.Tensor:
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
            raise ValueError("Rank must be less than the minimum dimension of the weight matrix.")
        
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