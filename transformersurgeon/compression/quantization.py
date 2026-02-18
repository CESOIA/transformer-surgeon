import torch
from .abstract import Compressor
from .pruning import (
    _unstructured_mask,
    validate_pruning_criterion,
)
from typing import Union

def _quantize(weight, qbits, norm=2) -> torch.Tensor:
    """
    Perform single scale/zero-point quantization of the weight matrix.
    This function preserves the norm of the original weight matrix.

    Args:
        weight (torch.Tensor): The weight matrix to be quantized.
        qbits (int): The number of bits for quantization.
        norm (int): The norm to use for scaling the quantized weights. Default is 2 (L2 norm).

    Returns:
        torch.int32: The quantized weight matrix.
        torch.float32: The scale factor used for quantization.
        torch.float32: The zero point used for quantization.
        int: Quantization block rows number. In this implementation, it is set to the number of columns in the original weight matrix (input features).
        int: Quantization block columns number. In this implementation, it is set to the number of rows in the original weight matrix (output features).
    """
    # Evaluate number of quantization intervals
    q_levels = 2 ** qbits

    # Evaluate norm of the original weight matrix
    w_norm = weight.norm(p=norm)

    # Quantize weights to integers in the range [0, q_levels-1]
    w_min, w_max = weight.min(), weight.max()
    w_centered = weight - w_min
    w_scale = (w_max - w_min) / (q_levels - 1 + 1e-8) # Add small epsilon to avoid division by zero
    w_q = torch.round(w_centered / w_scale).clamp(0, q_levels - 1)

    # Evaluate norm of the quantized weight matrix
    w_q_norm = (w_q * w_scale + w_min).norm(p=norm)

    # Adjust scale to preserve the norm of the original weight matrix
    w_scale = w_scale * (w_norm / (w_q_norm + 1e-8)) # Add small epsilon to avoid division by zero

    # Calculate zero point to preserve the minimum value of the original weight matrix
    w_zero_point = w_min

    block_r, block_c = weight.size() # Single block quantization
    return w_q.to(torch.int32), w_scale.to(weight.dtype), w_zero_point.to(weight.dtype), block_r, block_c

def _binarize(weight):
    """
    Binarize the weight matrix using the sign function.

    Args:
        weight (torch.Tensor): The weight matrix to be binarized.

    Returns:
        torch.int8: The binarized weight matrix.
    """
    w_b = torch.sign(weight).to(torch.int8)
    scale = weight.abs().mean()
    zero = torch.tensor([0.0])
    block_r, block_c = weight.size() # Single block binarization
    return w_b, scale, zero, block_r, block_c

class Quantizer(Compressor):
    def __init__(
        self,
        precision="full",
        sparsity=0.0,
        sparse_criterion="magnitude",
        sparse_reverse=False
        ):
        # Configuration
        self.precision = precision
        self.sparsity = sparsity
        self.sparse_criterion = sparse_criterion
        self.sparse_reverse = sparse_reverse

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return # No compression needed based on the configuration

        # Extract quantization configuration
        precision = self.precision
        sparsity = self.sparsity
        sparse_criterion = self.sparse_criterion
        sparse_reverse = self.sparse_reverse

        if precision:
            if not hard and not soft_applied:
                module.init_soft_quantization(precision, sparsity)
                # Possibly apply sparse quantizaition if configured
                if sparsity > 0.0:
                    mask = _unstructured_mask(
                        module.weight,
                        criterion=sparse_criterion,
                        pruning_ratio=sparsity,
                        reverse=sparse_reverse
                        )
                    module.qmask.copy_(mask)
            if hard:
                # NOT IMPLEMENTED
                raise NotImplementedError("Hard quantization is not implemented yet.")

    def restore(self, module):
        module.precision = "full" # Reset precision to full to restore original weights
        module.qsparsity = 0.0 # Reset qsparsity to 0 to restore original weights
        
    def _to_compress(self):
        # Check if quantization has to be applied based on the precision configuration
        return self.precision != "full"
    
    def __repr__(self):
        string = f"Quantizer(precision={self.precision}, sparsity={self.sparsity}, sparse_criterion='{self.sparse_criterion}', sparse_reverse={self.sparse_reverse})"
        return string
        
def validate_precision(precision: Union[str, int]) -> None:
    if precision is not None:
        return
    
    if isinstance(precision, str):
        if precision not in ["full", "binary"]:
            raise ValueError(
                f"Precision must be an integer, 'full' or 'binary', but got '{precision}'."
            )
    elif isinstance(precision, int):
        if precision < 2 or precision > 16:
            raise ValueError(
                f"Precision must be a positive integer between 2 and 16, but got {precision}."
            )
    else:
        raise ValueError(
            f"Precision must be an integer, 'full' or 'binary', but got type {type(precision)}."
        )
        
def validate_sparsity(sparsity: float) -> None:
    if sparsity < 0.0 or sparsity > 1.0:
        raise ValueError(f"Sparsity must be between 0.0 and 1.0, but got {sparsity}.")
    
def validate_sparse_criterion(criterion: str) -> None:
    validate_pruning_criterion(criterion)

def validate_sparse_reverse(reverse: bool) -> None:
    if not isinstance(reverse, bool):
        raise ValueError(f"Sparse reverse must be a boolean value, but got type {type(reverse)}.")
        
__all__ = [
    "Quantizer",
    "validate_precision",
    "validate_sparsity",
    "validate_sparse_criterion",
    "validate_sparse_reverse"
]