import torch
from .abstract import Compressor
from .pruning import (
    _unstructured_mask,
    validate_pruning_criterion,
)
from typing import Union

def _quantize_maxabs(weight, precision) -> torch.Tensor:
    """
    Perform single scale/zero-point fake-quantization of the weight matrix.
    This function preserves the norm of the original weight matrix.

    Args:
        weight (torch.Tensor): The weight matrix to be quantized.
        qbits (int): The number of bits for quantization.

    Returns:
        torch.Tensor: The fake-quantized weight matrix.
    """
    qmax = 2**precision - 1
    scale = weight.abs().max() / qmax
    q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
    return q * scale

def _quantize_binarize(weight):
    """
    Binarize the weight matrix using the sign function.

    Args:
        weight (torch.Tensor): The weight matrix to be binarized.

    Returns:
        torch.int8: The binarized weight matrix.
    """
    s = torch.sign(weight)
    scale = weight.abs().mean()
    return s * scale

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
                for param in module.parameters():
                    if param.name not in ["weight", "weight_2"]:
                        continue

                    if precision == "binary":
                        qweight = _quantize_binarize(param.data)
                    else:
                        qweight = _quantize_maxabs(param.data, precision)
                    
                    with torch.no_grad():
                        # Possibly apply sparse quantization if configured
                        if sparsity > 0.0:
                            mask = _unstructured_mask(
                                param,
                                criterion=sparse_criterion,
                                pruning_ratio=sparsity,
                                reverse=sparse_reverse
                                )
                            param.copy_(qweight * mask + param.data * (~mask))
                        else:
                            param.copy_(qweight)
                    
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