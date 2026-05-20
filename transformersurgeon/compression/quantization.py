import torch
from typing import Union

from .abstract import Compressor
from .quantization_methods import METHOD_FUNCTIONS
from .unstructured_pruning_methods import METHOD_FUNCTIONS as SPARSITY_FUNCTIONS
from .unstructured_pruning import validate_unstructured_pruning_ratio


VALID_METHODS = ["vanilla"]
VALID_SPARSE_METHODS = ["magnitude", "random"]
CALIBRATED_METHOD_DICT = {}


class Quantizer(Compressor):
    def __init__(self, config):
        self.config = config
        self.method = self.config["method"]
        self.precision = self.config["precision"]
        self.sparsity = self.config["sparsity"]
        self.sparse_method = self.config["sparse_method"]
        self.eps = self.config["eps"]
        self.calibration_store = None

    def set_calibration_store(self, calibration_data):
        self.calibration_store = calibration_data

    def needs_calibration(self):
        if not self._to_compress():
            return ()
        return CALIBRATED_METHOD_DICT.get(self.method, ())

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return

        method = self.method
        precision = self.precision
        sparsity = self.sparsity
        sparse_method = self.sparse_method
        eps = self.eps

        validate_quantization_method(method)
        validate_precision(precision)
        validate_unstructured_pruning_ratio(sparsity)
        validate_sparse_method(sparse_method)
        validate_quantization_eps(eps)

        self.config["method"] = method
        self.config["precision"] = precision
        self.config["sparsity"] = sparsity
        self.config["sparse_method"] = sparse_method
        self.config["eps"] = eps

        if precision and not hard and not soft_applied:
            # Apply quantization to all parameters in the module
            for name, param in module.named_parameters():
                if name not in ["weight", "weight_2"]:
                    continue

                # To maintain flexibility for future methods, we handle methods with different input requirements separately.

                if method == "vanilla":
                    qweight = METHOD_FUNCTIONS[method](
                        param.data,
                        precision,
                        eps
                    )

                else:
                    raise ValueError(f"Unsupported quantization method '{method}'.")

                # Support for sparse quantization (useful for iterative compression)
                if sparsity > 0.0:
                    if sparse_method == "magnitude":
                        mask = SPARSITY_FUNCTIONS[sparse_method](
                            param,
                            pruning_ratio=sparsity,
                        )
                    elif sparse_method == "random":
                        mask = SPARSITY_FUNCTIONS[sparse_method](
                            param,
                            pruning_ratio=sparsity,
                        )
                    else:
                        raise ValueError(f"Unsupported sparsity method '{sparse_method}'.")

                # Apply quantization, optionally with sparsity
                if sparsity > 0.0:
                    param.copy_(qweight * mask + param.data * (~mask))
                else:
                    param.copy_(qweight)

        if hard:
            raise Warning("Currently, there is no support for quantized operators. Parameters are quantized and dequantized in-place, so hard quantization is effectively the same as soft quantization.")

    def restore(self, module):
        # N.B. module argument is not used but required for API consistency with other compressors
        self.config["method"] = "vanilla"
        self.config["precision"] = "full"
        self.config["sparsity"] = 0.0
        self.config["sparse_method"] = "magnitude"

    def _to_compress(self):
        return self.precision != "full"

    def __repr__(self):
        return (
            f"Quantizer(method='{self.method}', precision={self.precision}, "
            f"sparsity={self.sparsity}, sparse_method='{self.sparse_method}')"
        )


### CONFIGURATION VALIDATORS ###

def validate_quantization_method(method: str) -> None:
    if method not in VALID_METHODS:
        raise ValueError(f"Quantization method must be one of {VALID_METHODS}, but got '{method}'.")


def validate_precision(precision: Union[str, int]) -> None:
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
    
def validate_sparse_method(sparse_method: str) -> None:
    if sparse_method not in VALID_SPARSE_METHODS:
        raise ValueError(f"Sparse method must be one of {VALID_SPARSE_METHODS}, but got '{sparse_method}'.")
    
def validate_quantization_eps(eps: float) -> None:
    if not isinstance(eps, (int, float)):
        raise ValueError(f"Quantization eps must be numeric, but got type {type(eps)}.")
    if eps <= 0:
        raise ValueError(f"Quantization eps must be positive, but got {eps}.")

__all__ = [
    "Quantizer",
    "validate_quantization_method",
    "validate_precision",
    "validate_sparse_method",
]
