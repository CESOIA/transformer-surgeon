import warnings
import torch
import torch.nn as nn
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
        self.granularity = self.config["granularity"]
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
        granularity = self.granularity

        validate_quantization_method(method)
        validate_precision(precision)
        validate_unstructured_pruning_ratio(sparsity)
        validate_sparse_method(sparse_method)
        validate_quantization_eps(eps)
        validate_quantization_granularity(granularity)

        self.config["method"] = method
        self.config["precision"] = precision
        self.config["sparsity"] = sparsity
        self.config["sparse_method"] = sparse_method
        self.config["eps"] = eps
        self.config["granularity"] = granularity

        # Hard-path preconditions: validate and import before iterating parameters.
        if hard:
            try:
                from torchao.quantization import Int8Tensor
            except ImportError as exc:
                raise ImportError(
                    "torchao is required for hard quantization. "
                    "Install it with: pip install torchao"
                ) from exc

            if isinstance(precision, int) and precision > 8:
                raise ValueError(
                    f"Hard quantization requires precision ≤ 8 (int8 storage); got {precision}."
                )

            if precision != 8:
                warnings.warn(
                    f"Precision {precision!r} is not natively supported by standard torchao; "
                    "using custom Int8Tensor wrapper."
                )

        # Single loop: quantize once, then branch on soft vs. hard.
        if precision and (hard or not soft_applied):
            for name, param in module.named_parameters():
                if name not in ["weight", "weight_2"]:
                    continue

                qdata, scale, dequantize_fn = METHOD_FUNCTIONS[method](
                    param.data, precision, eps, granularity
                )

                if hard:
                    flat_scale = scale.to(torch.float32).reshape(-1)
                    scale_shape = (len(flat_scale), 1)
                    int8_weight = Int8Tensor(
                        qdata=qdata,
                        scale=flat_scale.reshape(scale_shape),
                        block_size=list(param.data.shape),
                        dtype=param.data.dtype,
                        zero_point=torch.zeros(scale_shape, dtype=torch.int8),
                    )
                    setattr(module, name, nn.Parameter(int8_weight, requires_grad=False))

                else:
                    qweight = dequantize_fn(qdata, scale)
                    if sparsity > 0.0:
                        if sparse_method == "magnitude":
                            mask = SPARSITY_FUNCTIONS[sparse_method](
                                param, pruning_ratio=sparsity
                            )
                        elif sparse_method == "random":
                            mask = SPARSITY_FUNCTIONS[sparse_method](
                                param, pruning_ratio=sparsity
                            )
                        else:
                            raise ValueError(f"Unsupported sparsity method '{sparse_method}'.")
                        param.data.copy_(qweight * mask + param.data * (~mask))
                    else:
                        param.data.copy_(qweight)

    def restore(self, module):
        # N.B. module argument is not used but required for API consistency with other compressors
        self.config["method"] = "vanilla"
        self.config["precision"] = "full"
        self.config["sparsity"] = 0.0
        self.config["sparse_method"] = "magnitude"
        self.config["granularity"] = "per_tensor"

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

VALID_GRANULARITIES = ["per_tensor", "per_channel"]

def validate_quantization_granularity(granularity: str) -> None:
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"Quantization granularity must be one of {VALID_GRANULARITIES}, but got '{granularity}'."
        )

__all__ = [
    "Quantizer",
    "validate_quantization_method",
    "validate_precision",
    "validate_sparse_method",
    "validate_quantization_granularity",
]
