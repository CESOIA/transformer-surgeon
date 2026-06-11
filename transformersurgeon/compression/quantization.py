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

        # Hard-path: use torchao's native quantize_() API, bypassing soft-quant.
        if hard:
            _apply_torchao_hard_quantization(module, precision, granularity)
            return

        # Soft-path: fake-quant via dequantize.
        if precision and not soft_applied:
            for name, param in module.named_parameters():
                if name not in ["weight", "weight_2"]:
                    continue

                qdata, scale, dequantize_fn = METHOD_FUNCTIONS[method](
                    param.data, precision, eps, granularity
                )

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


def _apply_torchao_hard_quantization(module, precision, granularity: str) -> None:
    """Apply torchao native weight-only quantization to *module* in-place."""
    try:
        from torchao.quantization import (
            quantize_,
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
            IntxWeightOnlyConfig,
        )
        from torchao.quantization.granularity import PerRow, PerTensor, PerAxis
    except ImportError as exc:
        raise ImportError(
            "torchao is required for hard quantization. "
            "Install it with: pip install torchao"
        ) from exc

    ao_granularity_row = PerRow()        # per-channel (output dim)
    ao_granularity_tensor = PerTensor()
    ao_granularity_axis = PerAxis(axis=0)  # used by IntxWeightOnlyConfig

    per_channel = granularity == "per_channel"

    if precision == "binary":
        raise ValueError(
            "Hard quantization does not support 'binary' precision. "
            "Use soft quantization (hard=False) for binary."
        )

    if isinstance(precision, int) and precision > 8:
        raise ValueError(
            f"Hard quantization requires precision ≤ 8; got {precision}."
        )

    if precision == 8:
        config = Int8WeightOnlyConfig(
            granularity=ao_granularity_row if per_channel else ao_granularity_tensor
        )
    elif precision == 4:
        # Int4WeightOnlyConfig uses group_size-based granularity internally;
        # per_channel / per_tensor distinction is not exposed. CUDA-only — silently
        # no-ops on CPU (torchao behavior).
        config = Int4WeightOnlyConfig()
    elif isinstance(precision, int) and 2 <= precision <= 7:
        # IntxWeightOnlyConfig supports arbitrary sub-byte widths but only
        # PerAxis / PerGroup granularities (not PerRow / PerTensor).
        torch_dtype = getattr(torch, f"int{precision}")
        config = IntxWeightOnlyConfig(
            weight_dtype=torch_dtype,
            granularity=ao_granularity_axis,
        )
    else:
        raise ValueError(f"Unsupported precision for hard quantization: {precision!r}.")

    quantize_(module, config)


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
