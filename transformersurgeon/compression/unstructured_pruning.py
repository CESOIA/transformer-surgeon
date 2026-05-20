import torch

from .abstract import Compressor
from .unstructured_pruning_methods import METHOD_FUNCTIONS


VALID_METHODS = ["magnitude", "gradient", "random"]
CALIBRATED_METHOD_DICT = {
    "gradient": ("weight_grad",),
}


class UnstructuredPruner(Compressor):
    def __init__(self, config):
        self.config = config
        self.ratio = self.config["ratio"]
        self.method = self.config.get("method", self.config.get("criterion", "magnitude"))
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

        ratio = self.ratio
        method = self.method
        granularity = self.granularity

        validate_unstructured_pruning_ratio(ratio)
        validate_unstructured_pruning_method(method)
        validate_unstructured_pruning_granularity(granularity)

        self.config["ratio"] = ratio
        self.config["method"] = method
        self.config["granularity"] = granularity

        if ratio > 0 and not soft_applied:

            # To maintain flexibility for future methods, we handle methods with different input requirements separately.
            
            if method == "magnitude":
                mask = METHOD_FUNCTIONS[method](
                    module.weight,
                    pruning_ratio=ratio,
                    granularity=granularity,
                )

            elif method == "gradient":
                weight_grad = self.calibration_store["weight_grad"]
                mask = METHOD_FUNCTIONS[method](
                    module.weight,
                    weight_grad,
                    pruning_ratio=ratio,
                    granularity=granularity,
                )

            elif method == "random":
                mask = METHOD_FUNCTIONS[method](
                    module.weight,
                    pruning_ratio=ratio,
                    granularity=granularity,
                )

            else:
                raise ValueError(f"Unsupported pruning method '{method}'.")

            # Apply the pruning mask to the module
            with torch.no_grad():
                module.weight.mul_(mask.to(module.weight.dtype))

        if hard:
            raise Warning("Currently, there is no support for sparse operators. Parameters are masked in-place, so hard pruning is effectively the same as soft pruning.")

    def restore(self, module):
        # N.B. module argument is not used but required for API consistency with other compressors
        self.config["ratio"] = 0.0
        self.config["method"] = "magnitude"
        self.config["granularity"] = "layer"

    def _to_compress(self):
        return self.ratio > 0.0

    def __repr__(self):
        return (
            f"UnstructuredPruner(ratio={self.ratio}, "
            f"method='{self.method}', granularity={self.granularity})"
        )


### CONFIGURATION VALIDATORS ###

def validate_unstructured_pruning_ratio(ratio: float) -> None:
    if ratio is not None and not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Pruning ratio must be between 0.0 and 1.0, but got {ratio}.")

def validate_unstructured_pruning_method(method: str) -> None:
    if method is not None and method not in VALID_METHODS:
        raise ValueError(f"Pruning method must be one of {VALID_METHODS}, but got '{method}'.")

def validate_unstructured_pruning_granularity(granularity: str) -> None:
    valid_granularities = ["layer", "neuron"]
    if granularity is not None and granularity not in valid_granularities and not (
        isinstance(granularity, int) and granularity > 0
    ):
        raise ValueError(
            f"Pruning granularity must be one of {valid_granularities} or a positive integer, but got '{granularity}'."
        )

__all__ = [
    "UnstructuredPruner",
    "validate_unstructured_pruning_ratio",
    "validate_unstructured_pruning_method",
    "validate_unstructured_pruning_granularity",
]