import torch

from .abstract import Compressor
from .structured_pruning_methods import METHOD_FUNCTIONS


VALID_METHODS = ["magnitude", "gradient", "random"]
CALIBRATED_METHOD_DICT = {
    "gradient": ("weight_grad",),
}


class StructuredPruner(Compressor):
    def __init__(self, config):
        self.config = config
        self.ratio = self.config["ratio"]
        self.method = self.config.get("method", self.config.get("criterion", "magnitude"))
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

        validate_structured_pruning_ratio(ratio)
        validate_structured_pruning_method(method)

        self.config["ratio"] = ratio
        self.config["method"] = method

        if ratio > 0 and not soft_applied:

            # To maintain flexibility for future methods, we handle methods with different input requirements separately.

            if method == "magnitude":
                prune_mask = METHOD_FUNCTIONS[method](
                    module.weight,
                    pruning_ratio=ratio,
                )
            elif method == "gradient":
                weight_grad = self.calibration_store["weight_grad"]
                prune_mask = METHOD_FUNCTIONS[method](
                    module.weight,
                    weight_grad,
                    pruning_ratio=ratio,
                )
            elif method == "random":
                prune_mask = METHOD_FUNCTIONS[method](
                    module.weight,
                    pruning_ratio=ratio,
                )
            else:
                raise ValueError(f"Unsupported structured pruning method '{method}'.")

            # Apply the pruning mask to the weights and biases (if present) in-place
            with torch.no_grad():
                dtype = module.weight.dtype
                module.weight.mul_(prune_mask.unsqueeze(1).to(dtype))
                if module.bias is not None:
                    module.bias.mul_(prune_mask.to(dtype))

        if hard:
            raise Warning("Hard structured pruning is not implemented yet. Currently, parameters are masked in-place, so hard pruning is effectively the same as soft pruning.")

    def restore(self, module):
        # N.B. module argument is not used but required for API consistency with other compressors
        self.config["ratio"] = 0.0
        self.config["method"] = "magnitude"

    def _to_compress(self):
        return self.ratio > 0.0

    def __repr__(self):
        return f"StructuredPruner(ratio={self.ratio}, method='{self.method}')"


### CONFIGURATION VALIDATORS ###

def validate_structured_pruning_ratio(ratio: float) -> None:
    if ratio is not None and not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Pruning ratio must be between 0.0 and 1.0, but got {ratio}.")

def validate_structured_pruning_method(method: str) -> None:
    if method is not None and method not in VALID_METHODS:
        raise ValueError(f"Pruning method must be one of {VALID_METHODS}, but got '{method}'.")

__all__ = [
    "StructuredPruner",
    "validate_structured_pruning_ratio",
    "validate_structured_pruning_method",
]