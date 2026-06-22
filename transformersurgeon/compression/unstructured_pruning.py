import logging
import torch

from .abstract import Compressor
from .unstructured_pruning_methods import METHOD_FUNCTIONS

logger = logging.getLogger(__name__)


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

            # Register mask as a module buffer so it survives restore() and is
            # included in state_dict(). The mask is 2-D (same shape as weight):
            # 1 = keep element, 0 = prune element.
            module.register_buffer('weight_mask', mask)

            # Apply the pruning mask to the module in-place.
            # Zeroed elements still receive non-zero gradients during backward
            # (dL/dW = x^T @ dL/dy has no dependence on W), giving STE behaviour.
            with torch.no_grad():
                module.weight.mul_(mask.to(module.weight.dtype))

        if hard:
            # TODO: convert to sparse tensor representation for hardware acceleration.
            # Hard unstructured pruning is currently identical to soft: weights are
            # zeroed in-place and the mask buffer is registered on the module.
            logger.warning(
                "Hard unstructured pruning for '%s' has no sparse-operator backend. "
                "Falling back to in-place zeroing, same as soft pruning.",
                getattr(module, 'name', repr(module)),
            )

    def restore(self, module):
        # Reset config only. The weight_mask buffer is intentionally left on
        # the module so the same mask can be re-applied after restoration.
        # Call remove_mask() to explicitly drop it.
        self.config["ratio"] = 0.0
        self.config["method"] = "magnitude"
        self.config["granularity"] = "layer"

    def reapply_mask(self, module):
        """Re-zero pruned elements using the stored weight_mask buffer.

        Call after optimizer.step() in STE fine-tuning loops to prevent mask
        drift.
        """
        mask = module._buffers.get('weight_mask')
        if mask is None:
            return
        with torch.no_grad():
            module.weight.mul_(mask.to(module.weight.dtype))

    def remove_mask(self, module):
        """Drop the weight_mask buffer from the module."""
        module._buffers.pop('weight_mask', None)

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