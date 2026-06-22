import logging
import torch

from .abstract import Compressor
from .structured_pruning_methods import METHOD_FUNCTIONS

logger = logging.getLogger(__name__)


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

            # Register mask as a module buffer so it survives restore() and is
            # included in state_dict(). The mask is 1-D over the output dimension:
            # 1 = keep row, 0 = prune row.
            module.register_buffer('weight_mask', prune_mask)

            # Apply the pruning mask to the weights and biases (if present) in-place.
            # Zeroed rows still receive non-zero gradients during the backward pass
            # (dL/dW = x^T @ dL/dy has no dependence on W), giving STE behaviour.
            with torch.no_grad():
                dtype = module.weight.dtype
                module.weight.mul_(prune_mask.unsqueeze(1).to(dtype))
                if module.bias is not None:
                    module.bias.mul_(prune_mask.to(dtype))

        if hard:
            # TODO: actual row removal (requires adjacent-layer coordination).
            # Hard structured pruning is currently identical to soft: weights are
            # zeroed in-place and the mask buffer is registered on the module.
            logger.warning(
                "Hard structured pruning for '%s' is not yet implemented as actual "
                "row removal (requires adjacent-layer coordination). Falling back to "
                "in-place zeroing, same as soft pruning.",
                getattr(module, 'name', repr(module)),
            )

    def restore(self, module):
        # Reset config only. The weight_mask buffer is intentionally left on
        # the module so the same mask can be re-applied after restoration
        # (e.g. to start STE fine-tuning from fresh weights). Call
        # remove_mask() to explicitly drop it.
        self.config["ratio"] = 0.0
        self.config["method"] = "magnitude"

    def reapply_mask(self, module):
        """Re-zero pruned rows using the stored weight_mask buffer.

        Call after optimizer.step() in STE fine-tuning loops to prevent mask
        drift (pruned positions receive non-zero gradient updates and must be
        zeroed back after each parameter update).
        """
        mask = module._buffers.get('weight_mask')
        if mask is None:
            return
        with torch.no_grad():
            dtype = module.weight.dtype
            module.weight.mul_(mask.unsqueeze(1).to(dtype))
            if module.bias is not None:
                module.bias.mul_(mask.to(dtype))

    def remove_mask(self, module):
        """Drop the weight_mask buffer from the module."""
        module._buffers.pop('weight_mask', None)

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