"""Input-channel pruning with baseline bias compensation (FLAP, An et al., 2023)."""

import torch
import torch.nn as nn

from .abstract import Compressor
from .validators import boolean, in_range, one_of, positive_int

VALID_CHANNEL_PRUNING_METHODS = ["flap", "magnitude"]  # magnitude: uncalibrated, scores from the weight norms (no calibration data, no bias)


class ChannelPruner(Compressor):
    """Prune input channels of a layer, scored by their fluctuation, with a bias that stands in for them.

    Configured on the **consumer** layer (e.g. ``mlp.down_proj``): its input channels are scored with the
    calibration statistics of its own input (``input_mean``, ``input_sq_mean``), the keep-mask comes from the
    global ``flap_scores`` summary (one threshold over all configured layers, so the pruning ratio adapts per layer),
    and the removed channels are replaced by their baseline (mean) value through an added bias
    ``b0 = W[:, pruned] @ mean[pruned]``.

    Soft apply zeroes the pruned weight columns (re-zeroed after optimiser steps by ``reapply_mask``) and adds the
    bias. Hard apply also removes the columns and the matching rows of the producer layers (resolved from the model
    dictionary), which requires the producers' output size to equal this layer's input size (e.g. the MLP; not
    GQA attention).
    """

    MASK_BUFFER = "input_mask"
    BACKUP_ATTR = "_channel_prune_backup"

    def __init__(self, config):
        self.config = config
        self.ratio = self.config["ratio"]
        self.method = self.config["method"]
        self.bias_compensation = self.config["bias_compensation"]
        self.group_size = self.config.get("group_size", 1)
        self.rank_group = self.config.get("rank_group", "all")
        self.calibration_store = None
        self.scheme = None

    def set_calibration_store(self, calibration_data):
        self.calibration_store = calibration_data

    def needs_calibration(self):
        if not self._to_compress() or self.method == "magnitude":
            return ()
        return ("input_mean", "input_sq_mean")

    def needs_global_calibration(self):
        if not self._to_compress():
            return {}
        return {f"{self.method}_scores" if self.method == "magnitude" else "flap_scores": {"ratio": self.ratio}}

    def _to_compress(self):
        return self.ratio > 0.0

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return
        validate_channel_pruning_method(self.method)
        validate_channel_pruning_ratio(self.ratio)
        validate_bias_compensation(self.bias_compensation)
        validate_channel_group_size(self.group_size)
        if module.in_features % self.group_size:
            raise ValueError(f"group_size {self.group_size} must divide in_features {module.in_features}.")
        if getattr(module, "skip", False):
            return
        if isinstance(getattr(module, "rank", "full"), int):
            raise NotImplementedError("channel_pruning does not support low-rank decomposed layers.")

        if self.MASK_BUFFER not in module._buffers:
            store_key = "magnitude_scores" if self.method == "magnitude" else "flap_scores"
            result = self.scheme.global_calibration_data.get(store_key)
            if result is None or self.scheme.path not in result["keep"]:
                raise RuntimeError(
                    f"channel_pruning needs the global '{store_key}' calibration summary for this layer; "
                    "call manager.run_calibration()/apply() with calibration data set."
                )
            keep = result["keep"][self.scheme.path].to(module.weight.device)
            mean = result["mean"][self.scheme.path].to(module.weight.device)
            module.register_buffer(self.MASK_BUFFER, keep)
            with torch.no_grad():
                w = module.weight
                if self.bias_compensation:
                    pruned = ~keep
                    b0 = (w.float()[:, pruned] @ mean[pruned]).to(w.dtype)
                    setattr(module, self.BACKUP_ATTR, dict(bias=None if module.bias is None else module.bias.detach().clone()))
                    if module.bias is None:
                        module.bias = nn.Parameter(b0)
                    else:
                        module.bias.add_(b0)
                else:
                    setattr(module, self.BACKUP_ATTR, dict(bias=None if module.bias is None else module.bias.detach().clone()))
            self.reapply_mask(module)
        if hard:
            self._hard_remove(module, module._buffers[self.MASK_BUFFER])

    def _hard_remove(self, module, keep):
        keep = keep.to(torch.bool)
        sources = self.scheme.coupled_source_modules()
        if any(src.weight.shape[0] != module.in_features for src in sources) or not sources:
            raise NotImplementedError(
                "Hard channel pruning needs producer layers whose output size equals this layer's input size "
                f"(got {[tuple(s.weight.shape) for s in sources]} for in_features={module.in_features})."
            )
        with torch.no_grad():
            module.weight = nn.Parameter(module.weight.detach()[:, keep].clone(), requires_grad=module.weight.requires_grad)
            module.in_features = int(keep.sum())
            for src in sources:
                src.weight = nn.Parameter(src.weight.detach()[keep].clone(), requires_grad=src.weight.requires_grad)
                if src.bias is not None:
                    src.bias = nn.Parameter(src.bias.detach()[keep].clone(), requires_grad=src.bias.requires_grad)
                src.out_features = int(keep.sum())
                src._buffers.pop("weight_mask", None)
        module._buffers.pop(self.MASK_BUFFER, None)

    def reapply_mask(self, module):
        if not self._to_compress():
            return
        keep = module._buffers.get(self.MASK_BUFFER)
        if keep is None:
            return
        with torch.no_grad():
            module.weight.mul_(keep.to(module.weight.dtype).unsqueeze(0))

    def remove_mask(self, module):
        module._buffers.pop(self.MASK_BUFFER, None)

    def restore(self, module):
        backup = getattr(module, self.BACKUP_ATTR, None)
        if backup is not None:
            if backup["bias"] is None:
                module.bias = None
            else:
                with torch.no_grad():
                    module.bias.copy_(backup["bias"])
            delattr(module, self.BACKUP_ATTR)
        module._buffers.pop(self.MASK_BUFFER, None)
        self.config["ratio"] = 0.0
        self.ratio = 0.0

    def __repr__(self):
        return f"ChannelPruner(method={self.method}, ratio={self.ratio}, group_size={self.group_size}, bias_compensation={self.bias_compensation})"


validate_channel_pruning_method = one_of("Channel pruning method", VALID_CHANNEL_PRUNING_METHODS)
validate_channel_pruning_ratio = in_range("Channel pruning ratio", 0.0, 1.0)
validate_bias_compensation = boolean("bias_compensation")
validate_channel_group_size = positive_int("Channel pruning group_size")


def validate_rank_group(value) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"rank_group must be a non-empty string, got {value!r}.")

__all__ = ["ChannelPruner", "validate_channel_pruning_method", "validate_channel_pruning_ratio", "validate_bias_compensation", "validate_channel_group_size", "validate_rank_group"]
