"""Linear layer that can hold a low-rank factorisation, pruning masks and quantisation tags."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .low_rank import LowRankMixin
from .runtime_rank import active_rank


def _float_values(weight):
    """A weight as a plain tensor that can be sliced anywhere: quantized tensor subclasses (whose slicing
    must follow their blocks) are dequantized first -- the same values the layer computes with."""
    data = weight.data if isinstance(weight, nn.Parameter) else weight
    if is_traceable_wrapper_subclass(data) and hasattr(data, "dequantize"):
        return data.dequantize()
    return weight


def _scale_input(module, inputs):
    """Forward pre-hook of a layer with an ``input_scale``: ``x * input_scale`` before anything else."""
    return (inputs[0] * module.input_scale.to(inputs[0].dtype),) + tuple(inputs[1:])


class _BinarizeSTE(torch.autograd.Function):
    """sign(w) forward (zeros map to +1), identity gradient backward (straight-through estimator)."""

    @staticmethod
    def forward(ctx, w):
        return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))

    @staticmethod
    def backward(ctx, grad):
        return grad


class LinearCompressed(LowRankMixin, nn.Linear):
    """Drop-in replacement of ``nn.Linear`` used by every compressible linear layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample. ``0`` means the layer was fully
            pruned: it then returns its input unchanged.
        bias: Whether the layer has an additive bias.
        rank: Rank of the low-rank factorisation, or ``None``/``"full"`` for none.
        runtime_rank_group: Label of the runtime rank knob this (LRD) layer follows, or ``None``
            (see ``runtime_rank.py``): inside ``active_ranks({group: r})`` the layer runs with its
            first ``r`` components. Set by the compression step, only for importance-ordered factors.

    Low-rank layers expose ``lrd_hidden``, an identity between the two factors: the place where
    hooks observe or alter the inner activation ``h = V x`` (e.g. its quantisation).

    ``set_input_scale(s)`` gives the layer a per-input-channel multiplier applied to its input before
    anything else (``y = W (x * s)``), for an input whose per-channel scale cannot be folded into the
    producer (e.g. one coming out of a non-linearity). It runs as the first forward pre-hook, so
    calibration and activation fake-quant hooks see the scaled input, and it is a plain ``Mul`` in an
    exported graph. The buffer survives state-dict loads (the hook is re-installed).

    Inputs with more than two dimensions are flattened to two dimensions around
    the matrix product and restored afterwards: the same graph then lowers to a
    plain fully-connected operation on every backend (some backends, such as
    Qualcomm QNN, handle rank-2 inputs more robustly).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rank=None, device=None, dtype=None, runtime_rank_group=None):
        if out_features <= 0:
            # Fully pruned layer: a plain module that returns its input unchanged.
            nn.Module.__init__(self)
            self.skip = True
            self.in_features, self.out_features, self.rank = in_features, 0, "full"
            self.weight = self.bias = self.linear_V = None
            self.runtime_rank_group = None
            return
        self.skip = False
        super().__init__(in_features=in_features, out_features=out_features, bias=bias,
                         device=device, dtype=dtype)
        self.linear_V = None
        # Binary QAT (quantization option ``qat``): per-output-channel trainable scale, shape (out, 1).
        # When set, ``weight`` is a latent float tensor and the layer computes ``bin_scale * sign(weight)``.
        self.bin_scale = None
        self.runtime_rank_group = runtime_rank_group
        self.lrd_hidden = nn.Identity()
        self.init_lrd(rank)

    def set_input_scale(self, scale):
        """Multiply the input by ``scale`` (``(in_features,)``) before the layer; ``None`` removes it."""
        for key, hook in list(self._forward_pre_hooks.items()):
            if hook is _scale_input:
                del self._forward_pre_hooks[key]
        if scale is None:
            self._buffers.pop("input_scale", None)
            return
        scale = torch.as_tensor(scale).detach().reshape(-1)
        if scale.numel() != self.in_features:
            raise ValueError(f"input_scale has {scale.numel()} entries, the layer reads {self.in_features} channels.")
        device = self.weight.device if self.weight is not None else scale.device
        self.register_buffer("input_scale", scale.to(device=device, dtype=torch.float32).clone())
        self.register_forward_pre_hook(_scale_input, prepend=True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "input_scale" in state_dict and "input_scale" not in self._buffers:
            self.register_buffer("input_scale", torch.empty_like(state_dict[prefix + "input_scale"]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if "input_scale" in self._buffers and _scale_input not in self._forward_pre_hooks.values():
            self.register_forward_pre_hook(_scale_input, prepend=True)

    @property
    def _lrd_rows(self):
        return self.out_features

    @property
    def _lrd_cols(self):
        return self.in_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.skip:
            return x
        restore_shape = None
        if x.dim() > 2:
            restore_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        w = self.weight
        if isinstance(self.rank, int):
            r = active_rank(self.runtime_rank_group)
            if r is None:
                x = self.linear_V(x)
            else:
                if isinstance(r, int):
                    r = min(r, self.rank)
                x = F.linear(x, _float_values(self.linear_V.weight)[:r])
                w = _float_values(w)[:, :r]
            x = self.lrd_hidden(x)
        if getattr(self, "bin_scale", None) is not None:
            w = _BinarizeSTE.apply(w) * self.bin_scale.to(w.dtype)
        y = F.linear(x, w, self.bias)
        if restore_shape is not None:
            y = y.reshape(*restore_shape, y.shape[-1])
        return y

    def __repr__(self):
        return (f"LinearCompressed(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, rank={self.rank})")

    __str__ = __repr__


def linear_from_spec(spec, in_features, out_features, bias=False, dtype=None):
    """Build a :class:`LinearCompressed` from a layer specification.

    ``spec`` is a dict with any of ``in_features``, ``out_features``, ``rank``,
    ``runtime_rank_group`` and ``bias``; missing entries fall back to the given defaults. Specifications are
    produced when a model is converted for export (``export/convert.py``) and
    carry the geometry the source layer already has after compression.
    """
    spec = spec or {}
    return LinearCompressed(
        spec.get("in_features", in_features),
        spec.get("out_features", out_features),
        bias=spec.get("bias", bias),
        rank=spec.get("rank", "full"),
        dtype=dtype,
        runtime_rank_group=spec.get("runtime_rank_group"),
    )


__all__ = ["LinearCompressed", "linear_from_spec"]
