import torch.nn as nn


class Conv2dCompressed(nn.Conv2d):
    """
    Conv2d preprocessing layer (e.g. ViT patch-embed), compatible with
    structured pruning and quantization. ``weight``/``bias``/``out_channels``/
    ``in_channels`` are native to ``nn.Conv2d`` and already in the layout
    ``StructuredPruner``/soft ``Quantizer`` expect (``out_channels`` at dim 0,
    same convention as ``LinearCompressed``) -- no attribute translation
    needed, unlike ``EmbeddingCompressed``.

    LRD is not supported: a low-rank factorization of a conv kernel isn't a
    drop-in replacement the way it is for a matmul weight, so this class
    intentionally has no ``rank``/``linear_V``/``init_lrd`` -- attempting
    ``manager.set("lrd", ...)`` on it fails loudly instead of silently
    corrupting the module.
    """
    def __repr__(self):
        return f"Conv2dCompressed({super().__repr__()})"


class Conv3dCompressed(nn.Conv3d):
    """Conv3d counterpart of :class:`Conv2dCompressed` (e.g. Qwen-VL's
    temporal patch-embed). See :class:`Conv2dCompressed` for details."""
    def __repr__(self):
        return f"Conv3dCompressed({super().__repr__()})"


__all__ = ["Conv2dCompressed", "Conv3dCompressed"]
