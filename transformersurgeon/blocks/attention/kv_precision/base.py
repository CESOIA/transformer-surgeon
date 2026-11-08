"""Common interface of the KV-cache precision atoms, and the integer row format they share."""

import math

import torch

# One scale per stored row, kept as two bytes after the codes: an exponent byte (e + 128)
# and a mantissa byte (m), scale = (1 + m / 256) * 2 ** e.
SCALE_BYTES = 2
_MIN_ABS = 2.0 ** -100   # floor of a row's max |x|: keeps the exponent inside its byte
_LN2 = math.log(2.0)


class KVPrecision:
    """How one cached key or value row (one token, one KV head) is stored.

    A precision atom is a stateless strategy object owned by a causal attention
    module, next to its KV-cache atom. The KV-cache atom moves stored rows without
    looking at them; this atom turns a ``head_dim``-wide float row into its stored
    form (``encode``) and back (``decode``), so the attention kernel always reads
    float keys and values.

    - ``bits``: stored bits per element (``None``: the model dtype, unchanged).
    - ``requires_cache``: the only KV-cache atom that can hold these rows, or ``None``.
    - ``scale_dtype``: dtype of a second stored tensor per cache holding block scales, or
      ``None``. With one, a cache is the pair ``(rows, scales)`` wherever it is passed
      (``encode`` returns it, ``decode`` takes it) and the graph has a ``*_scale`` input
      and output next to each cache.
    """

    name = None
    bits = None
    requires_cache = None
    scale_dtype = None

    def storage_dtype(self, dtype):
        """dtype of the cache buffers of a module computing in ``dtype``."""
        raise NotImplementedError

    def row_width(self, head_dim):
        """Stored width of a ``head_dim``-wide row."""
        raise NotImplementedError

    def scale_width(self, head_dim):
        """Stored width of the block scales of a ``head_dim``-wide row (only with ``scale_dtype``)."""
        raise NotImplementedError

    def encode(self, x):
        """``(..., head_dim)`` float rows -> ``(..., row_width)`` stored rows (with ``scale_dtype``:
        ``(rows, scales)``, the scales ``(..., scale_width)``)."""
        raise NotImplementedError

    def decode(self, rows, dtype):
        """``(..., row_width)`` stored rows (or ``(rows, scales)``) -> ``(..., head_dim)`` rows in ``dtype``."""
        raise NotImplementedError


class IntegerKVPrecision(KVPrecision):
    """Symmetric integer codes with one scale per row, stored as ``uint8``.

    A stored row is ``[codes | exponent byte | mantissa byte]``. The scale is the
    row's max |x| divided by ``qmax``, rounded *up* to the 8-bit-mantissa format,
    so no code is clipped. Codes are offset by ``qmax + 1`` to be unsigned and
    packed by the subclass (``_pack``/``_unpack``). Encoding and decoding use only
    element-wise float arithmetic, one reduction and casts, so they trace to
    portable operations; the attention kernel reads the decoded float rows. A
    zero-filled row (a fresh cache) decodes to zero up to ~1e-37.
    """

    def storage_dtype(self, dtype):
        return torch.uint8

    @property
    def qmax(self):
        return 2 ** (self.bits - 1) - 1

    def _pack(self, codes):
        raise NotImplementedError

    def _unpack(self, packed):
        raise NotImplementedError

    def encode(self, x):
        x = x.float()
        needed = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=_MIN_ABS) / self.qmax
        exponent = torch.clamp(torch.floor(torch.log(needed) / _LN2), -128.0, 127.0)
        mantissa = torch.ceil((needed / torch.pow(2.0, exponent) - 1.0) * 256.0)
        carry = mantissa >= 256.0   # needed rounded up to the next power of two
        exponent = torch.where(carry, exponent + 1.0, exponent)
        mantissa = torch.where(carry, torch.zeros_like(mantissa), torch.clamp(mantissa, min=0.0))
        scale = (1.0 + mantissa / 256.0) * torch.pow(2.0, exponent)
        codes = torch.clamp(torch.round(x / scale), -self.qmax, self.qmax) + (self.qmax + 1)
        return torch.cat([self._pack(codes), exponent + 128.0, mantissa], dim=-1).to(torch.uint8)

    def decode(self, rows, dtype):
        rows = rows.float()
        codes = self._unpack(rows[..., :-SCALE_BYTES]) - (self.qmax + 1)
        scale = (1.0 + rows[..., -1:] / 256.0) * torch.pow(2.0, rows[..., -2:-1] - 128.0)
        return (codes * scale).to(dtype)


__all__ = ["KVPrecision", "IntegerKVPrecision", "SCALE_BYTES"]
