"""Integer row format shared by the ``int8`` and ``int4`` KV-cache precision atoms."""

import math

import torch

from .base import KVPrecision

# One scale per stored row, kept as two bytes after the codes: an exponent byte (e + 128)
# and a mantissa byte (m), scale = (1 + m / 256) * 2 ** e.
SCALE_BYTES = 2
_MIN_ABS = 2.0 ** -100   # floor of a row's max |x|: keeps the exponent inside its byte
_LN2 = math.log(2.0)


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


__all__ = ["IntegerKVPrecision", "SCALE_BYTES"]
