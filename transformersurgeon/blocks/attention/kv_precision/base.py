"""Common interface of the KV-cache precision atoms."""


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


__all__ = ["KVPrecision"]
