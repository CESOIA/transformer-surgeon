"""Projections and pruning-aware geometry shared by the attention atoms."""

import math

import torch

from ..layers import linear_from_spec
from ..positions.rope import apply_rope_multihead, build_rope_prune_projection
from .kernels import ATTENTION_KERNELS


class MHABase(torch.nn.Module):
    """Multi-head attention projections (``q_proj``, ``k_proj``, ``v_proj``, ``out_proj``).

    Args:
        embed_dim: Model width.
        num_heads: Number of query heads.
        kv_num_heads: Number of key/value heads (grouped-query attention); defaults
            to ``num_heads``.
        attn_impl: Attention kernel name (see ``kernels/``); ``None`` picks
            ``"sdpa"`` if ``use_sdpa`` else ``"manual"``.
        linear_specs: Optional ``{projection name: spec}`` with the geometry of
            already-compressed projections (see ``layers.linear_from_spec``).
        dtype: Parameter dtype.

    Projections may have been pruned before conversion. The per-head dims are
    therefore read from the projections' current ``out_features`` (see
    ``key_head_dim``/``value_head_dim``), while ``head_dim`` keeps the original
    value. Geometry that depends on loaded weights is frozen once by the
    ``finalize_*`` hooks, which the converter calls after copying the weights.
    """

    projections = ("q_proj", "k_proj", "v_proj", "out_proj")

    def __init__(self, embed_dim, num_heads, kv_num_heads=None, attn_impl=None, use_sdpa=False,
                 linear_specs=None, dtype=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert kv_num_heads is None or num_heads % kv_num_heads == 0, "num_heads must be divisible by kv_num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_num_heads = num_heads if kv_num_heads is None else kv_num_heads
        self.use_sdpa = use_sdpa
        self.kv_out_dim = self.kv_num_heads * self.head_dim
        self.dtype = dtype

        specs = linear_specs or {}
        self.q_proj = linear_from_spec(specs.get("q_proj"), embed_dim, embed_dim, dtype=dtype)
        self.k_proj = linear_from_spec(specs.get("k_proj"), embed_dim, self.kv_out_dim, dtype=dtype)
        self.v_proj = linear_from_spec(specs.get("v_proj"), embed_dim, self.kv_out_dim, dtype=dtype)
        self.out_proj = linear_from_spec(specs.get("out_proj"), embed_dim, embed_dim, dtype=dtype)

        self.attn_impl = attn_impl or self._default_attn_impl()
        if self.attn_impl not in ATTENTION_KERNELS:
            raise ValueError(f"Unsupported attn_impl {self.attn_impl!r}; expected one of {tuple(ATTENTION_KERNELS)}")
        self.kernel = ATTENTION_KERNELS[self.attn_impl]

        # RoPE-pruning geometry: frozen by finalize_rope_pruning() at conversion time.
        self.rope_pruned = False
        self.register_buffer("rope_freq_proj", None, persistent=False)

    def _default_attn_impl(self):
        return "sdpa" if self.use_sdpa else "manual"

    def _project_qkv(self, x, seq_len):
        """Project ``x`` and split heads: ``(seq_len, heads, per-head dim)`` each."""
        q = self.q_proj(x).view(seq_len, self.num_heads, self.q_proj.out_features // self.num_heads)
        k = self.k_proj(x).view(seq_len, self.kv_num_heads, self.key_head_dim)
        v = self.v_proj(x).view(seq_len, self.kv_num_heads, self.value_head_dim)
        return q, k, v

    def _apply_rope(self, q, k, rope):
        if rope is None:
            return q, k
        cos, sin = rope
        cos_q, sin_q, cos_k, sin_k = self._project_rope(cos, sin)
        return apply_rope_multihead(q, cos_q, sin_q), apply_rope_multihead(k, cos_k, sin_k)

    @property
    def score_scale(self):
        """1/sqrt(original head_dim): pruning q/k channels must not change the softmax
        temperature (it is a property of the model, and the soft-pruned HF model keeps it)."""
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def key_head_dim(self):
        """Per-head key dim from k_proj's *current* out_features (pruned-aware)."""
        return self.k_proj.out_features // self.kv_num_heads

    @property
    def value_head_dim(self):
        """Per-head value dim from v_proj's *current* out_features (pruned-aware).

        v_proj is uncoupled from q/k rotary pruning, so this stays at head_dim
        until (unsupported) value pruning lands.
        """
        return self.v_proj.out_features // self.kv_num_heads

    def finalize_head_pruning(self):
        """Adopt the query-head count left by block pruning (whole heads removed,
        ``compression/block_pruning.py``). Conversion-time hook like
        ``finalize_rope_pruning``: call once after q_proj's weights and its
        ``block_prune_mask`` buffer (kept heads) are loaded, before the RoPE
        hook and any forward(). head_dim is unchanged; the heads kept per KV
        group are equal, so grouped attention stays regular."""
        kept = getattr(self.q_proj, "block_prune_mask", None)
        if kept is None:
            return
        num_heads = int(kept.sum().item())
        if self.q_proj.out_features != num_heads * self.head_dim:
            raise ValueError(
                f"q_proj has {self.q_proj.out_features} rows but block_prune_mask keeps {num_heads} heads "
                f"of head_dim {self.head_dim}."
            )
        if num_heads % self.kv_num_heads:
            raise ValueError(f"{num_heads} query heads left by block pruning do not split evenly over "
                             f"{self.kv_num_heads} KV heads.")
        if getattr(self.q_proj, "rope_prune_mask", None) is not None:
            raise NotImplementedError("Block pruning combined with RoPE-linked structured pruning of q_proj "
                                      "is not supported.")
        self.num_heads = num_heads

    def finalize_rope_pruning(self):
        """Freeze this module's RoPE-pruning geometry from q_proj/k_proj's
        *current* shapes and ``rope_prune_mask`` buffers. Instantiation-time
        setup: call this once, after weights (and any ``rope_prune_mask``
        buffers left by structured pruning) have been loaded and before the
        first ``forward()`` call -- ``export.convert.convert_for_export`` does
        this right after copying q_proj/k_proj's state during HF-model ->
        blocks/ conversion. blocks/ modules are a frozen, already-compressed
        model; this is the only place pruning geometry is
        decided, so ``_project_rope`` can stay a pure, static runtime
        application with no shape checks or mask lookups of its own.

        Sets ``self.rope_pruned`` and, if pruned, ``self.rope_freq_proj`` --
        the static per-kv-group 0/1 selection matrix (see
        ``build_rope_prune_projection`` in ``blocks/positions/rope_pruning.py``) used by
        ``_project_rope`` to re-index precomputed ``cos``/``sin`` onto the
        rotary frequencies that survived pruning.

        A freshly constructed module is already unpruned (``__init__`` sets
        ``rope_pruned = False``) and never needs this called. Raises
        ``ValueError`` if only one of q_proj/k_proj was pruned, or to
        different head dims -- the pair must always be pruned together
        (coupled via ``pruning.position_linked`` / ``pruning.coupled_masks``)
        for RoPE to remain valid.
        """
        q_head_dim = self.q_proj.out_features // self.num_heads
        k_head_dim = self.k_proj.out_features // self.kv_num_heads
        q_pruned = q_head_dim < self.head_dim
        k_pruned = k_head_dim < self.head_dim

        if not q_pruned and not k_pruned:
            self.rope_pruned = False
            self.rope_freq_proj = None
            return

        if q_pruned != k_pruned:
            raise ValueError(
                "Structured pruning was applied to only one of q_proj/k_proj "
                f"(head_dim: q_proj={q_head_dim}, k_proj={k_head_dim}, original="
                f"{self.head_dim}). RoPE requires q_proj and k_proj to be pruned "
                "together (they are coupled via pruning.position_linked / "
                "pruning.coupled_masks) -- prune both with a shared mask, or "
                "neither."
            )

        if q_head_dim != k_head_dim:
            raise ValueError(
                f"q_proj and k_proj were pruned to different head dims "
                f"({q_head_dim} vs {k_head_dim}); RoPE requires them to match."
            )

        q_mask = getattr(self.q_proj, "rope_prune_mask", None)
        k_mask = getattr(self.k_proj, "rope_prune_mask", None)
        if q_mask is None or k_mask is None:
            raise ValueError(
                "q_proj/k_proj head_dim has shrunk but no rope_prune_mask was "
                "recorded by structured pruning; cannot determine which rotary "
                "frequencies survived. This should not happen if pruning went "
                "through StructuredPruner on a position_linked scheme."
            )

        self.rope_freq_proj = build_rope_prune_projection(
            q_mask, k_mask, self.head_dim, self.num_heads, self.kv_num_heads,
        )
        self.rope_pruned = True

    def _project_rope(self, cos, sin):
        """Apply this module's already-finalized RoPE-pruning geometry to
        precomputed ``cos``/``sin``, splitting them per q/k. Pure runtime
        application -- see ``finalize_rope_pruning`` for how
        ``self.rope_pruned``/``self.rope_freq_proj`` are decided; this method
        performs no shape checks or mask lookups, only the (data-independent)
        projection/broadcast math, so it stays cheap and static under
        ``torch.export``.

        When this module was never pruned, ``cos``/``sin`` are returned
        as-is (shared for q and k): they already carry a singleton head
        dimension (``(seq_len, 1, head_dim//2)``) that broadcasts across all
        heads in ``apply_rope_multihead`` -- standard RoPE, no repeat.

        When pruned, the projection is **per kv-group** (different kv-groups
        may keep different rotary frequencies): it yields a group-wise
        ``cos``/``sin`` that ``repeat_interleave``s to each group's query
        heads for ``q``, and is used one-per-head for ``k``. Returns
        ``(cos_q, sin_q, cos_k, sin_k)``.
        """
        if not self.rope_pruned:
            return cos, sin, cos, sin

        proj = self.rope_freq_proj.to(device=cos.device, dtype=cos.dtype)
        num_groups, kept, half = proj.shape
        group_size = self.num_heads // num_groups

        # Project cos/sin (..., 1, half) per kv-group -> (..., num_groups, kept).
        lead = cos.shape[:-2]
        proj_flat = proj.reshape(num_groups * kept, half)
        cos_g = (cos.reshape(-1, half) @ proj_flat.t()).reshape(*lead, num_groups, kept)
        sin_g = (sin.reshape(-1, half) @ proj_flat.t()).reshape(*lead, num_groups, kept)

        # q broadcasts each group to its query heads; k uses one per kv head.
        cos_q = cos_g.repeat_interleave(group_size, dim=-2)
        sin_q = sin_g.repeat_interleave(group_size, dim=-2)
        return cos_q, sin_q, cos_g, sin_g


__all__ = ["MHABase"]
