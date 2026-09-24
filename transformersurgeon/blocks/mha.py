# Simple blocks implementation for transformers
# Multi Head Attention (MHA) blocks
import math
import torch
import torch.nn.functional as F
from .rope import apply_rope_multihead, build_rope_prune_projection
from . import LinearCompressed  

def attention(query, key, value, attn_mask=None):
    """
    Explicit implementation of scaled dot-product attention.
    This is needed for cases where torch's built-in SDPA is not suited for model export (e.g., ONNX).
    It supports GQA automatically by handling different head dimensions for query and key/value.
    key_cache and value_cache are provided separately to minimize the concatenation overhead. They are used only if is_causal is False.

    Args:
        query: Tensor of shape (q_seq_length, q_head_num, q_head_dim)
        key:   Tensor of shape (kv_seq_length, kv_head_num, k_head_dim)
        value: Tensor of shape (kv_seq_length, kv_head_num, v_head_dim)
        attn_mask: Optional additive mask broadcastable against the scores'
            trailing (q_seq_length, kv_seq_length) dims -- e.g. (1, kv_seq_length)
            as built by TransformerDecoder, or (1, 1, q_seq_length, kv_seq_length).

    Returns:
        Tensor of shape (q_head_num, q_seq_length, v_head_dim).
    """
    q_seq_len, q_head_num, q_head_dim = query.size()
    _, kv_head_num, k_head_dim = key.size()
    _, _, v_head_dim = value.size()
    group_size = q_head_num // kv_head_num

    dtype = query.dtype
    # Score scale uses the query/key head dim. Under position-linked structured
    # pruning q/k shrink together (k_head_dim == q_head_dim) while value stays at
    # its own (possibly larger) v_head_dim, so key and value are handled with
    # their own last dims rather than one shared head_dim.
    scale = 1.0 / math.sqrt(q_head_dim)

    # GQA by broadcast, not by materialization: group the query heads under their
    # kv head and let matmul broadcast the single key/value head across the group,
    # so the KV cache is never copied into a repeat_interleave'd (q_head_num-wide)
    # tensor.
    #
    # This ordering matters a great deal on QNN/HTP. The previous formulation
    # expanded first and transposed second, which meant the Transpose ran over a
    # group_size-times-larger tensor: for a 2048-slot cache with 14 q heads / 2 kv
    # heads that is a 7.3 MB intermediate, too big for VTCM, so the HTP compiler
    # spilled it to DDR. Measured per attention op at that size (AOT "DDR
    # bandwidth summary", SM8650): 9.18 MB spill + 9.18 MB fill + 9.2 MB
    # read/write, versus 0 MB spill and ~0.02 MB read here -- and the key permute
    # is fused into the matmul's input layout rather than materialized at all.
    # The matching vendor reference is
    # executorch/examples/qualcomm/oss_scripts/llama/model/static_llama.py, which
    # likewise keeps the cache in matmul layout and never permutes a cache-sized
    # tensor.
    #
    # The math is unchanged -- this is the same contraction, so results match the
    # old path to float round-off (~1e-9 in fp32) -- and XNNPACK partitions it at
    # least as well (one fewer expand_copy outside the delegate).
    query = query.transpose(0, 1).reshape(kv_head_num, group_size, q_seq_len, q_head_dim)
    key   = key.permute(1, 2, 0).unsqueeze(1)   # (kv_head_num, 1, k_head_dim, kv_len)
    value = value.transpose(0, 1).unsqueeze(1)  # (kv_head_num, 1, kv_len, v_head_dim)

    scores = torch.matmul(query * scale, key)   # (kv_head_num, group, q_seq_len, kv_len)

    if attn_mask is not None:
        scores = scores + attn_mask

    scores = torch.nn.functional.softmax(scores, dim=-1)

    attn_output = torch.matmul(scores.to(dtype), value)  # (kv_head_num, group, q_seq_len, v_head_dim)
    attn_output = attn_output.reshape(q_head_num, q_seq_len, v_head_dim)

    # -------------------------------------------------------------------------
    # ALTERNATIVE — einsum GQA (also avoids KV materialization).
    # To use: comment out the block above and uncomment this one.
    # -------------------------------------------------------------------------
    # query = query.transpose(0, 1).reshape(kv_head_num, group_size, -1, q_head_dim)
    # key   = key.transpose(0, 1)    # (kv_heads, kv_len, k_head_dim)
    # value = value.transpose(0, 1)  # (kv_heads, kv_len, v_head_dim)
    #
    # scores = torch.einsum("hgsd,hkd->hgsk", query, key) * scale
    #
    # if attn_mask is not None:
    #     scores = scores + attn_mask
    #
    # scores = torch.nn.functional.softmax(scores, dim=-1)
    #
    # attn_output = torch.einsum("hgsk,hkd->hgsd", scores.to(dtype), value)
    # attn_output = attn_output.reshape(q_head_num, q_seq_len, v_head_dim)
    # -------------------------------------------------------------------------

    return attn_output.to(dtype)  # (q_head_num, q_seq_len, v_head_dim)

class MHABase(torch.nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            bias_required=None,
            kv_num_heads=None,
            use_sdpa=False,
            compression_config=None,
            dtype=None,
            **kwargs,
            ):
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

        # Setup compression configuration
        if compression_config is None:
            compression_config = {
                "q_proj": {"lrd": {"rank": "full"}},
                "k_proj": {"lrd": {"rank": "full"}},
                "v_proj": {"lrd": {"rank": "full"}},
                "out_proj": {"lrd": {"rank": "full"}}
            }

        # Setup bias requirement
        if bias_required is None:
            bias_required = {
                "q_proj": False,
                "k_proj": False,
                "v_proj": False,
                "out_proj": False
            }
        else:
            bias_required = dict(bias_required)
            bias_required.setdefault("q_proj", False)
            bias_required.setdefault("k_proj", False)
            bias_required.setdefault("v_proj", False)
            bias_required.setdefault("out_proj", False)
        
        q_lrd_rank = compression_config["q_proj"]["lrd"]["rank"]
        k_lrd_rank = compression_config["k_proj"]["lrd"]["rank"]
        v_lrd_rank = compression_config["v_proj"]["lrd"]["rank"]
        out_lrd_rank = compression_config["out_proj"]["lrd"]["rank"]
        # Structured-pruning-aware sizing for attention:
        # - q_proj/k_proj may be hard-pruned (their rotary head_dim shrinks) --
        #   forward() reads their *current* out_features rather than a fixed
        #   head_dim, and _project_rope projects RoPE's cos/sin to match (per
        #   kv-group) for position_linked q/k.
        # - v_proj may be hard-pruned too: its v->o coupling is not 1:1 under GQA
        #   (attention repeat_interleaves each kv head's value slice group_size
        #   times before o_proj), so the cascade onto o_proj's input uses
        #   coupled_repeated_pattern=group_size to repeat each kv head's kept-mask
        #   chunk accordingly. forward() reads value_head_dim and reshapes attn
        #   output to num_heads * value_head_dim (== o_proj's cascade-pruned
        #   in_features). value cache follows value_head_dim.

        self.q_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["q_proj"],
            rank=q_lrd_rank,
            dtype=dtype)
        self.k_proj = LinearCompressed(
            embed_dim,
            self.kv_out_dim,
            bias=bias_required["k_proj"],
            rank=k_lrd_rank,
            dtype=dtype)
        self.v_proj = LinearCompressed(
            embed_dim,
            self.kv_out_dim,
            bias=bias_required["v_proj"],
            rank=v_lrd_rank,
            dtype=dtype)
        self.out_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["out_proj"],
            rank=out_lrd_rank,
            dtype=dtype)

        # RoPE-pruning geometry: decided once, at instantiation, via
        # finalize_rope_pruning() -- never re-derived inside forward(). A
        # freshly constructed module (this __init__) is always unpruned;
        # utils.convert.convert_for_export calls finalize_rope_pruning()
        # once, right after q_proj/k_proj's (possibly hard-pruned) weights
        # and rope_prune_mask buffers are loaded during HF-model ->
        # blocks/ conversion, and before any forward() call. See
        # finalize_rope_pruning/_project_rope below.
        self.rope_pruned = False
        self.register_buffer("rope_freq_proj", None, persistent=False)

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

    def finalize_rope_pruning(self):
        """Freeze this module's RoPE-pruning geometry from q_proj/k_proj's
        *current* shapes and ``rope_prune_mask`` buffers. Instantiation-time
        setup: call this once, after weights (and any ``rope_prune_mask``
        buffers left by structured pruning) have been loaded and before the
        first ``forward()`` call -- ``utils.convert.convert_for_export`` does
        this right after copying q_proj/k_proj's state during HF-model ->
        blocks/ conversion. blocks/ modules are a frozen, already-compressed
        model (see CLAUDE.md); this is the only place pruning geometry is
        decided, so ``_project_rope`` can stay a pure, static runtime
        application with no shape checks or mask lookups of its own.

        Sets ``self.rope_pruned`` and, if pruned, ``self.rope_freq_proj`` --
        the static per-kv-group 0/1 selection matrix (see
        ``build_rope_prune_projection`` in ``blocks/rope.py``) used by
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

class MHAEncoder(MHABase): # No cache, no causal masking, for encoder-only use
    # TODO(custom_sdpa-encoder): extend attn_impl="custom_sdpa" to this class.
    # (torch.ops.llama.custom_sdpa, see MHACausal.) Currently only "manual"/use_sdpa=False
    # (attention()) and use_sdpa=True (F.scaled_dot_product_attention) are
    # available here -- the fused ExecuTorch kernel that gave MHACausal its
    # decode speedup (docs/investigations/XNNPACK_DECODE_SPEED_FIX.md) has no encoder-side
    # equivalent yet. custom_sdpa's op schema does support is_causal=False for
    # plain (non-causal, no-cache) self-attention, so this is plausibly a
    # straightforward addition -- but note MHACausal's is_causal=True bug/fix
    # doesn't directly apply here since MHAEncoder never has a cache to
    # over-scan in the first place (no start_pos concept, full seq_length
    # attention every call); the potential win is purely from using the fused
    # kernel instead of attention()/F.scaled_dot_product_attention, not from
    # avoiding a wasted cache scan.
    """Multi-head self-attention for encoder-only models (no cache, no causal mask).

    TODO(custom_sdpa-encoder): ``attn_impl="custom_sdpa"`` is not supported
    here yet -- only ``MHACausal`` has it. See the TODO comment above this
    docstring.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, rope=None):
        seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V. q_proj/k_proj's per-head dim may be smaller
        # than self.head_dim if position-linked structured pruning hard-pruned
        # them (see _project_rope) -- v_proj is uncoupled from that pruning and
        # always keeps self.head_dim.
        q = self.q_proj(x).view(seq_length, self.num_heads, self.q_proj.out_features // self.num_heads) # (seq_length, num_heads, q_head_dim)
        k = self.k_proj(x).view(seq_length, self.kv_num_heads, self.k_proj.out_features // self.kv_num_heads) # (seq_length, kv_num_heads, k_head_dim)
        v = self.v_proj(x).view(seq_length, self.kv_num_heads, self.value_head_dim) # (seq_length, kv_num_heads, v_head_dim)

        # Apply RoPE (expects (seq_length, num_heads, head_dim), not transposed)
        if rope is not None:
            cos, sin = rope
            cos_q, sin_q, cos_k, sin_k = self._project_rope(cos, sin)
            q = apply_rope_multihead(q, cos_q, sin_q)
            k = apply_rope_multihead(k, cos_k, sin_k)

        # Scaled dot-product attention. `attention()` takes (seq, heads, dim) and
        # transposes internally, same as `q`/`k`/`v` above -- only the SDPA path
        # needs the (heads, seq, dim) layout, so transpose just for that call.
        if self.use_sdpa:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                is_causal=False, enable_gqa=True,
            )
        else:
            attn_output = attention(q, k, v)

        # Concatenate heads and project output. The concatenated width is
        # num_heads * value_head_dim (== embed_dim unless v_proj was hard-pruned,
        # in which case out_proj's in_features was cascade-pruned to match).
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, self.num_heads * self.value_head_dim)
        output = self.out_proj(attn_output)

        return output

class MHAEncoderFusedProj(torch.nn.Module): # Qwen-style fused projection MHA (No GQA)
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias_required=None,
        use_sdpa=False,
        compression_config=None,
        ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_sdpa = use_sdpa

        # Setup compression configuration
        if compression_config is None:
            compression_config = {
                "qkv_proj": {"lrd": {"rank": "full"}},
                "out_proj": {"lrd": {"rank": "full"}}
            }

        # Setup bias requirement
        if bias_required is None:
            bias_required = {
                "qkv_proj": False,
                "out_proj": False
            }
        else:
            bias_required = dict(bias_required)
            bias_required.setdefault("qkv_proj", False)
            bias_required.setdefault("out_proj", False)
        
        qkv_lrd_rank = compression_config["qkv_proj"]["lrd"]["rank"]
        out_lrd_rank = compression_config["out_proj"]["lrd"]["rank"]
        
        # Instantiate layers
        self.qkv_proj = LinearCompressed(
            embed_dim, 
            3 * embed_dim, 
            bias=bias_required["qkv_proj"],
            rank=qkv_lrd_rank)
        self.out_proj = LinearCompressed(
            embed_dim,
            embed_dim,
            bias=bias_required["out_proj"],
            rank=out_lrd_rank)
        
    def forward(self, x, rope=None):
        seq_length, embed_dim = x.size()
        
        # Project inputs to Q, K, V in a single projection
        qkv = self.qkv_proj(x).view(seq_length, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        if rope is not None:
            cos, sin = rope
            q = apply_rope_multihead(q, cos, sin)
            k = apply_rope_multihead(k, cos, sin)
    
        # Scaled dot-product attention
        if self.use_sdpa:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=False)
        else:
            attn_output = attention(q, k, v)
        
        # Concatenate heads and project output
        attn_output = attn_output.permute(1, 0, 2).view(seq_length, embed_dim).contiguous()
        output = self.out_proj(attn_output)
        
        return output
    
CACHE_IMPLS = ("mutable", "io_scatter", "io_concat")
ATTN_IMPLS = ("manual", "sdpa", "custom_sdpa")


def _ensure_custom_sdpa_ops():
    """Import ExecuTorch's LLM custom-ops extension, registering
    torch.ops.llama.custom_sdpa / update_cache as a side effect of the
    module-level torch.ops.load_library(...) inside custom_ops.py. Must run
    before torch.export() traces the model -- MHACausal.__init__ always runs
    (via convert_for_export) well before any backend exporter calls
    torch_export(), so this ordering is naturally satisfied.
    """
    try:
        import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "attn_impl='custom_sdpa' requires ExecuTorch's LLM custom-ops "
            "extension (torch.ops.llama.custom_sdpa / update_cache). Install "
            "the `executorch` extra so `executorch.extension.llm.custom_ops` "
            "is importable."
        ) from e


class MHACausal(MHABase): # Causal MHA with caching for decoder use
    """Causal multi-head attention over a fixed-size KV cache.

    The cache write is selected by ``cache_impl`` (see ``CACHE_IMPLS``):

    - ``"mutable"``   : in-place ``index_copy_`` on an internal buffer. The cache
                        is module state (not graph I/O). QNN-only; the current
                        default so existing exports/tests are unchanged.
    - ``"io_scatter"``: functional ``index_put`` on a cache passed in as a
                        forward argument and returned as output. Portable to
                        QNN (ScatterNd) and TensorRT (ScatterND).
    - ``"io_concat"`` : scatter-free functional write built from a positional
                        mask + ``where`` (Concat/select family). Universal
                        fallback for backends lacking an index_put converter.

    All three produce an updated **fixed-size** ``(max_cache_len, kv_num_heads,
    head_dim)`` cache and then run the identical mask + attention code, so the
    three paths are numerically equivalent. The framework feeds one token at a
    time (``in_seq_len == 1``); the writers assume that.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache_impl = kwargs.get("cache_impl", "mutable")
        if self.cache_impl not in CACHE_IMPLS:
            raise ValueError(
                f"Unsupported cache_impl {self.cache_impl!r}; expected one of {CACHE_IMPLS}"
            )

        # add_batch_dim: accept a leading (size-1) batch dim on `x` (and thus on
        # q/k/v right after projection) instead of the framework's usual bare
        # (in_seq_len, embed_dim) contract -- see forward() for how it's absorbed
        # immediately after q/k/v projection (attention/cache internals below are
        # completely unaffected: they still only ever see unbatched tensors) and
        # re-attached on the output so the residual stream carries a batch dim
        # end to end. Opt-in and False by default so QNN/TensorRT and existing
        # XNNPACK exports are unaffected; see AGENTS.md for the export config
        # option (`add_batch_dim` in convert_options) that threads this through.
        self.add_batch_dim = kwargs.get("add_batch_dim", False)

        # Initialize key and value caches
        self.max_cache_length=kwargs.get("max_cache_len", 2048)

        # Buffers back the "mutable" path; harmless (and cheap) for io_* modes,
        # where they also serve as the initial cache when none is passed in.
        # Sized from key_head_dim/value_head_dim so a hard-pruned k_proj (shrunk
        # rotary head_dim) gets a matching key cache; v is uncoupled and keeps
        # value_head_dim. At construction these equal head_dim (weights are loaded
        # -- and possibly resized -- afterwards), so the mutable path re-checks
        # geometry lazily in forward via _ensure_cache_geometry.
        self.register_buffer(
            "key_cache",
            torch.zeros(
                self.max_cache_length,
                self.kv_num_heads,
                self.key_head_dim,
                dtype=self.dtype),
            persistent=False
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                self.max_cache_length,
                self.kv_num_heads,
                self.value_head_dim,
                dtype=self.dtype),
            persistent=False
        )

        # attn_impl selects the attention kernel: "manual" (explicit GQA-aware
        # softmax/matmul, the default), "sdpa" (torch.nn.functional.scaled_dot_
        # product_attention, equivalent to the old use_sdpa=True), or
        # "custom_sdpa" (ExecuTorch's fused CPU custom ops -- optimized for
        # XNNPACK/portable-CPU export, untested with QNN/TensorRT). Derives
        # from use_sdpa when not given explicitly, so existing callers that
        # only pass use_sdpa are unaffected.
        self.attn_impl = kwargs.get("attn_impl", None) or ("sdpa" if self.use_sdpa else "manual")
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"Unsupported attn_impl {self.attn_impl!r}; expected one of {ATTN_IMPLS}")

        if self.attn_impl == "custom_sdpa":
            if self.cache_impl != "mutable":
                raise ValueError(
                    "attn_impl='custom_sdpa' requires cache_impl='mutable': "
                    "torch.ops.llama.update_cache in-place-mutates a dedicated "
                    "batched fp32 cache buffer, which has no functional "
                    "(io_scatter/io_concat) equivalent."
                )
            _ensure_custom_sdpa_ops()
            # Dedicated fp32, batched (bsz=1) caches -- torch.ops.llama.custom_sdpa/
            # update_cache require rank-4 (bsz, seqlen, kv_heads, head_dim) float32
            # tensors, unlike this class's existing unbatched, model-dtype buffers.
            self.register_buffer(
                "custom_sdpa_key_cache",
                torch.zeros(1, self.max_cache_length, self.kv_num_heads, self.key_head_dim, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "custom_sdpa_value_cache",
                torch.zeros(1, self.max_cache_length, self.kv_num_heads, self.value_head_dim, dtype=torch.float32),
                persistent=False,
            )

    def _ensure_cache_geometry(self):
        """Reallocate the internal (mutable-path) KV buffers if q/k pruning has
        changed the key head_dim since construction. No-op once sized correctly."""
        if self.key_cache.shape[-1] != self.key_head_dim:
            self.key_cache = torch.zeros(
                self.max_cache_length, self.kv_num_heads, self.key_head_dim,
                dtype=self.key_cache.dtype, device=self.key_cache.device,
            )
        if self.value_cache.shape[-1] != self.value_head_dim:
            self.value_cache = torch.zeros(
                self.max_cache_length, self.kv_num_heads, self.value_head_dim,
                dtype=self.value_cache.dtype, device=self.value_cache.device,
            )

        if self.attn_impl == "custom_sdpa":
            q_head_dim = self.q_proj.out_features // self.num_heads
            if q_head_dim != self.value_head_dim:
                raise RuntimeError(
                    "attn_impl='custom_sdpa' requires q_head_dim == value_head_dim "
                    "(torch.ops.llama.custom_sdpa sizes its output like `query`, "
                    f"not `value`); got q_head_dim={q_head_dim}, value_head_dim="
                    f"{self.value_head_dim}. This happens when RoPE-linked "
                    "structured pruning shrank q/k's head_dim while v_proj stayed "
                    "unpruned -- use attn_impl='manual' or 'sdpa' for such models."
                )
            if self.custom_sdpa_key_cache.shape[-1] != self.key_head_dim:
                self.custom_sdpa_key_cache = torch.zeros(
                    1, self.max_cache_length, self.kv_num_heads, self.key_head_dim,
                    dtype=torch.float32, device=self.custom_sdpa_key_cache.device,
                )
            if self.custom_sdpa_value_cache.shape[-1] != self.value_head_dim:
                self.custom_sdpa_value_cache = torch.zeros(
                    1, self.max_cache_length, self.kv_num_heads, self.value_head_dim,
                    dtype=torch.float32, device=self.custom_sdpa_value_cache.device,
                )

    def _write_mutable(self, k, v, write_index, key_cache, value_cache):
        """In-place write on the internal buffers (QNN peak path)."""
        with torch.no_grad():
            self.key_cache.index_copy_(0, write_index, k)
            self.value_cache.index_copy_(0, write_index, v)
        return self.key_cache, self.value_cache

    def _write_scatter(self, k, v, write_index, key_cache, value_cache):
        """Functional scatter -> updated fixed-size cache (aten.index_put)."""
        key_cache = key_cache.index_put((write_index,), k)
        value_cache = value_cache.index_put((write_index,), v)
        return key_cache, value_cache

    def _write_concat(self, k, v, write_index, key_cache, value_cache):
        """Scatter-free write: positional one-hot mask + where (in_seq_len==1)."""
        idx = torch.arange(self.max_cache_length, device=k.device)
        sel = (idx == write_index).view(self.max_cache_length, 1, 1)  # (L,1,1)
        k_row = k.reshape(1, self.kv_num_heads, self.key_head_dim)
        v_row = v.reshape(1, self.kv_num_heads, self.value_head_dim)
        key_cache = torch.where(sel, k_row, key_cache)
        value_cache = torch.where(sel, v_row, value_cache)
        return key_cache, value_cache

    def _forward_custom_sdpa(self, q, k, v, pos_id, attn_mask):
        """attn_impl == 'custom_sdpa': ExecuTorch's fused CPU custom ops.
        float32-only, batched (bsz=1) layout -- see _ensure_cache_geometry for
        the q_head_dim==value_head_dim precondition and __init__ for the
        cache_impl=='mutable' constraint this path requires.
        """
        orig_dtype = q.dtype
        start_pos = pos_id[0].item()

        q_b = q.unsqueeze(0).to(torch.float32)  # (1, in_seq_len, num_heads, q_head_dim)
        k_b = k.unsqueeze(0).to(torch.float32)  # (1, in_seq_len, kv_num_heads, key_head_dim)
        v_b = v.unsqueeze(0).to(torch.float32)  # (1, in_seq_len, kv_num_heads, value_head_dim)

        _ = torch.ops.llama.update_cache(k_b, self.custom_sdpa_key_cache, start_pos)
        _ = torch.ops.llama.update_cache(v_b, self.custom_sdpa_value_cache, start_pos)

        q_head_dim = self.q_proj.out_features // self.num_heads
        scale = 1.0 / math.sqrt(q_head_dim)  # matches attention()'s own convention

        # Let the kernel derive the causal mask itself from start_pos
        # (is_causal=True, attn_mask=None) instead of handing it an explicit
        # full-length mask with start_pos=0.
        #
        # This is not just a style difference, it changes the amount of work:
        # with start_pos=0 + explicit mask the kernel has no way to know which
        # cache slots are live, so it scores the query against *all*
        # max_cache_len slots and then masks the invalid ones away. Told the
        # real start_pos with is_causal=True, it only scores positions
        # [0, start_pos + in_seq_len) -- at decode step p that is O(p) instead
        # of O(max_cache_len) per layer per token. This is exactly what
        # ExecuTorch's own llama exporter does (see SDPACustom.forward in
        # executorch/examples/models/llama/source_transformation/sdpa.py, the
        # use_attention_mask=False branch, which is its default).
        #
        # Semantically identical: `attn_mask` as built by TransformerDecoder is
        # plain causal (mask out k_pos > q_pos), which is precisely what
        # is_causal=True generates. The only numeric difference is that the
        # explicit mask used a finite -10000.0 penalty while the kernel uses a
        # true -inf/skip, i.e. the kernel result is if anything slightly more
        # exact. `attn_mask` is therefore intentionally unused on this path --
        # it is still built for (and used by) the "manual"/"sdpa" kernels.
        attn_output = torch.ops.llama.custom_sdpa(
            q_b, self.custom_sdpa_key_cache, self.custom_sdpa_value_cache,
            start_pos,                  # real position: bounds the kernel's scan
            None,                       # no explicit mask -- derived from is_causal
            0.0,                        # dropout
            True,                       # is_causal
            scale,
        )  # (1, in_seq_len, num_heads, value_head_dim)

        return attn_output.squeeze(0).to(orig_dtype).transpose(0, 1)  # (num_heads, in_seq_len, value_head_dim)

    def forward(self, x, pos_id, attn_mask,
                key_cache=None, value_cache=None, rope=None):
        """
        Forward pass of the causal Multi-Head Attention with caching.

        Args:
            x (torch.Tensor): Input tensor of shape (in_seq_len, embed_dim), or
                (1, in_seq_len, embed_dim) when ``self.add_batch_dim``.
            pos_id (int): Current length of the cache (number of tokens already in cache). This is used to determine where to write the new keys and values in the cache.
            attn_mask (torch.Tensor): Precomputed additive causal mask for this
                decode step, shape (1, max_cache_len). Callers (TransformerDecoder)
                compute this once per forward() call and pass the same tensor to
                every layer -- it depends only on pos_id, not on layer-specific
                state, so it must not be re-derived per layer.
            key_cache/value_cache (torch.Tensor, optional): Incoming fixed-size
                caches for the ``io_*`` modes. If omitted, the internal buffers
                are used as the initial cache.
            rope (tuple, optional): Tuple of (cos, sin) tensors for RoPE application.

        Returns:
            output (mutable mode) or (output, key_cache, value_cache) (io modes).
        """
        if self.add_batch_dim:
            _, in_seq_len, embed_dim = x.size()
        else:
            in_seq_len, embed_dim = x.size()

        # Match the internal buffers to the current (possibly pruned) key head_dim.
        self._ensure_cache_geometry()

        # Fall back to internal buffers when caches are not threaded in.
        if key_cache is None:
            key_cache = self.key_cache
        if value_cache is None:
            value_cache = self.value_cache

        # Project inputs to Q, K, V. q_proj/k_proj's per-head dim may be smaller
        # than self.head_dim if position-linked structured pruning hard-pruned
        # them (see _project_rope) -- v_proj is uncoupled from that pruning and
        # keeps value_head_dim. The key cache follows the pruned key_head_dim (set
        # by _ensure_cache_geometry / build_zero_caches), so a hard-pruned k_proj
        # writes into a matching cache.
        #
        # When add_batch_dim, q_proj/k_proj/v_proj (LinearCompressed) see a
        # batched (1, in_seq_len, embed_dim) input and produce a batched
        # (1, in_seq_len, out_features) output -- but since the leading dim is
        # always exactly 1, .view(in_seq_len, heads, head_dim) below absorbs it
        # directly (same total element count either way), so q/k/v end up
        # unbatched from this point on regardless of add_batch_dim. Everything
        # downstream (RoPE, attention, cache read/write, custom_sdpa) is
        # therefore identical in both modes -- only the *projections themselves*
        # (and the final out_proj below) actually see a batch dim.
        q = self.q_proj(x).view(in_seq_len, self.num_heads, self.q_proj.out_features // self.num_heads) # (in_seq_len, num_heads, q_head_dim)
        k = self.k_proj(x).view(in_seq_len, self.kv_num_heads, self.key_head_dim) # (in_seq_len, kv_num_heads, k_head_dim)
        v = self.v_proj(x).view(in_seq_len, self.kv_num_heads, self.value_head_dim) # (in_seq_len, kv_num_heads, v_head_dim)

        # Apply RoPE. `rope` is already indexed by pos_id (done once per decode
        # step by TransformerDecoder, shared across all layers) -- each layer
        # only applies its own (possibly pruning-projected) split on top.
        if rope is not None:
            cos, sin = rope
            cos_q, sin_q, cos_k, sin_k = self._project_rope(cos, sin)
            q = apply_rope_multihead(q, cos_q, sin_q)
            k = apply_rope_multihead(k, cos_k, sin_k)

        if self.attn_impl == "custom_sdpa":
            # custom_sdpa owns its own dedicated fp32 cache (see __init__/
            # _ensure_cache_geometry) and writes it via torch.ops.llama.update_cache
            # inside _forward_custom_sdpa -- the unbatched key_cache/value_cache
            # buffers above are left untouched/unused on this path.
            attn_output = self._forward_custom_sdpa(q, k, v, pos_id, attn_mask)
        else:
            # Write new K/V into the fixed-size cache (mechanism per cache_impl).
            write_index = torch.clamp(pos_id, 0, self.max_cache_length-1).long()
            if self.cache_impl == "mutable":
                key_cache, value_cache = self._write_mutable(k, v, write_index, key_cache, value_cache)
            elif self.cache_impl == "io_scatter":
                key_cache, value_cache = self._write_scatter(k, v, write_index, key_cache, value_cache)
            else:  # io_concat
                key_cache, value_cache = self._write_concat(k, v, write_index, key_cache, value_cache)

            if self.attn_impl == "sdpa":
                q_t = q.transpose(0, 1) # (num_heads, in_seq_len, head_dim)
                k_t = key_cache.transpose(0, 1) # (kv_num_heads, pos_id, head_dim)
                v_t = value_cache.transpose(0, 1) # (kv_num_heads, pos_id, head_dim)
                attn_output = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, is_causal=False, enable_gqa=True)
            else:
                attn_output = attention(q, key_cache, value_cache, attn_mask)

        # Concatenate heads and project output. Width is num_heads * value_head_dim
        # (== embed_dim unless v_proj was hard-pruned, in which case out_proj's
        # in_features was cascade-pruned to match). Re-attach the batch dim here
        # (if add_batch_dim) so out_proj -- and everything downstream in the
        # residual stream (norm/mlp/next layer) -- sees a batched input/output,
        # same as q/k/v's projections above.
        out_shape = (
            (1, in_seq_len, self.num_heads * self.value_head_dim) if self.add_batch_dim
            else (in_seq_len, self.num_heads * self.value_head_dim)
        )
        attn_output = attn_output.transpose(0, 1).reshape(out_shape)
        output = self.out_proj(attn_output)

        if self.cache_impl == "mutable":
            return output
        return output, key_cache, value_cache

__all__ = [
    "MHAEncoder",
    "MHACausal",
    "MHAEncoderFusedProj",
    "CACHE_IMPLS",
    "ATTN_IMPLS",
    ]
