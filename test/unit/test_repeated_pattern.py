from __future__ import annotations

import torch

from transformersurgeon.compression.structured_pruning_methods import (
    build_structured_mask,
    reduce_member_to_patterns,
    reduce_pattern_scores,
    reduce_scores,
    tile_pattern_mask,
)


def test_reduce_pattern_scores_repeat_chunks_groups():
    # 6 groups of granularity 2: chunk into runs of N=2 groups, reduce each
    # chunk independently instead of reducing all 6 groups into one pattern.
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    reduced = reduce_pattern_scores(scores, granularity=2, op="add", repeat=2)
    # chunk0 = groups (1,2)+(3,4) -> [4,6]; chunk1 = (5,6)+(7,8) -> [12,14];
    # chunk2 = (9,10)+(11,12) -> [20,22]
    assert reduced.tolist() == [4.0, 6.0, 12.0, 14.0, 20.0, 22.0]


def test_reduce_pattern_scores_repeat_requires_divisibility():
    scores = torch.arange(12, dtype=torch.float32)  # 6 groups of granularity 2
    import pytest

    with pytest.raises(ValueError):
        reduce_pattern_scores(scores, granularity=2, op="add", repeat=4)


def test_tile_pattern_mask_repeat_tiles_each_chunk_n_times():
    # Two independent length-2 patterns, each meant to repeat 3 times.
    chunk_mask = torch.tensor([True, False, False, True])
    tiled = tile_pattern_mask(chunk_mask, out_dim=12, granularity=2, repeat=3)
    expected = [True, False] * 3 + [False, True] * 3
    assert tiled.tolist() == expected


def test_repeated_pattern_n_matches_user_example():
    # granularity=4, repeated_pattern=2: three independent masks
    # [0,1,1,0], [0,1,0,0], [1,0,1,0], each applied twice consecutively.
    granularity = 4
    ratio = 0.25  # prune 1 of 4 positions per pattern
    op = "add"

    torch.manual_seed(0)
    scores = torch.rand(6 * granularity)  # 6 groups -> 3 chunks of N=2

    reduced = reduce_pattern_scores(scores, granularity, op, repeat=2)
    chunk_mask = build_structured_mask(reduced, ratio, granularity)
    out_dim = scores.numel()
    tiled = tile_pattern_mask(chunk_mask, out_dim, granularity=granularity, repeat=2)

    chunks = chunk_mask.view(3, granularity)
    tiled_view = tiled.view(6, granularity)
    for i in range(3):
        assert torch.equal(tiled_view[2 * i], chunks[i])
        assert torch.equal(tiled_view[2 * i + 1], chunks[i])


def test_repeated_pattern_max_behaviour_unchanged():
    # repeat=None (True/"max") still reduces every group into one pattern.
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 3 groups of 2
    reduced = reduce_pattern_scores(scores, granularity=2, op="add")
    assert reduced.tolist() == [9.0, 12.0]

    pattern = torch.tensor([True, False])
    tiled = tile_pattern_mask(pattern, out_dim=6)
    assert tiled.tolist() == [True, False, True, False, True, False]


# --- GQA auto-derived per-kv-group patterns -------------------------------


def test_reduce_member_to_patterns_duplicates_smaller_member():
    # k head (1 group per pattern) is duplicated up to target_gpp=3 before summing,
    # so it contributes 3x -- equal weight to a member with 3 groups per pattern.
    k = torch.tensor([1.0, 2.0, 3.0, 4.0])  # P=2 patterns of granularity 2, gpp=1
    reduced = reduce_member_to_patterns(k, granularity=2, num_patterns=2, op="add", target_gpp=3)
    assert reduced.tolist() == [3.0, 6.0, 9.0, 12.0]  # each pattern position x3


def test_auto_gqa_shared_mask_layout():
    # q_proj: 14 heads x head_dim 64; k_proj: 2 heads x 64. granularity=head_dim.
    g, num_q, num_k = 64, 14, 2
    torch.manual_seed(0)
    q, k = torch.rand(num_q * g), torch.rand(num_k * g)

    counts = [num_q, num_k]           # group counts at granularity g
    P = min(counts)                   # 2 == num_kv_heads
    max_gpp = max(c // P for c in counts)  # 7

    qp = reduce_member_to_patterns(q, g, P, "add", max_gpp)
    kp = reduce_member_to_patterns(k, g, P, "add", max_gpp)
    assert qp.shape == kp.shape == (P * g,)  # no shape mismatch

    reduced = reduce_scores([qp, kp], "add")
    mask = build_structured_mask(reduced, ratio=0.25, granularity=g)  # 2 patterns of 64
    patterns = mask.view(P, g)
    assert not torch.equal(patterns[0], patterns[1])  # two DISTINCT masks
    assert patterns.sum(1).tolist() == [g - g // 4, g - g // 4]  # same count each

    q_mask = tile_pattern_mask(mask, num_q * g, granularity=g, repeat=num_q // P).view(num_q, g)
    k_mask = tile_pattern_mask(mask, num_k * g, granularity=g, repeat=num_k // P).view(num_k, g)
    # q heads 0..6 -> mask0, 7..13 -> mask1; k head 0 -> mask0, head 1 -> mask1.
    for i in range(7):
        assert torch.equal(q_mask[i], patterns[0])
        assert torch.equal(q_mask[i + 7], patterns[1])
    assert torch.equal(k_mask[0], patterns[0])
    assert torch.equal(k_mask[1], patterns[1])


def test_auto_gqa_rope_ties_real_imag():
    # position_linked (q/k): each per-kv-group 64-pattern must keep freq i and
    # i+32 together -- fold halves, build [P*32] mask, expand back to [P*64].
    g, P = 64, 2
    torch.manual_seed(1)
    reduced = torch.rand(P * g)
    half = g // 2
    folded = reduce_pattern_scores(reduced, half, "add", repeat=2)
    assert folded.shape == (P * half,)
    half_mask = build_structured_mask(folded, ratio=0.25, granularity=half)
    full = half_mask.view(P, half).repeat(1, 2).reshape(-1).view(P, g)
    for p in range(P):
        assert torch.equal(full[p][:half], full[p][half:])  # real == imag
