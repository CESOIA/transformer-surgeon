from __future__ import annotations

import torch

from transformersurgeon.compression.structured_pruning_methods import (
    build_structured_mask,
    reduce_pattern_scores,
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
