"""Hermetic unit tests for structured (coupled) pruning and scheme grouping.

Builds a tiny model of stacked ``LinearCompressed`` layers (no downloads) with an
inline INDEXING dict that declares generic ``pruning`` annotations, then exercises:
  * pure mask/effective-dim/reduce helpers,
  * real hard neuron removal + matrix resizing,
  * per-head (granularity) pruning,
  * coupled cascade to the next layer's inputs,
  * grouping: create/delete/auto_groups, shared masks, reduce ops, and the
    group-only option rules.

Run: pytest test/compression_tests/structured_pruning_test.py -v
"""

import unittest

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from transformersurgeon.blocks import LinearCompressed
from transformersurgeon.utils import CompressionSchemesManager
from transformersurgeon.utils.configuration import init_compressed_config
from transformersurgeon.compression.structured_pruning_methods import (
    build_structured_mask,
    effective_out_features,
    effective_num_pruned,
    reduce_scores,
    reduce_pattern_scores,
    tile_pattern_mask,
)

HID = 8       # residual / hidden dim
INTER = 16    # mlp intermediate dim
LAYERS = 2

INDEXING = {
    "block": {
        "config_attr": "",
        "num_blocks_attr": "num_hidden_layers",
        "path_list": {
            "self_attn": ["v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
        },
        "calibration_groups": {},
        "pruning": {
            "output_dependence": {
                "self_attn.v_proj": ["self_attn.o_proj"],
                "mlp.gate_proj": ["mlp.down_proj"],
                "mlp.up_proj": ["mlp.down_proj"],
            },
            "coupled_masks": [
                ["mlp.gate_proj", "mlp.up_proj"],
            ],
            "coupled_masks_all": [
                ["self_attn.o_proj", "mlp.down_proj"],
            ],
            "per_head_uniform": ["self_attn.v_proj"],
        },
        "path_template": "layers.{block_index}.{path}",
    }
}


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_proj = LinearCompressed(HID, HID, bias=True)
        self.o_proj = LinearCompressed(HID, HID, bias=True)

    def forward(self, x):
        return self.o_proj(self.v_proj(x))


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = LinearCompressed(HID, INTER, bias=False)
        self.up_proj = LinearCompressed(HID, INTER, bias=False)
        self.down_proj = LinearCompressed(INTER, HID, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MLP()

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class _TinyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([_Block() for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _make_manager():
    config = PretrainedConfig()
    config.num_hidden_layers = LAYERS
    init_compressed_config(config, INDEXING["block"])
    model = _TinyModel(config)
    model.eval()
    return model, CompressionSchemesManager(model, INDEXING)


class TestPruningHelpers(unittest.TestCase):
    def test_effective_dims_layer(self):
        self.assertEqual(effective_num_pruned(16, 0.5, "layer"), 8)
        self.assertEqual(effective_out_features(16, 0.5, "layer"), 8)
        self.assertEqual(effective_out_features(16, 0.0, "layer"), 16)

    def test_effective_dims_granularity(self):
        # ratio 0.5 within chunks of 4 -> prune 2 per chunk, 4 chunks -> 8 pruned.
        self.assertEqual(effective_num_pruned(16, 0.5, 4), 8)
        with self.assertRaises(ValueError):
            effective_num_pruned(15, 0.5, 4)  # 4 does not divide 15

    def test_build_mask_layer(self):
        scores = torch.tensor([0.1, 5.0, 0.2, 4.0])
        mask = build_structured_mask(scores, 0.5, "layer")
        self.assertEqual(mask.tolist(), [False, True, False, True])

    def test_build_mask_granularity_uniform_per_chunk(self):
        # Two chunks of 2; one row pruned per chunk regardless of global ranking.
        scores = torch.tensor([9.0, 8.0, 0.1, 0.2])
        mask = build_structured_mask(scores, 0.5, 2)
        self.assertEqual(int(mask[:2].sum()), 1)
        self.assertEqual(int(mask[2:].sum()), 1)

    def test_reduce_scores(self):
        a, b = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        self.assertEqual(reduce_scores([a, b], "add").tolist(), [4.0, 6.0])
        self.assertEqual(reduce_scores([a, b], "multiply").tolist(), [3.0, 8.0])
        with self.assertRaises(ValueError):
            reduce_scores([a, b], None)


class TestHardPruningAndCascade(unittest.TestCase):
    def test_hard_removal_resizes_and_cascades(self):
        model, manager = _make_manager()
        x = torch.randn(2, 3, HID)
        # Prune the attention value path only.
        manager.set("structured_pruning", "ratio", 0.5, criteria="self_attn.v_proj")
        manager.apply(hard=True)

        v = model.layers[0].self_attn.v_proj
        o = model.layers[0].self_attn.o_proj
        self.assertEqual(v.out_features, effective_out_features(HID, 0.5, "layer"))
        self.assertEqual(v.weight.shape[0], v.out_features)
        # Coupled: o_proj input columns shrank to match v_proj output.
        self.assertEqual(o.in_features, v.out_features)
        self.assertEqual(o.weight.shape[1], v.out_features)
        # Model still runs and preserves hidden dim.
        self.assertEqual(model(x).shape, (2, 3, HID))

    def test_hard_pruning_granularity(self):
        model, manager = _make_manager()
        manager.set("structured_pruning", "ratio", 0.5, criteria="self_attn.v_proj")
        manager.set("structured_pruning", "granularity", 4, criteria="self_attn.v_proj")
        manager.apply(hard=True)
        v = model.layers[0].self_attn.v_proj
        self.assertEqual(v.out_features, effective_out_features(HID, 0.5, 4))


# --- GQA-style model: q_proj (3 heads) and k_proj (1 head), head_dim = 4 ------
HEAD_DIM = 4
INDEXING_GQA = {
    "block": {
        "config_attr": "",
        "num_blocks_attr": "num_hidden_layers",
        "path_list": {"self_attn": ["q_proj", "k_proj"]},
        "calibration_groups": {},
        "pruning": {
            "output_dependence": {},
            "coupled_masks": [["self_attn.q_proj", "self_attn.k_proj"]],
            "per_head_uniform": ["self_attn.q_proj", "self_attn.k_proj"],
        },
        "path_template": "layers.{block_index}.{path}",
    }
}


class _GQAAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = LinearCompressed(HID, 3 * HEAD_DIM, bias=False)  # 3 heads
        self.k_proj = LinearCompressed(HID, 1 * HEAD_DIM, bias=False)  # 1 head (GQA)


class _GQABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _GQAAttn()


class _GQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([_GQABlock() for _ in range(config.num_hidden_layers)])


def _make_gqa_manager():
    config = PretrainedConfig()
    config.num_hidden_layers = 1
    init_compressed_config(config, INDEXING_GQA["block"])
    model = _GQAModel(config)
    model.eval()
    return model, CompressionSchemesManager(model, INDEXING_GQA)


class TestRepeatedPattern(unittest.TestCase):
    def test_reduce_pattern_scores(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 3 groups of 2
        self.assertEqual(reduce_pattern_scores(scores, 2, "add").tolist(), [9.0, 12.0])
        with self.assertRaises(ValueError):
            reduce_pattern_scores(scores, 2, None)

    def test_tile_pattern_mask(self):
        pattern = torch.tensor([True, False])
        self.assertEqual(
            tile_pattern_mask(pattern, 6).tolist(),
            [True, False, True, False, True, False],
        )

    def test_repeated_pattern_prunes_same_position_per_group(self):
        model, manager = _make_gqa_manager()
        manager.set("structured_pruning", "method", "magnitude", criteria="self_attn.q_proj")
        manager.set("structured_pruning", "granularity", HEAD_DIM, criteria="self_attn.q_proj")
        manager.set("structured_pruning", "repeated_pattern", True, criteria="self_attn.q_proj")
        manager.set("structured_pruning", "reduce_op", "add", criteria="self_attn.q_proj")
        manager.set("structured_pruning", "ratio", 0.25, criteria="self_attn.q_proj")  # 1 of 4 per head
        manager.apply(hard=False)

        mask = model.layers[0].self_attn.q_proj._buffers["weight_mask"]
        self.assertEqual(mask.numel(), 3 * HEAD_DIM)
        per_head = mask.view(3, HEAD_DIM)
        # Same position pruned in every head.
        self.assertTrue(torch.equal(per_head[0], per_head[1]))
        self.assertTrue(torch.equal(per_head[1], per_head[2]))
        self.assertEqual(int((~per_head[0]).sum()), 1)

    def test_shared_pattern_across_gqa_q_and_k(self):
        model, manager = _make_gqa_manager()
        manager.create_group(
            ["layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj"], name="group1"
        )
        manager.set("structured_pruning", "share_mask", True, group="group1")
        manager.set("structured_pruning", "method", "magnitude", criteria="self_attn")
        manager.set("structured_pruning", "granularity", HEAD_DIM, criteria="self_attn")
        manager.set("structured_pruning", "repeated_pattern", True, criteria="self_attn")
        manager.set("structured_pruning", "reduce_op", "add", criteria="self_attn")
        manager.set("structured_pruning", "ratio", 0.25, criteria="self_attn")
        manager.apply(hard=True)

        q = model.layers[0].self_attn.q_proj
        k = model.layers[0].self_attn.k_proj
        # q had 3 heads, k had 1 head; both keep 3 of 4 positions per head.
        self.assertEqual(q.out_features, 3 * (HEAD_DIM - 1))
        self.assertEqual(k.out_features, 1 * (HEAD_DIM - 1))


class TestCoupledPruner(unittest.TestCase):
    def test_apply_and_revert_roundtrip(self):
        from transformersurgeon.compression import CoupledPruner

        module = LinearCompressed(INTER, HID, bias=False)
        keep = torch.zeros(INTER, dtype=torch.bool)
        keep[:6] = True  # keep 6 of 16 input columns

        CoupledPruner().apply(module, keep, hard=True)
        self.assertEqual(module.in_features, 6)
        self.assertEqual(module.weight.shape[1], 6)

        CoupledPruner().revert(module)
        self.assertEqual(module.in_features, INTER)
        self.assertEqual(module.weight.shape[1], INTER)

    def test_soft_apply_is_noop(self):
        from transformersurgeon.compression import CoupledPruner

        module = LinearCompressed(INTER, HID, bias=False)
        keep = torch.ones(INTER, dtype=torch.bool)
        CoupledPruner().apply(module, keep, hard=False)
        self.assertEqual(module.in_features, INTER)


class TestGrouping(unittest.TestCase):
    def test_create_and_delete_group(self):
        _, manager = _make_manager()
        g = manager.create_group(
            ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"]
        )
        self.assertEqual(g.name, "group1")
        self.assertEqual(len(g), 2)
        self.assertIn("group1", manager.groups)
        manager.delete_group("group1")
        self.assertNotIn("group1", manager.groups)
        # Scheme membership cleaned up.
        gate = manager._find_scheme("layers.0.mlp.gate_proj")
        self.assertEqual(gate.groups, {})

    def test_scheme_at_most_one_group(self):
        _, manager = _make_manager()
        manager.create_group(["layers.0.mlp.gate_proj"], name="group1")
        with self.assertRaises(ValueError):
            manager.create_group(["layers.0.mlp.gate_proj"], name="group2")

    def test_auto_groups_returns_dict(self):
        _, manager = _make_manager()
        created = manager.auto_groups()
        self.assertIsInstance(created, dict)
        # LAYERS per-block gate/up groups + 1 cross-block o_proj/down_proj group.
        self.assertEqual(len(created), LAYERS + 1)
        for name in created:
            self.assertTrue(name.startswith("group"))

    def test_auto_groups_cross_block_coupling(self):
        _, manager = _make_manager()
        created = manager.auto_groups()
        # Exactly one group spans all blocks (o_proj + down_proj from every block).
        cross = [paths for paths in created.values() if len(paths) == 2 * LAYERS]
        self.assertEqual(len(cross), 1)
        members = cross[0]
        self.assertIn("layers.0.self_attn.o_proj", members)
        self.assertIn("layers.1.mlp.down_proj", members)

    def test_shared_mask_identical_across_group(self):
        model, manager = _make_manager()
        manager.create_group(
            ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"], name="group1"
        )
        manager.set("structured_pruning", "share_mask", True, group="group1")
        manager.set("structured_pruning", "reduce_op", "add", group="group1")
        manager.set("structured_pruning", "ratio", 0.5, group="group1")
        manager.apply(hard=False)  # soft: mask buffers stay on the modules

        gate = model.layers[0].mlp.gate_proj
        up = model.layers[0].mlp.up_proj
        self.assertTrue(torch.equal(gate._buffers["weight_mask"], up._buffers["weight_mask"]))

    def test_shared_mask_hard_cascade_single_target(self):
        model, manager = _make_manager()
        x = torch.randn(2, 3, HID)
        manager.create_group(
            ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"], name="group1"
        )
        manager.set("structured_pruning", "share_mask", True, group="group1")
        manager.set("structured_pruning", "reduce_op", "add", group="group1")
        manager.set("structured_pruning", "ratio", 0.5, group="group1")
        manager.apply(hard=True)

        gate = model.layers[0].mlp.gate_proj
        up = model.layers[0].mlp.up_proj
        down = model.layers[0].mlp.down_proj
        keep = effective_out_features(INTER, 0.5, "layer")
        self.assertEqual(gate.out_features, keep)
        self.assertEqual(up.out_features, keep)
        # down_proj input pruned exactly once to the shared kept dim.
        self.assertEqual(down.in_features, keep)
        self.assertEqual(model(x).shape, (2, 3, HID))

    def test_reduce_op_none_raises_on_apply(self):
        _, manager = _make_manager()
        manager.create_group(
            ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"], name="group1"
        )
        manager.set("structured_pruning", "share_mask", True, group="group1")
        manager.set("structured_pruning", "ratio", 0.5, group="group1")
        with self.assertRaises(ValueError):
            manager.apply(hard=False)


class TestGroupOptionRules(unittest.TestCase):
    def test_group_option_without_group_raises(self):
        _, manager = _make_manager()
        with self.assertRaises(ValueError):
            manager.set("structured_pruning", "share_mask", True)

    def test_group_option_with_criteria_raises(self):
        _, manager = _make_manager()
        manager.create_group(["layers.0.mlp.gate_proj"], name="group1")
        with self.assertRaises(ValueError):
            manager.set(
                "structured_pruning", "share_mask", True,
                group="group1", criteria="mlp",
            )

    def test_enabling_share_mask_resets_regular_config(self):
        _, manager = _make_manager()
        manager.create_group(
            ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"], name="group1"
        )
        # Set a ratio first, then enabling share_mask should reset it.
        manager.set("structured_pruning", "ratio", 0.5, group="group1")
        manager.set("structured_pruning", "share_mask", True, group="group1")
        gate = manager._find_scheme("layers.0.mlp.gate_proj")
        self.assertEqual(gate.compressors["structured_pruning"].ratio, 0.0)
        self.assertTrue(gate.compressors["structured_pruning"].share_mask)


if __name__ == "__main__":
    unittest.main()
