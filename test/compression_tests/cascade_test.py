import unittest
import importlib.util
import types
import sys
from pathlib import Path


def _load_cascade_module():
    repo_root = Path(__file__).resolve().parents[2]
    cascade_path = repo_root / "transformersurgeon" / "utils" / "cascade.py"

    # Build lightweight package stubs so relative imports in cascade.py resolve
    # without importing the full transformersurgeon package tree.
    pkg_ts = types.ModuleType("transformersurgeon")
    pkg_ts.__path__ = [str(repo_root / "transformersurgeon")]

    pkg_utils = types.ModuleType("transformersurgeon.utils")
    pkg_utils.__path__ = [str(repo_root / "transformersurgeon" / "utils")]

    pkg_calib = types.ModuleType("transformersurgeon.calibration")
    pkg_calib.__path__ = [str(repo_root / "transformersurgeon" / "calibration")]
    pkg_calib.run_compression_calibration = lambda *args, **kwargs: None

    pkg_calib_raw = types.ModuleType("transformersurgeon.calibration.raw_data")
    pkg_calib_raw.normalize_calibration_batch = lambda batch, batch_id: (tuple(), dict(batch), None)

    pkg_utils_utils = types.ModuleType("transformersurgeon.utils.utils")
    pkg_utils_utils.get_submodule = lambda module, path: module
    pkg_utils_utils.infer_model_device = lambda model: "cpu"
    pkg_utils_utils.move_to_device = lambda value, device: value

    sys.modules["transformersurgeon"] = pkg_ts
    sys.modules["transformersurgeon.utils"] = pkg_utils
    sys.modules["transformersurgeon.calibration"] = pkg_calib
    sys.modules["transformersurgeon.calibration.raw_data"] = pkg_calib_raw
    sys.modules["transformersurgeon.utils.utils"] = pkg_utils_utils

    spec = importlib.util.spec_from_file_location("transformersurgeon.utils.cascade", cascade_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_cascade = _load_cascade_module()
_build_layer_stages_for_block = _cascade._build_layer_stages_for_block
_extract_path_entries = _cascade._extract_path_entries


class _FakeScheme:
    def __init__(self, name, block_id):
        self.name = name
        self.block_id = block_id


class _FakeManager:
    def __init__(self, calibration_groups=None, calibration_no_data_dependency=False):
        self.calibration_groups = calibration_groups or []
        self.calibration_no_data_dependency = calibration_no_data_dependency


def _build_block_scheme_dict(path_entries, block_id):
    schemes = {}
    for _, layers in path_entries.items():
        for layer_path in layers:
            # Only .name and .block_id are needed by _build_layer_stages_for_block.
            schemes[layer_path] = _FakeScheme(name=layer_path, block_id=block_id)
    return schemes


class CascadeStageBuilderTests(unittest.TestCase):
    def test_layer_level_ordering_and_ungrouped_singletons(self):
        path_list = {
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
            "input_layernorm": [],
        }
        path_entries, ordered_subblocks = _extract_path_entries(path_list)
        block_scheme_dict = _build_block_scheme_dict(path_entries, block_id=0)
        selected_set = set(block_scheme_dict.values())

        manager = _FakeManager(
            calibration_groups=[
                ["q_proj", "k_proj"],
                ["up_proj", "down_proj"],
            ]
        )

        stages = _build_layer_stages_for_block(
            manager=manager,
            block_id=0,
            selected_set=selected_set,
            block_scheme_dict=block_scheme_dict,
            path_entries=path_entries,
            ordered_subblocks=ordered_subblocks,
            block_name="text",
        )

        self.assertEqual(
            stages,
            [
                ["attn.q_proj", "attn.k_proj"],
                ["mlp.up_proj", "mlp.down_proj"],
                ["attn.v_proj"],
                ["attn.o_proj"],
                ["mlp.gate_proj"],
                ["input_layernorm"],
            ],
        )

    def test_non_consecutive_group_raises(self):
        path_list = {
            "attn": ["q_proj", "k_proj", "v_proj"],
            "mlp": ["up_proj"],
        }
        path_entries, ordered_subblocks = _extract_path_entries(path_list)
        block_scheme_dict = _build_block_scheme_dict(path_entries, block_id=0)
        selected_set = set(block_scheme_dict.values())

        manager = _FakeManager(calibration_groups=[["q_proj", "v_proj"]])

        with self.assertRaises(ValueError):
            _build_layer_stages_for_block(
                manager=manager,
                block_id=0,
                selected_set=selected_set,
                block_scheme_dict=block_scheme_dict,
                path_entries=path_entries,
                ordered_subblocks=ordered_subblocks,
                block_name="text",
            )

    def test_empty_subblock_list_is_treated_as_single_layer(self):
        path_list = {
            "self_attn": ["q_proj"],
            "input_layernorm": [],
        }
        path_entries, ordered_subblocks = _extract_path_entries(path_list)

        self.assertEqual(path_entries["input_layernorm"], ["input_layernorm"])
        self.assertEqual(ordered_subblocks, ["self_attn", "input_layernorm"])


if __name__ == "__main__":
    unittest.main()
