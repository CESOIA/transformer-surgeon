"""export/operations.py: the block-type -> device-operation dictionary and how exporters use it."""
import dataclasses

import pytest

from transformersurgeon.blocks import BLOCK_CATALOG
from transformersurgeon.export import BACKEND_OPERATIONS, BACKENDS, default_for, options_for
from transformersurgeon.export.backends import (
    ONNXExportConfig,
    QNNExportConfig,
    TensorRTExportConfig,
    XNNPACKExportConfig,
)
from transformersurgeon.export.operations import (
    check_choice,
    check_config_choices,
    config_default,
    resolve_convert_options,
)

CATALOG_FAMILY = {"kv_cache": "kv_cache", "attention_kernel": "attention_kernel", "rmsnorm": "norm",
                  "mlp": "mlp", "rope_frequencies": "rope_frequencies"}


def test_every_entry_has_a_valid_default_for_every_backend():
    for block_type, spec in BACKEND_OPERATIONS.items():
        assert spec["description"]
        assert sum(k in spec for k in ("convert_option", "config_field", "model_key")) == 1, block_type
        for backend in BACKENDS:
            entry = spec["backends"][backend]
            if spec.get("model_key"):  # chosen by the model dictionary, not by the exporter
                assert entry["default"] is None, (block_type, backend)
            else:
                assert entry["default"] in entry["options"], (block_type, backend)
            assert all(isinstance(text, str) and text for text in entry["options"].values())


def test_convert_options_name_existing_block_atoms():
    for block_type, family in CATALOG_FAMILY.items():
        for backend in BACKENDS:
            assert set(options_for(backend, block_type)) <= set(BLOCK_CATALOG[family]), (block_type, backend)


def test_backend_config_defaults_come_from_the_dictionary():
    assert XNNPACKExportConfig(output_path="x").quant_path == config_default("xnnpack", "quant_path") == "pt2e"
    assert QNNExportConfig(output_path="x").quant_scope == default_for("qnn", "linear_quantization")
    assert QNNExportConfig(output_path="x").convert_linear_to_conv2d is False
    assert TensorRTExportConfig(output_path="x").int4_backend == "dequantize"
    assert ONNXExportConfig(output_path="x").backend == "onnx"


def test_check_config_choices_rejects_unlisted_options():
    check_config_choices("qnn", QNNExportConfig(output_path="x", quant_scope="all"))
    with pytest.raises(ValueError, match="not supported by the qnn backend"):
        check_config_choices("qnn", QNNExportConfig(output_path="x", quant_scope="everything"))
    with pytest.raises(ValueError, match="int4"):
        check_config_choices("tensorrt", dataclasses.replace(TensorRTExportConfig(output_path="x"),
                                                              int4_backend="int4_magic"))


@pytest.mark.parametrize("backend, cache, kernel", [
    ("xnnpack", "mutable", "manual"), ("qnn", "mutable", "manual"),
    ("onnx", "io_inplace", "manual"), ("tensorrt", "io_inplace", "manual"),
])
def test_resolve_convert_options_fills_backend_defaults(backend, cache, kernel):
    options = resolve_convert_options(backend)
    assert (options["cache_impl"], options["attn_impl"], options["rmsnorm_impl"]) == (cache, kernel, "rmsnorm_prescaled")


def test_resolve_convert_options_honours_aliases_and_compatibility():
    assert resolve_convert_options("xnnpack", {"use_sdpa": True})["attn_impl"] == "sdpa"
    assert resolve_convert_options("qnn", {"rmsnorm_prescale": False})["rmsnorm_impl"] == "rmsnorm"
    assert resolve_convert_options("tensorrt", {"cache_impl": "io_paged"})["attn_impl"] == "paged"
    with pytest.raises(ValueError, match="mutable"):
        resolve_convert_options("onnx", {"cache_impl": "mutable"})
    with pytest.raises(ValueError, match="custom_sdpa"):
        resolve_convert_options("qnn", {"attn_impl": "custom_sdpa"})
    with pytest.raises(ValueError, match="Unknown convert options"):
        resolve_convert_options("qnn", {"cache": "mutable"})
    with pytest.raises(ValueError):
        check_choice("onnx", "batch_dim", True)
