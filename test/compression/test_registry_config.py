"""compression/registry.py, config.py, compressible_layers.py: the declarative side of compression."""
import pytest
import torch

from transformersurgeon.blocks import LinearCompressed
from transformersurgeon.compression import (
    APPLICATION_ORDER,
    COMPRESSION_REGISTRY,
    COMPRESSOR_DICT,
    GROUP_OPTIONS,
    init_compressed_config,
    replace_layers_upon_init,
)
from transformersurgeon.compression.abstract import Compressor


def _defaults(name):
    return {key: meta["default"] for key, meta in COMPRESSION_REGISTRY[name].items()}


def test_every_compressor_is_registered_ordered_and_configurable():
    assert set(COMPRESSOR_DICT) == set(COMPRESSION_REGISTRY) == set(APPLICATION_ORDER)
    for name, cls in COMPRESSOR_DICT.items():
        assert issubclass(cls, Compressor)
        compressor = cls(_defaults(name))
        for key, meta in COMPRESSION_REGISTRY[name].items():
            assert hasattr(compressor, key), (name, key)  # manager.set relies on setattr
            meta["validator"](meta["default"])          # every default is valid
        assert compressor.needs_calibration() == ()       # nothing to calibrate when inactive
    for name, keys in GROUP_OPTIONS.items():
        assert set(keys) <= set(COMPRESSION_REGISTRY[name])


@pytest.mark.parametrize("name, key, bad", [
    ("lrd", "rank", -3), ("lrd", "method", "qr"), ("quantization", "precision", 99),
    ("quantization", "granularity", "per_row"), ("structured_pruning", "ratio", 1.5),
    ("unstructured_pruning", "method", "lottery"), ("block_pruning", "ratio", 1.0),
])
def test_validators_reject_bad_values(name, key, bad):
    with pytest.raises((ValueError, TypeError)):
        COMPRESSION_REGISTRY[name][key]["validator"](bad)


def test_init_compressed_config_fills_defaults_for_every_layer():
    class Cfg:
        layers = 2
    cfg = Cfg()
    indexing = {"path_template": "blocks.{block_index}.{path}", "num_blocks_attr": "layers",
                "path_list": {"attn": ["q"], "norm": []}}
    init_compressed_config(cfg, indexing, {"blocks.0.attn.q": {"lrd": {"rank": 4}}})
    assert set(cfg.compression_config) == {"blocks.0.attn.q", "blocks.0.norm", "blocks.1.attn.q", "blocks.1.norm"}
    assert cfg.compression_config["blocks.0.attn.q"]["lrd"]["rank"] == 4
    assert cfg.compression_config["blocks.1.attn.q"]["lrd"]["rank"] == "full"
    with pytest.raises(ValueError):
        init_compressed_config(Cfg(), indexing, {"blocks.0.attn.q": {"lrd": {"rank": -1}}})


def test_replace_layers_upon_init_swaps_listed_linears():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(4, 4)
            self.other = torch.nn.Linear(4, 4)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

    class Cfg:
        layers = 2
        compression_config = {f"blocks.{i}.q": {"lrd": {"rank": 2}} for i in range(2)}

        def to_dict(self):
            return {"layers": self.layers, "compression_config": self.compression_config}

    model = Model()
    indexing = {"config_attr": "", "path_template": "blocks.{block_index}.{path}", "num_blocks_attr": "layers",
                "path_list": ["q"]}
    replace_layers_upon_init(model, indexing, Cfg())
    assert isinstance(model.blocks[1].q, LinearCompressed) and model.blocks[1].q.rank == 2
    assert type(model.blocks[1].other) is torch.nn.Linear
