"""utils/indexing.py: resolution of the generic layout keys of a model dictionary."""
import types

import pytest

from transformersurgeon.utils import (
    block_layer_paths,
    expand_grouped_paths,
    flatten_index_paths,
    get_block_config,
    num_blocks,
)

GROUPED = {"norm": [], "attn": ["q", "k"], "mlp": ["up"]}


def test_flatten_index_paths_grouped_and_flat():
    assert flatten_index_paths(GROUPED) == ["norm", "attn.q", "attn.k", "mlp.up"]
    assert flatten_index_paths(["a", "b"]) == ["a", "b"]


def test_flatten_index_paths_rejects_bad_shapes():
    with pytest.raises(TypeError):
        flatten_index_paths({"attn": "q"})
    with pytest.raises(TypeError):
        flatten_index_paths("attn.q")


def test_expand_grouped_paths_is_a_partition_of_the_flat_list():
    groups = expand_grouped_paths(GROUPED)
    assert groups == {"norm": ["norm"], "attn": ["attn.q", "attn.k"], "mlp": ["mlp.up"]}
    assert sum(groups.values(), []) == flatten_index_paths(GROUPED)
    assert expand_grouped_paths(["a"]) == {"": ["a"]}


def test_get_block_config_and_num_blocks():
    text = types.SimpleNamespace(layers=3)
    root = types.SimpleNamespace(layers=2, text_config=text)
    assert get_block_config(root, {"config_attr": ""}) is root
    assert get_block_config(root, {"config_attr": "text_config"}) is text
    assert num_blocks(root, {"config_attr": "text_config", "num_blocks_attr": "layers"}) == 3


def test_block_layer_paths_formats_the_template():
    indexing = {"path_template": "model.layers.{block_index}.{path}", "path_list": GROUPED}
    assert block_layer_paths(indexing, 1) == [
        "model.layers.1.norm", "model.layers.1.attn.q", "model.layers.1.attn.k", "model.layers.1.mlp.up"]
