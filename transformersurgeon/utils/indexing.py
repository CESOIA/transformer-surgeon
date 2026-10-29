"""Helpers that read the generic layout keys of a model dictionary.

A model dictionary (see ``transformersurgeon/models``) describes where the
repeated blocks of a model live (``path_template``, ``num_blocks_attr``) and which
layers each block contains (``path_list``). These helpers only resolve that
layout into module paths; they carry no knowledge of any specific model.
"""


def flatten_index_paths(path_list):
    """Normalize indexing path definitions to a flat list of dotted paths."""
    if isinstance(path_list, dict):
        flattened_paths = []
        for parent_path, child_paths in path_list.items():
            if not isinstance(child_paths, (list, tuple)):
                raise TypeError(
                    "Grouped path_list values must be a list/tuple of sublayers. "
                    f"Found {type(child_paths).__name__} for key '{parent_path}'."
                )

            if len(child_paths) == 0:
                flattened_paths.append(parent_path)
                continue

            for child_path in child_paths:
                child_path = str(child_path)
                if parent_path:
                    flattened_paths.append(f"{parent_path}.{child_path}")
                else:
                    flattened_paths.append(child_path)

        return flattened_paths

    if isinstance(path_list, (list, tuple)):
        return list(path_list)

    raise TypeError(
        "Indexing 'path_list' must be a list/tuple or grouped dict, "
        f"got {type(path_list).__name__}."
    )


def expand_grouped_paths(path_list):
    """Return ``{group: [dotted paths]}`` for a grouped ``path_list``.

    A grouped ``path_list`` maps a sub-module name to the layer names it contains
    (``{"self_attn": ["q_proj", "k_proj"]}``). A flat list becomes a single group
    named ``""``. The concatenation of all values equals ``flatten_index_paths``.
    """
    if isinstance(path_list, dict):
        return {
            parent: flatten_index_paths({parent: children})
            for parent, children in path_list.items()
        }
    return {"": flatten_index_paths(path_list)}


def get_block_config(config, indexing):
    """Return the configuration object of the component described by ``indexing``.

    ``indexing["config_attr"]`` names the attribute of ``config`` that holds the
    component's own configuration; an empty string means ``config`` itself.
    """
    config_attr = indexing.get("config_attr", "")
    return config if config_attr == "" else getattr(config, config_attr)


def num_blocks(config, indexing):
    """Number of repeated blocks of the component described by ``indexing``."""
    return getattr(get_block_config(config, indexing), indexing["num_blocks_attr"])


def block_layer_paths(indexing, block_index):
    """Full module paths of every layer listed in ``path_list`` for one block."""
    template = indexing["path_template"]
    return [
        template.format(block_index=block_index, path=path)
        for path in flatten_index_paths(indexing["path_list"])
    ]


__all__ = [
    "flatten_index_paths",
    "expand_grouped_paths",
    "get_block_config",
    "num_blocks",
    "block_layer_paths",
]
