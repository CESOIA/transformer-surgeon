"""Access and replace submodules of a ``torch.nn.Module`` by dotted path."""

import torch


def get_submodule(module, submodule_path):
    """
    Returns the submodule of a given module based on the dot-separated path.

    Args:
        module: The parent module from which to retrieve the submodule.

    Returns:
        The submodule located at the specified path.
    """
    split_path = submodule_path.split('.')
    # Traverse the module iteratively to find the submodule
    tmp_module = module
    for path_piece in split_path:
        tmp_module = getattr(tmp_module, path_piece, None)

        if tmp_module is None:
            raise ValueError(f"Module at path '{submodule_path}' not found in module {module.__class__.__name__}.")

    return tmp_module


def get_parent_module(module, submodule_path):
    """Return ``(parent, attribute_name)`` for the submodule at ``submodule_path``.

    ``"a.b.c"`` resolves to ``(module.a.b, "c")``; a path without dots resolves to
    ``(module, path)``.
    """
    parent_path, _, name = submodule_path.rpartition(".")
    parent = get_submodule(module, parent_path) if parent_path else module
    return parent, name


def set_submodule(module, submodule_path, new_submodule):
    """Replace the submodule at ``submodule_path`` with ``new_submodule``."""
    parent, name = get_parent_module(module, submodule_path)
    setattr(parent, name, new_submodule)


def replace_parameter(module, name, tensor, requires_grad=None):
    """Replace parameter ``name`` of ``module`` with a new ``nn.Parameter`` holding ``tensor``.

    Used when a layer changes shape (for example after removing rows or columns):
    the old parameter object cannot be resized in place. ``requires_grad`` defaults
    to the old parameter's flag.
    """
    old = getattr(module, name)
    if requires_grad is None:
        requires_grad = getattr(old, "requires_grad", True)
    setattr(module, name, torch.nn.Parameter(tensor, requires_grad=requires_grad))


__all__ = ["get_submodule", "get_parent_module", "set_submodule", "replace_parameter"]
