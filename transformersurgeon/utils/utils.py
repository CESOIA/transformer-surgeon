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
            raise ValueError(f"Module at path '{submodule_path}' not found in module {module}.")

    return tmp_module

__all__ = ['get_submodule']