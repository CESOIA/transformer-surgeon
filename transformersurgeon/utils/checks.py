def get_validated_value(value, default, min_value=None, max_value=None):
    """
    Returns `value` if it is not None and valid.
    If `value` is None, returns `default`.
    If `value` is outside [min_value, max_value], raises ValueError.
    """
    if value is None:
        return default
    if min_value is not None and value < min_value:
        raise ValueError(f"Value ({value}) is less than minimum allowed ({min_value}).")
    if max_value is not None and value > max_value:
        raise ValueError(f"Value ({value}) is greater than maximum allowed ({max_value}).")
    return value

def get_validated_dict_value(dictionary, key, index, default, min_value=None, max_value=None):
    """
    Returns the value for `key` in `dictionary` if present and valid.
    If the key is missing or the value is None, returns `default`.
    If the value is outside [min_value, max_value], raises ValueError.
    """
    value = dictionary.get(key, None)
    if value is not None:
        value = value[index] if isinstance(value, list) else value
    value = get_validated_value(value, default, min_value, max_value)
    return value

__all__ = [
    "get_validated_value",
    "get_validated_dict_value",
]