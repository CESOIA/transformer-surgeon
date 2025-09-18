def get_validated_value(value, default, min_value=None, max_value=None):
    """
    Returns `value` if it is not None and valid.
    If `value` is None, returns `default`.
    If `value` is outside [min_value, max_value], raises ValueError.

    Args:
        value: The value to validate.
        default: The default value to return if `value` is None.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).

    Returns:
        The validated value or the default.
    """
    
    if value is None:
        return default
    if type(value) is str:
        return value  # Strings are always valid
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

    Args:
        dictionary (Dict): The dictionary to get the value from.
        key (str): The key to look for in the dictionary.
        index (int): The index to use if the value is a list.
        default: The default value to return if the key is missing or value is None.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).
        
    Returns:
        The validated value or the default.
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