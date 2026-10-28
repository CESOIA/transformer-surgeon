"""Factories of the option validators used by ``registry.py``.

Every compression option has a validator ``f(value) -> None`` that raises
``ValueError`` on an invalid value. Most of them follow a few shapes (value in a
set, number in a range, positive integer, boolean); these factories build them
with a uniform message ``"<name> must be <rule>, but got <value>."``.
"""
from collections.abc import Callable, Iterable


def _fail(name: str, rule: str, value) -> None:
    raise ValueError(f"{name} must be {rule}, but got {value!r}.")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def one_of(name: str, choices: Iterable, *, allow_none: bool = False) -> Callable:
    """Value in ``choices`` (or ``None`` when allowed)."""
    choices = list(choices)

    def validate(value) -> None:
        if (value is None and allow_none) or value in choices:
            return
        _fail(name, f"one of {choices}", value)

    return validate


def in_range(name: str, low: float, high: float, *, include_high: bool = True, allow_none: bool = True) -> Callable:
    """Number in ``[low, high]`` (``[low, high)`` with ``include_high=False``)."""
    rule = f"in [{low}, {high}{']' if include_high else ')'}"

    def validate(value) -> None:
        if value is None and allow_none:
            return
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not (
                low <= value and (value <= high if include_high else value < high)):
            _fail(name, rule, value)

    return validate


def positive_number(name: str) -> Callable:
    """Strictly positive int or float."""

    def validate(value) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            _fail(name, "a positive number", value)

    return validate


def positive_int(name: str, *, also: Iterable = (), allow_none: bool = False,
                 low: int = 1, high: int | None = None) -> Callable:
    """Integer in ``[low, high]`` (``high=None``: unbounded), or one of the ``also`` values."""
    also = list(also)
    bounds = f"an integer in [{low}, {high}]" if high is not None else (
        "a positive integer" if low == 1 else f"an integer >= {low}")
    rule = bounds + (f" or one of {also}" if also else "")

    def validate(value) -> None:
        if (value is None and allow_none) or (not _is_int(value) and value in also):
            return
        if not _is_int(value) or value < low or (high is not None and value > high):
            _fail(name, rule, value)

    return validate


def boolean(name: str) -> Callable:
    """``True`` or ``False``."""

    def validate(value) -> None:
        if not isinstance(value, bool):
            _fail(name, "a boolean", value)

    return validate


__all__ = ["one_of", "in_range", "positive_number", "positive_int", "boolean"]
