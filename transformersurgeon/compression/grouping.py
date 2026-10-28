"""
grouping.py

Model- and compression-agnostic grouping primitive. A :class:`SchemeGroup`
bundles several ``CompressionScheme`` objects that need to be compressed together
(e.g. layers that must share a pruning mask, or whose scores are reduced jointly).

The group holds **pointers** to the existing schemes (never copies) plus a plain
``properties`` dict used by compressors to share transient state across the group's
layers — for instance a single computed pruning mask that every member reuses. No
torch-model or algorithm specifics live here; the group only stores references and
a bag of properties.
"""

from typing import Any, Dict, List


class SchemeGroup:
    """A named group of compression schemes with shared, group-level properties.

    Args:
        name: Unique group name (e.g. ``"group1"``).
        schemes: Optional initial list of ``CompressionScheme`` pointers.

    Attributes:
        name: The group name.
        schemes: List of member scheme pointers (order preserved).
        properties: Free-form dict of shared state used by compressors (e.g. a
            shared pruning mask, reduce operator, transient ready flags).
    """

    def __init__(self, name: str, schemes: List[Any] = None):
        self.name = name
        self.schemes: List[Any] = []
        self.properties: Dict[str, Any] = {}
        for scheme in (schemes or []):
            self.add(scheme)

    def add(self, scheme) -> None:
        """Add a scheme pointer to the group (idempotent on identity)."""
        if scheme not in self.schemes:
            self.schemes.append(scheme)

    def remove(self, scheme) -> None:
        """Remove a scheme pointer from the group if present."""
        if scheme in self.schemes:
            self.schemes.remove(scheme)

    def set_property(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def reset(self) -> None:
        """Clear all transient shared properties (e.g. a stored shared mask)."""
        self.properties.clear()

    def __iter__(self):
        return iter(self.schemes)

    def __contains__(self, scheme) -> bool:
        return scheme in self.schemes

    def __len__(self) -> int:
        return len(self.schemes)

    def __repr__(self) -> str:
        paths = [getattr(s, "path", repr(s)) for s in self.schemes]
        return f"SchemeGroup(name={self.name!r}, schemes={paths})"


__all__ = ["SchemeGroup"]
