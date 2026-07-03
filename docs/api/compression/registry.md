# Compression Registry

`COMPRESSOR_DICT` maps compression method names to their `Compressor` classes.
`COMPRESSION_REGISTRY` defines the parameter schema (defaults and validators)
for each method.

To register a new compressor, add entries to both dicts.

::: transformersurgeon.compression.registry
