# Export Registry

`EXPORT_ROUTINES` maps backend names (`"xnnpack"`, `"qnn"`, `"tensorrt"`) to
their `export` function and `config_class`. To register a new backend, add an
entry here — see [Extending TransformerSurgeon → New export backend](../../concepts.md#new-export-backend).

::: transformersurgeon.export.registry
