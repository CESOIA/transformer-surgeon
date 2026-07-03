# export_to_backend

Backend-agnostic export entry point. Resolves any supported model input into
`{embedding, decoder, final_layer, config}` and delegates to the backend named
in `config.backend` via `EXPORT_ROUTINES` (see [Registry](registry.md)).

`export_to_executorch(...)` is a deprecated alias — use `export_to_backend`.

::: transformersurgeon.export.export
