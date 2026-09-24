# export_to_backend

Backend-agnostic export entry point. Resolves any supported model input into
`{embedding, decoder, final_layer, config}` and delegates to the backend named
in `config.backend` via `EXPORT_ROUTINES` (see [Registry](registry.md)).

::: transformersurgeon.export.export
