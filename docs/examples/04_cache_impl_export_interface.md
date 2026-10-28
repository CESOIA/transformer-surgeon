# KV-cache interfaces of the exported model

Converts a causal language model under each KV-cache implementation (`mutable`, `io_scatter`, `io_concat`),
prints the input/output contract a runtime will see, and runs the same greedy generation loop on each, showing
how to drive every interface. The three are numerically equivalent; they differ only in who owns the cache:
the graph itself (`mutable`) or the caller, who passes the caches in and gets the updated ones back (`io_*`).
Which one each backend uses by default is in the [operations dictionary](../folders/export.md#the-operations-dictionary).

**Run it:**

```bash
python examples/04_cache_impl_export_interface.py
```

**Source** (`examples/04_cache_impl_export_interface.py`):

```python
--8<-- "examples/04_cache_impl_export_interface.py"
```
