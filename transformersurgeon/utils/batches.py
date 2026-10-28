"""Normalise the batches yielded by a data loader into model call arguments."""

from collections.abc import Mapping

_ARGS, _KWARGS, _LABEL = "__ts_args__", "__ts_kwargs__", "__ts_label__"


def pack_batch(args=(), kwargs=None, label=None):
    """Pack already-split model arguments so :func:`normalize_batch` returns them unchanged."""
    return {_ARGS: args, _KWARGS: kwargs or {}, _LABEL: label}


def normalize_batch(batch, batch_id: int):
    """Split one data-loader batch into ``(model_args, model_kwargs, label)``.

    Accepted forms: a mapping (keyword arguments), a ``(mapping, label)`` or
    ``(label, mapping)`` pair, a tuple/list of positional arguments, a single
    object (one positional argument), or a batch already packed by
    :func:`pack_batch`.
    """
    # Pre-split batch produced by pack_batch().
    # Expected shape:
    # {
    #   "__ts_args__": tuple | list | single positional arg,
    #   "__ts_kwargs__": mapping of keyword args,
    #   "__ts_label__": optional target/label
    # }
    if isinstance(batch, Mapping) and "__ts_args__" in batch:
        # Read reserved pre-normalized fields with safe defaults.
        model_args = batch.get("__ts_args__", tuple())
        model_kwargs = batch.get("__ts_kwargs__", {})
        label = batch.get("__ts_label__", None)

        # Normalize positional args to tuple for a stable downstream call contract.
        if not isinstance(model_args, tuple):
            if isinstance(model_args, list):
                model_args = tuple(model_args)
            else:
                # Single positional value -> one-element args tuple.
                model_args = (model_args,)

        # kwargs must always be mapping-like (dict-compatible for **kwargs usage).
        if not isinstance(model_kwargs, Mapping):
            raise TypeError(
                "Packed batch '__ts_kwargs__' must be a mapping. "
                f"Got {type(model_kwargs)} at batch index {batch_id}."
            )

        return model_args, dict(model_kwargs), label

    # Mapping batch => keyword-only model call.
    if isinstance(batch, Mapping):
        return tuple(), dict(batch), None

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError(f"Batch is empty at batch index {batch_id}.")

        if len(batch) == 1:
            item = batch[0]
            # Preserve mapping semantics even when wrapped in a single-element tuple/list.
            if isinstance(item, Mapping):
                return tuple(), dict(item), None
            return (item,), {}, None

        # Canonical (data_mapping, target) form.
        if len(batch) == 2 and isinstance(batch[0], Mapping):
            return tuple(), dict(batch[0]), batch[1]

        # Also support swapped form (target, data_mapping).
        if len(batch) == 2 and isinstance(batch[1], Mapping):
            return tuple(), dict(batch[1]), batch[0]

        # Fallback: treat as pure positional model args.
        return tuple(batch), {}, None

    # Single object/tensor batch => one positional argument.
    return (batch,), {}, None


__all__ = ["pack_batch", "normalize_batch"]
