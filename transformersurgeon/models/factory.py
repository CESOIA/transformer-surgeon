"""Build the compressible classes of a model family from its family dictionary.

Every model folder holds two dictionaries:

- the *model dictionary* (``indexing_<name>.py``): layout of the model, layers to
  compress, pruning couplings, calibration groups and the atoms its export
  needs;
- the *family dictionary* (``family_<name>.py``): which Hugging Face classes the
  family extends and the names of the generated classes.

``build_family(family)`` generates, for the family dictionary:

- one configuration class per component with its own configuration
  (``sub_configs``), plus the root configuration class. Each one fills the
  per-layer ``compression_config`` with defaults when it is created;
- the model class: the Hugging Face model whose linear/embedding/convolution
  layers listed in the model dictionary are replaced by compressible ones;
- the compression-manager class bound to the model dictionary.

Family dictionary keys:

``indexing``          the model dictionary.
``hf_config``         dotted path of the Hugging Face configuration class.
``hf_model``          dotted path of the Hugging Face model class.
``sub_configs``       ``{config attribute: dotted path of the HF sub-configuration}``
                      for components whose ``config_attr`` is not empty.
``class_names``       ``{"config", "model", "manager"}`` names of the generated
                      classes, plus one ``{config attribute}`` name per sub-configuration.
``auto_class``        Hugging Face auto class recorded when the model is saved
                      with remote code (``hf/hf_export.py``).
``checkpoint_key_mapping``  optional function ``(checkpoint keys) -> {old: new}``
                      used to load checkpoints written with an older layer naming.
"""

import inspect
from importlib import import_module

from ..compression import CompressionSchemesManager, init_compressed_config, replace_layers_upon_init


def _import(dotted_path):
    module_path, _, name = dotted_path.rpartition(".")
    return getattr(import_module(module_path), name)


def _components_with_config(indexing, config_attr):
    """Model-dictionary components whose configuration lives at ``config_attr``."""
    return [comp for comp in indexing.values() if comp.get("config_attr", "") == config_attr]


def _make_config_class(name, base, components, module, sub_configs=None):
    """Configuration subclass that fills ``compression_config`` for ``components``."""
    def __init__(self, **kwargs):
        compression_config = kwargs.pop("compression_config", None) or {}
        base.__init__(self, **kwargs)
        for indexing in components:
            init_compressed_config(config_instance=self, indexing=indexing,
                                   compression_config=compression_config)

    # Hugging Face inspects the constructor signature of (sub-)configurations.
    __init__.__signature__ = inspect.signature(base.__init__)
    namespace = {"__init__": __init__, "__module__": module}
    if sub_configs:
        namespace["sub_configs"] = {**getattr(base, "sub_configs", {}), **sub_configs}
    return type(name, (base,), namespace)


def _make_model_class(name, base, config_class, indexing, auto_class, key_mapping_fn, module):
    def __init__(self, config):
        base.__init__(self, config)
        for component in indexing.values():
            replace_layers_upon_init(self, component, config)

    namespace = {"__init__": __init__, "__module__": module, "config_class": config_class,
                 "indexing": indexing, "auto_class": auto_class}

    if key_mapping_fn is not None:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            # Checkpoints written with an older layer naming: compute the key mapping
            # from the checkpoint's own keys (best effort; loading proceeds without it).
            if kwargs.get("key_mapping") is None:
                try:
                    mapping = key_mapping_fn(pretrained_model_name_or_path, **{
                        k: kwargs.get(k) for k in ("revision", "subfolder", "cache_dir", "token")})
                except Exception:
                    mapping = None
                if mapping:
                    kwargs["key_mapping"] = mapping
            return super(model_cls, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        namespace["from_pretrained"] = from_pretrained

    model_cls = type(name, (base,), namespace)
    return model_cls


def _make_manager_class(name, indexing, module):
    def __init__(self, model):
        CompressionSchemesManager.__init__(self, model, indexing)

    return type(name, (CompressionSchemesManager,), {"__init__": __init__, "__module__": module})


def build_family(family, module):
    """Generate the classes of ``family`` and return ``{class name: class}``.

    ``module`` is the ``__name__`` of the defining module, recorded as the
    classes' ``__module__`` so they pickle and serialise under their own names.
    """
    indexing = family["indexing"]
    names = family["class_names"]
    classes = {}

    sub_configs = {}
    for config_attr, hf_path in family.get("sub_configs", {}).items():
        cls = _make_config_class(names[config_attr], _import(hf_path),
                                 _components_with_config(indexing, config_attr), module)
        sub_configs[config_attr] = classes[cls.__name__] = cls

    root_config = _make_config_class(names["config"], _import(family["hf_config"]),
                                     _components_with_config(indexing, ""), module, sub_configs)
    classes[root_config.__name__] = root_config

    model_cls = _make_model_class(names["model"], _import(family["hf_model"]), root_config, indexing,
                                  family.get("auto_class"), family.get("checkpoint_key_mapping"), module)
    classes[model_cls.__name__] = model_cls
    manager_cls = _make_manager_class(names["manager"], indexing, module)
    classes[manager_cls.__name__] = manager_cls
    return classes


__all__ = ["build_family"]
