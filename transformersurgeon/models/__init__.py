"""Model families: one folder per family, each holding two dictionaries.

- ``indexing_<name>.py``: the model dictionary (layout, compressible layers,
  couplings, calibration groups, export atoms);
- ``family_<name>.py``: the Hugging Face classes the family extends; the
  compressible classes are generated from it by ``factory.build_family``.

Adding a model means adding one such folder and listing it below.
"""

import warnings
from importlib import import_module

FAMILIES = [
    "qwen2_c",
    "qwen2_vl_c",
    "qwen2_5_vl_c",
    "vit_c",
    "bert_c",
    "modernbert_c",
    "distilbert_c",
    "llama_c",
]

__all__ = []
for _family in FAMILIES:
    try:
        _module = import_module(f"{__name__}.{_family}")
    except ImportError as exc:
        # The installed transformers version may lack this model family.
        warnings.warn(f"Model family {_family!r} is unavailable: {exc}")
        continue
    for _name in _module.__all__:
        globals()[_name] = getattr(_module, _name)
    __all__.extend(_module.__all__)

del _family, _module, _name, import_module
