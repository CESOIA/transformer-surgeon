"""Export-ready re-implementation of the transformer, built from selectable atoms.

Sub-packages (one atom per file):

- ``layers/``    compressible elementary layers (linear, embedding, convolution) and helpers;
- ``norm/``      normalisations;
- ``mlp/``       feed-forward blocks and activations;
- ``positions/`` rotary position embeddings;
- ``attention/`` attention modules, attention kernels, KV-cache and KV-precision atoms;
- ``stacks/``    complete decoder/encoder stacks and their configurations.

``catalog.py`` lists every selectable atom (``BLOCK_CATALOG``). Nothing here
depends on a specific model family: the model dictionaries in ``models/`` name
the atoms a model needs, and ``export/operations.py`` names the implementation
each backend prefers.
"""
from .layers import *
from .catalog import *
from .norm import *
from .mlp import *
from .positions import *
from .attention import (MHABase, MHACausal, MHAEncoder, ATTENTION_KERNELS, KV_CACHES, KV_PRECISIONS, CACHE_IMPLS,
                        ATTN_IMPLS)
from .stacks import *
