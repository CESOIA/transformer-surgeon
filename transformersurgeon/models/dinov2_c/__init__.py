"""dinov2_c: model dictionary and generated compressible classes."""
from . import indexing_dinov2_c, family_dinov2_c
from .indexing_dinov2_c import *
from .family_dinov2_c import *

__all__ = indexing_dinov2_c.__all__ + family_dinov2_c.__all__
