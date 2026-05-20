from .magnitude import *
from .gradient import *
from .random import *

METHOD_FUNCTIONS = {
    "magnitude": mask_magnitude,
    "gradient": mask_gradient,
    "random": mask_random,
}

__all__ = ["METHOD_FUNCTIONS"]
