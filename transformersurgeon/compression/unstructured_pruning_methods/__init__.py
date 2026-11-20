from .magnitude import *
from .gradient import *
from .random import *
from .wanda import *

METHOD_FUNCTIONS = {
    "magnitude": mask_magnitude,
    "gradient": mask_gradient,
    "random": mask_random,
    "wanda": mask_wanda,
}

__all__ = ["METHOD_FUNCTIONS"]
