from .vanilla import *
from .act_maxmin import *
from .gptq import *
from .hqq import *

METHOD_FUNCTIONS = {
    "vanilla": quantize_vanilla,
    "gptq": quantize_gptq,
    "hqq": quantize_hqq,
}

ACT_METHOD_FUNCTIONS = {
    "maxmin": compute_activation_scale_zp_maxmin,
}

__all__ = ["METHOD_FUNCTIONS", "ACT_METHOD_FUNCTIONS"]