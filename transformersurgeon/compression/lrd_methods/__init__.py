from .svd import *
from .svd_llm_v2 import *
from .aa_svd import *

METHOD_FUNCTIONS = {
    "svd": low_rank_svd,
    "svd-llm-v2": low_rank_svd_llm_v2,
    "aa-svd": low_rank_aa_svd,
}

# Methods whose components come sorted by decreasing importance: the first r components of a rank-R
# factorisation are the method's own rank-r solution (truncated SVD, also after whitening), so the
# factors can run at any rank <= R (runtime rank, blocks/layers/runtime_rank.py).
ORDERED_METHODS = ("svd", "svd-llm-v2")

__all__ = ["METHOD_FUNCTIONS", "ORDERED_METHODS"]