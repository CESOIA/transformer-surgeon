from ..utils import CompressionSchemesManager
from .indexing_qwen2_5_vl_c import QWEN2_5_VL_C_INDEXING as INDEXING

class Qwen2_5_VLCompressionSchemesManager(CompressionSchemesManager):
    """
    Backward compatible CompressionSchemesManager for Qwen2-VL-C.
    This class is deprecated. Use create_compression_manager() instead.
    """
    
    def __init__(self, config, model):
        super().__init__(config, model, INDEXING)

__all__ = ["Qwen2_5_VLCompressionSchemesManager"]