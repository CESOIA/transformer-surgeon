from ..utils import CompressionSchemesManager
from .indexing_qwen2_vl_c import QWEN2_VL_C_INDEXING

class Qwen2VLCompressionSchemesManager(CompressionSchemesManager):
    """
    Backward compatible CompressionSchemesManager for Qwen2-VL-C.
    This class is deprecated. Use create_compression_manager() instead.
    """
    
    def __init__(self, config, model):
        super().__init__(config, model, QWEN2_VL_C_INDEXING)

__all__ = ["Qwen2VLCompressionSchemesManager"]