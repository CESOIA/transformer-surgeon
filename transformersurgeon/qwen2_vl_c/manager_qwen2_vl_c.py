from ..utils import CompressionSchemesManager
from .manager_config import QWEN2_VL_C_CONFIG

class Qwen2VLCompressionSchemesManager(CompressionSchemesManager):
    """
    Backward compatible CompressionSchemesManager for Qwen2-VL-C.
    This class is deprecated. Use create_compression_manager() instead.
    """
    
    def __init__(self, config, model):
        super().__init__(config, model, QWEN2_VL_C_CONFIG)

__all__ = ["Qwen2VLCompressionSchemesManager"]