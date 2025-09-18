from ..utils import CompressionSchemesManager
from .indexing_qwen2_vl_c import QWEN2_VL_C_INDEXING

class Qwen2VLCompressionSchemesManager(CompressionSchemesManager):
    """
    Manager for compression schemes specific to Qwen2-VL models.
    Refer to the base class `CompressionSchemesManager` for method details.
    """
    
    def __init__(self, config, model):
        super().__init__(config, model, QWEN2_VL_C_INDEXING)

__all__ = ["Qwen2VLCompressionSchemesManager"]