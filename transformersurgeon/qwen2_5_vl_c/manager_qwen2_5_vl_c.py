from ..utils import CompressionSchemesManager
from .indexing_qwen2_5_vl_c import QWEN2_5_VL_C_INDEXING as INDEXING

class Qwen2_5_VLCompressionSchemesManager(CompressionSchemesManager):
    """
    Manager for compression schemes specific to Qwen2.5-VL models.
    Refer to the base class `CompressionSchemesManager` for method details.
    """
    
    def __init__(self, config, model):
        super().__init__(config, model, INDEXING)

__all__ = ["Qwen2_5_VLCompressionSchemesManager"]