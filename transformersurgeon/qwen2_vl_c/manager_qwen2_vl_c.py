from ..utils import CompressionSchemesManager
from .indexing_qwen2_vl_c import QWEN2_VL_C_INDEXING as INDEXING

class Qwen2VLCompressionSchemesManager(CompressionSchemesManager):
    """
    Manager for compression schemes specific to Qwen2-VL models.
    Refer to the base class `CompressionSchemesManager` for method details.
    """
    
    def __init__(self, model, model_bis=None): # model_bis is for compatibility, not used
        if model_bis is not None: # for compatibility
            model_bis.config = model 
            model = model_bis
            
        super().__init__(model, INDEXING)

__all__ = ["Qwen2VLCompressionSchemesManager"]