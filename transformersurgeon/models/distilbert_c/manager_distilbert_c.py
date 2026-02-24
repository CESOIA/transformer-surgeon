# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
from ...utils import CompressionSchemesManager
from .indexing_distilbert_c import DISTILBERT_C_INDEXING as INDEXING


class DistilBertCompressionSchemesManager(CompressionSchemesManager):
    """
    Manager for compression schemes specific to DistilBERT models.
    Refer to the base class `CompressionSchemesManager` for method details.
    """

    def __init__(self, model):
        super().__init__(model, INDEXING)


__all__ = ["DistilBertCompressionSchemesManager"]
