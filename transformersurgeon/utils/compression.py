import torch

class CompressionScheme:
    def __init__(self, path, pruning_ratio, lrd_rank, is_qkv=False):
        self.path = path
        self.pruning_ratio = pruning_ratio
        self.lrd_rank = lrd_rank
        self.is_qkv = is_qkv

    # WIP
    # def get_weights(self, model):
    #     """
    #     Retrieve the weights from the model based on the path.
    #     """
    #     return getattr(model, self.path).weight

    # def set_weights(self, model, weights):
    #     """
    #     Set the weights in the model based on the path.
    #     """
    #     getattr(model, self.path, weights)

__all__ = ["CompressionScheme"]