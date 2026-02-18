from abc import ABC, abstractmethod

class Compressor(ABC):

    @abstractmethod
    def apply(self, module, hard=False, soft_applied=False):
        """
        Apply the compression to the given module. The behavior of the method can be influenced by the parameters:
        If soft_applied = True and hard = True, the method should apply hard compression on top of the already soft-compressed module.
        Hard = True means that the compression is final and effective (e.g., for pruning, the pruned weights are removed from the model architecture, not just zeroed out).
        """
        pass

    @abstractmethod
    def restore(self, module):
        """
        This method restores the topology of the module using its compressed version (e.g., for pruning, zeroed-out weights are kept zeroed-out but all information to retrieve the original weights is removed, while for LRD the two low-rank matrices are merged back into a single weight matrix).
        """
        pass

    @abstractmethod
    def _to_compress(self):
        """
        Check if compression has to be applied based on the compressor's configuration.
        This method is used to determine whether the compressor should be applied to a given module based on its configuration parameters (e.g., pruning ratio = 0.0 -> no compression needed).
        """
        pass
    
    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of the compressor with its configuration for better interpretability.

        Template example for the string representation:

        string = f"CompressorName(param1={self.param1}, param2={self.param2}, ...)"
        return string
        """
        pass

__all__ = ["Compressor"]