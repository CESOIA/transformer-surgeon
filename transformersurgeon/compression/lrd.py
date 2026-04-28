import torch
from typing import Tuple, Union
from .abstract import Compressor

_DEFAULT_LRD_METHOD = "svd"
_SVD_LLM_V2_METHOD = "svd_llm_v2"
_LRD_METHODS = (_DEFAULT_LRD_METHOD, _SVD_LLM_V2_METHOD)


def _validate_rank(weight, rank: int) -> None:
    if rank >= min(weight.size()):
        # No decomposition possible, launch error
        raise ValueError(f"Rank {rank} must be less than the minimum dimension of the weight matrix ({weight.size(0), weight.size(1)}).")


def validate_lrd_method(method: str) -> None:
    if not isinstance(method, str):
        raise ValueError(f"LRD method must be a string, but got type {type(method)}.")

    if method not in _LRD_METHODS:
        raise ValueError(f"Unsupported LRD method '{method}'. Supported methods are: {_LRD_METHODS}.")


def _low_rank_svd(weight, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs low-rank decomposition on the given weight matrix using SVD.

    Args:
        weight (torch.Tensor): The weight matrix to be decomposed.
        rank (int): The target rank for the decomposition.

    Returns:
        torch.Tensor: The first matrix of the low-rank decomposition.
        torch.Tensor: The second matrix of the low-rank decomposition.
    """
    _validate_rank(weight, rank)
    
    # Perform SVD
    weight_f32 = weight.float() # Convert to float32 for SVD computation
    U, S, Vh = torch.linalg.svd(weight_f32, full_matrices=False)

    # Keep only the top 'rank' components
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    # Reconstruct the low-rank approximation
    US_r_out = (U_r * S_r.unsqueeze(0)).to(weight.dtype)
    V_r_out = Vh_r.to(weight.dtype)
    return US_r_out.contiguous(), V_r_out.contiguous()


def svd_llm_v2(
    weight,
    rank: int,
    covariance: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs activation-aware low-rank decomposition using the SVD-LLM v2
    transform.

    Args:
        weight (torch.Tensor): The weight matrix to be decomposed.
        rank (int): The target rank for the decomposition.
        covariance (torch.Tensor): Precomputed X.T @ X / num_tokens for this
            exact linear layer.
        eps (float): Minimum singular value used when inverting the covariance
            square root.

    Returns:
        torch.Tensor: The first matrix of the low-rank decomposition.
        torch.Tensor: The second matrix of the low-rank decomposition.
    """
    _validate_rank(weight, rank)

    if eps <= 0:
        raise ValueError(f"eps must be positive, but got {eps}.")

    in_features = weight.size(1)
    device = weight.device

    if covariance is None:
        raise ValueError(
            "The svd_llm_v2 LRD method requires a precomputed covariance tensor. "
            "Run manager.calibrate_lrd(...) before applying compression."
        )
    if covariance.shape != (in_features, in_features):
        raise ValueError(
            f"Covariance shape must be {(in_features, in_features)}, but got {tuple(covariance.shape)}."
        )
    covariance = covariance.to(device=device, dtype=torch.float32)

    weight_f32 = weight.float()
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5

    # Decompose C and clamp the singular values before taking inverse square roots.
    Uc, Sc, _ = torch.linalg.svd(covariance, full_matrices=False)
    Sc = torch.clamp_min(Sc, eps)
    sqrt_Sc = torch.sqrt(Sc)

    # Transform by C_sqrt = Uc @ diag(sqrt(Sc)) without materializing the diagonal.
    C_sqrt = Uc * sqrt_Sc.unsqueeze(0)
    W_tilde = weight_f32 @ C_sqrt

    U, S, Vh = torch.linalg.svd(W_tilde, full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    sqrt_S_r = torch.sqrt(S_r)

    # Map back with C_inv_sqrt = diag(1 / sqrt(Sc)) @ Uc.T.
    L = U_r * sqrt_S_r.unsqueeze(0)
    R = sqrt_S_r.unsqueeze(1) * ((Vh_r / sqrt_Sc.unsqueeze(0)) @ Uc.transpose(0, 1))

    return L.to(weight.dtype).contiguous(), R.to(weight.dtype).contiguous()

class LRDer(Compressor):
    def __init__(
        self,
        config,
        ):
        # Configuration
        self.config = config
        # Local temporary configuration
        self.rank = self.config["rank"]
        self.method = self.config.get("method", _DEFAULT_LRD_METHOD)
        self.eps = self.config.get("eps", 1e-6)
        self.covariance = self.config.get(
            "covariance",
            self.config.get("activation_covariance", None),
        )

    def set_covariance(self, covariance):
        self.covariance = covariance

    def clear_covariance(self):
        self.covariance = None
        for attr in ("_covariance_sum", "_covariance_tokens"):
            if hasattr(self, attr):
                delattr(self, attr)

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return # No compression needed based on the configuration

        # Extract temp configuration
        rank = self.rank
        method = self.method
        validate_lrd_method(method)
        eps = self.eps
        covariance = self.covariance
        # Apply temp configuration to module config
        self.config["rank"] = rank
        self.config["method"] = method
        self.config["eps"] = eps
    
        if rank:
            if not soft_applied:
                with torch.no_grad():
                    if method == _DEFAULT_LRD_METHOD:
                        US_r, V_r = _low_rank_svd(module.weight.detach(), rank)
                    elif method == _SVD_LLM_V2_METHOD:
                        US_r, V_r = svd_llm_v2(
                            module.weight.detach(),
                            rank,
                            covariance=covariance,
                            eps=eps,
                        )
                    else:
                        raise ValueError(f"Unsupported LRD method '{method}'.")
                    module.init_lrd(rank)
                    module.weight[:, :rank].copy_(US_r)
                    module.weight_2[:rank, :].copy_(V_r)

    def restore(self, module):
        if not self._to_compress():
            # Restore module configuration
            self.config["rank"] = "full"
            self.config["method"] = _DEFAULT_LRD_METHOD

        if not hasattr(module, 'weight_2'):
            raise AttributeError("Module does not have 'weight_2' attribute required for LRD restoration.")
        
        # Combine the low-rank components to restore the original weight matrix
        with torch.no_grad():
            restored_weight = module.weight.detach() @ module.weight_2.detach()
            module.cancel_lrd()
            module.weight.copy_(restored_weight)
        
    def _to_compress(self):
        # Check if LRD has to be applied based on the rank configuration
        return self.rank != "full"
    
    def __repr__(self):
        string = f"LRDer(rank={self.rank}, method='{self.method}')"
        return string

# Configuration validators
        
def validate_lrd_rank(rank: Union[int, str]) -> None:
    if rank is not None:
        if isinstance(rank, str):
            if rank != "full":
                raise ValueError(f"LRD rank must be 'full' or a positive integer, but got '{rank}'.")
        elif isinstance(rank, int):
            if rank <= 0:
                raise ValueError(f"LRD rank must be a positive integer, but got {rank}.")
        else:
            raise ValueError(f"LRD rank must be 'full' or a positive integer, but got type {type(rank)}.")

__all__ = [
    "LRDer",
    "svd_llm_v2",
    "validate_lrd_method",
    "validate_lrd_rank"
]