import torch
from typing import Union

from .abstract import Compressor
from .lrd_methods import METHOD_FUNCTIONS


VALID_METHODS = ["svd", "svd-llm-v2", "aa-svd"]
CALIBRATED_METHOD_DICT = {
    "svd-llm-v2": ("covariance",),
    "aa-svd": ("cross_covariance", "shifted_covariance"),
}


class LRDer(Compressor):
    def __init__(self, config):
        self.config = config
        self.rank = self.config["rank"]
        self.method = self.config.get("method", "svd")
        self.eps = self.config.get("eps", 1e-6)
        self.calibration_store = None

    def set_calibration_store(self, calibration_data):
        self.calibration_store = calibration_data

    def needs_calibration(self):
        if not self._to_compress():
            return ()
        return CALIBRATED_METHOD_DICT.get(self.method, ())

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return

        rank = self.rank
        method = self.method
        eps = self.eps
        
        validate_lrd_rank(rank)
        validate_lrd_method(method)
        validate_lrd_eps(eps)

        self.config["rank"] = rank
        self.config["method"] = method
        self.config["eps"] = eps

        if rank and not soft_applied:
            with torch.no_grad():

                # To maintain flexibility for future methods, we handle methods with different input requirements separately.

                if method == "svd":
                    US_r, V_r = METHOD_FUNCTIONS[method](
                        module.weight.detach(),
                        rank
                    )
                    
                elif method == "svd-llm-v2":
                    covariance = self.calibration_store["covariance"]
                    US_r, V_r = METHOD_FUNCTIONS[method](
                        module.weight.detach(),
                        rank,
                        covariance=covariance,
                        eps=eps,
                    )

                elif method == "aa-svd":
                    cross_covariance = self.calibration_store["cross_covariance"]
                    shifted_covariance = self.calibration_store["shifted_covariance"]
                    US_r, V_r = METHOD_FUNCTIONS[method](
                        module.weight.detach(),
                        rank,
                        cross_covariance=cross_covariance,
                        shifted_covariance=shifted_covariance,
                        eps=eps,
                    )

                else:
                    raise ValueError(f"Unsupported LRD method '{method}'.")

                # Apply the low-rank decomposition to the module
                module.init_lrd(rank)
                module.weight[:, :rank].copy_(US_r)
                module.weight_2[:rank, :].copy_(V_r)

        if hard:
            # The two-matmul topology (weight @ weight_2) is already the efficient
            # deployed form. Soft and hard LRD are structurally identical — this
            # no-op is intentional.
            pass

    def restore(self, module):
        if module.rank == "full":
            return  # No need to restore if the module is already full rank

        if not self._to_compress():
            self.config["rank"] = "full"

        if not hasattr(module, "weight_2"):
            raise AttributeError("Module does not have 'weight_2' attribute required for LRD restoration.")

        with torch.no_grad():
            restored_weight = module.weight.detach() @ module.weight_2.detach()
            module.cancel_lrd()
            module.weight.copy_(restored_weight)

    def _to_compress(self):
        return self.rank != "full"

    def __repr__(self):
        return f"LRDer(rank={self.rank}, method='{self.method}')"


### CONFIGURATION VALIDATORS ###

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


def validate_lrd_method(method: str) -> None:
    if not isinstance(method, str):
        raise ValueError(f"LRD method must be a string, but got type {type(method)}.")

    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported LRD method '{method}'. Supported methods are: {VALID_METHODS}.")


def validate_lrd_eps(eps: float) -> None:
    if not isinstance(eps, (int, float)):
        raise ValueError(f"LRD eps must be numeric, but got type {type(eps)}.")
    if eps <= 0:
        raise ValueError(f"LRD eps must be positive, but got {eps}.")


__all__ = [
    "LRDer",
    "validate_lrd_method",
    "validate_lrd_rank",
    "validate_lrd_eps",
]
