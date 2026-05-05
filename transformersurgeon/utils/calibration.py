"""
calibration.py

Provides LRDCalibration, an independent class for calibrating LRD compressors.
Can be used standalone with any model or integrated with CompressionSchemesManager.
"""

from collections.abc import Mapping
from typing import Optional, Union

import torch


class LRDCalibration:
    """
    Independent calibration helper for LRD (Low-Rank Decomposition) compressors.
    
    This class handles activation covariance collection and computation for weighted SVD (wsvd).
    It is model-agnostic and can function independently or be embedded in a compression manager.
    
    Internal workflow:
    1. Attach forward hooks to target modules to collect activations.
    2. Run batches through the model to accumulate covariance matrices.
    3. Compute the final covariance from accumulated statistics.
    4. Store computed covariance in each compressor for use during decomposition.
    """
    def __init__(self, model, calibration_data=None):
        self.model = model
        self.calibration_data = calibration_data

    def set_calibration_data(self, calibration_data):
        """
        Set the calibration dataset/dataloader.
        
        Args:
            calibration_data: An iterable of batches (Tensor, Mapping, tuple, or list).
        """
        self.calibration_data = calibration_data

    def _calibration_iter(self, calibration_data):
        if isinstance(calibration_data, torch.Tensor) or isinstance(calibration_data, Mapping):
            return (calibration_data,)
        if isinstance(calibration_data, tuple) and all(isinstance(item, torch.Tensor) for item in calibration_data):
            return (calibration_data,)
        return calibration_data

    def _run_calibration_batch(self, batch):
        if isinstance(batch, Mapping):
            return self.model(**batch)
        if isinstance(batch, (tuple, list)):
            return self.model(*batch)
        return self.model(batch)

    def _make_lrd_calibration_hook(self, compressor, offload_to_cpu: bool):
        def hook(module, inputs, output):
            if len(inputs) == 0:
                raise ValueError("Cannot collect LRD calibration data from a module without input activations.")
            activation = inputs[0].detach()
            in_features = getattr(module, "in_features", activation.size(-1))
            if activation.size(-1) != in_features:
                raise ValueError(
                    f"Calibration activation last dimension is {activation.size(-1)}, expected {in_features}."
                )
            if activation.dim() == 1:
                activation = activation.reshape(1, -1)
            else:
                activation = activation.reshape(-1, activation.size(-1))
            if offload_to_cpu:
                activation = activation.cpu()
            activation = activation.float()
            covariance_sum = activation.transpose(0, 1) @ activation
            if compressor._covariance_sum is None:
                compressor._covariance_sum = covariance_sum
            else:
                compressor._covariance_sum.add_(covariance_sum)
            compressor._covariance_tokens += activation.size(0)

        return hook

    def _prepare_lrd_covariance_collection(self, targets, offload_to_cpu: bool):
        for _, compressor in targets:
            compressor.clear_covariance()
            compressor._covariance_sum = None
            compressor._covariance_tokens = 0

        handles = []
        for scheme, compressor in targets:
            module = scheme.get_module()
            handles.append(module.register_forward_hook(self._make_lrd_calibration_hook(compressor, offload_to_cpu)))
        return handles

    def _finalize_lrd_covariance(self, targets):
        for scheme, compressor in targets:
            if compressor._covariance_tokens == 0:
                raise RuntimeError(f"No calibration activations were collected for module {scheme.path}.")
            compressor.set_covariance(compressor._covariance_sum / compressor._covariance_tokens)
            del compressor._covariance_sum
            del compressor._covariance_tokens

    def _infer_model_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            pass
        try:
            return next(self.model.buffers()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_to_device(self, value, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: self._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item, device) for item in value]
        if hasattr(value, "to"):
            try:
                return value.to(device)
            except TypeError:
                pass
        return value

    def _run_calibration_forward(
        self,
        max_batches: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ):
        if device is None:
            device = self._infer_model_device()
        device = torch.device(device)

        was_training = self.model.training
        num_batches = 0
        try:
            self.model.eval()
            with torch.no_grad():
                for batch_id, batch in enumerate(self._calibration_iter(self.calibration_data)):
                    if max_batches is not None and batch_id >= max_batches:
                        break
                    batch = self._move_to_device(batch, device)
                    self._run_calibration_batch(batch)
                    num_batches += 1
        finally:
            if was_training:
                self.model.train()

        if num_batches == 0:
            raise ValueError("Calibration data must contain at least one batch.")

        if verbose:
            print(f"Ran {num_batches} calibration batches.")

        return num_batches

    def run(
        self,
        targets,
        max_batches: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        offload_to_cpu: bool = True,
        verbose: bool = False,
    ):
        """
        Run calibration on target compressors by collecting covariance matrices.
        
        Args:
            targets: List of (scheme, compressor) tuples from the manager.
            max_batches: Maximum number of calibration batches to process. None = all.
            device: Device to move batches to before forward pass.
            offload_to_cpu: If True, accumulate covariance on CPU to save memory.
            verbose: If True, print calibration progress.
        
        Returns:
            Number of compressors calibrated.
        """
        if len(targets) == 0:
            if verbose:
                print("No LRD compressors found for calibration.")
            return 0

        if self.calibration_data is None:
            raise ValueError(
                "LRD method 'wsvd' requires calibration data before apply. "
                "Call set_calibration_data(...) before running calibration."
            )

        handles = self._prepare_lrd_covariance_collection(targets=targets, offload_to_cpu=offload_to_cpu)
        try:
            num_batches = self._run_calibration_forward(
                max_batches=max_batches,
                device=device,
                verbose=verbose,
            )
        finally:
            for handle in handles:
                handle.remove()

        self._finalize_lrd_covariance(targets)

        if verbose:
            print(f"Calibrated {len(targets)} LRD compressors over {num_batches} batches.")

        return len(targets)

    def clear(self, targets):
        """
        Clear covariance from target compressors.
        
        Args:
            targets: List of (scheme, compressor) tuples.
        """
        for _, compressor in targets:
            compressor.clear_covariance()


__all__ = ["LRDCalibration"]
