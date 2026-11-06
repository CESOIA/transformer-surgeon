"""FLAP scores (Fluctuation-based Adaptive Structured Pruning, An et al., 2023): standardised per layer, one global cut."""

from __future__ import annotations

import torch

from .global_base import GlobalCalibrationSummary


class FlapScoresSummary(GlobalCalibrationSummary):
    """Fluctuation score of every input channel of every participating layer, and the global keep-masks.

    Per layer (scheme), for input channel j: ``S_j = Var(x_j) * ||W[:, j]||^2`` (eq. 5 of the paper; the variance
    is taken over all calibration tokens around the baseline mean, ``E[x^2] - E[x]^2``). Scores are standardised
    per layer (zero mean, unit std), then the channels of all layers are ranked together and the lowest
    ``options['ratio']`` fraction is pruned (a channel costs the same in every participating layer here, so the
    fraction is of channels). This is the paper's adaptive structure search: layers with unimportant channels
    lose more.

    Result: ``global_store['flap_scores'] = {'keep': {scheme.path: BoolTensor[in]}, 'mean': {scheme.path: Tensor[in]},
    'ratios': {scheme.path: float}, 'threshold': float}``.
    """

    name = "flap_scores"
    requires = ("input_mean", "input_sq_mean")

    def compute(self, entries, options, global_store):
        ratio = float(options["ratio"])
        scores = {}
        means = {}
        for scheme, store in entries:
            module = scheme.get_compression_module()
            w = module.weight.detach().float()
            mean = store["input_mean"].float().to(w.device)
            var = (store["input_sq_mean"].float().to(w.device) - mean.pow(2)).clamp_min(0.0)
            s = var * w.pow(2).sum(0)  # [in_features]
            scores[scheme.path] = (s - s.mean()) / s.std().clamp_min(1e-12)
            means[scheme.path] = mean
        all_scores = torch.cat([s.cpu() for s in scores.values()])
        n_prune = int(round(ratio * all_scores.numel()))
        threshold = float("-inf") if n_prune <= 0 else all_scores.sort().values[n_prune - 1].item()
        keep = {path: (s > threshold) if n_prune > 0 else torch.ones_like(s, dtype=torch.bool) for path, s in scores.items()}
        global_store[self.name] = dict(
            keep=keep,
            mean=means,
            ratios={path: 1.0 - k.float().mean().item() for path, k in keep.items()},
            threshold=threshold,
        )
