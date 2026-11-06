"""FLAP scores (Fluctuation-based Adaptive Structured Pruning, An et al., 2023): standardised per layer, one global cut."""

from __future__ import annotations

import torch

from .global_base import GlobalCalibrationSummary


class FlapScoresSummary(GlobalCalibrationSummary):
    """Fluctuation score of every input channel of every participating layer, and the global keep-masks.

    Per layer (scheme), for input channel j: ``S_j = Var(x_j) * ||W[:, j]||^2`` (eq. 5 of the paper; the variance
    is taken over all calibration tokens around the baseline mean, ``E[x^2] - E[x]^2``). Channels are scored in
    groups of ``group_size`` consecutive channels (the compressor's option; e.g. an attention head), summing the
    channel scores. Scores are standardised per layer (zero mean, unit std), then the groups of all layers are
    ranked together and the lowest ones are pruned until ``options['ratio']`` of the weights of the participating
    (consumer) layers is removed. This is the paper's adaptive structure search: layers with unimportant channels
    lose more.

    Result: ``global_store['flap_scores'] = {'keep': {scheme.path: BoolTensor[in]}, 'mean': {scheme.path: Tensor[in]},
    'ratios': {scheme.path: float}, 'threshold': float}``.
    """

    name = "flap_scores"
    requires = ("input_mean", "input_sq_mean")

    def compute(self, entries, options, global_store):
        ratio = float(options["ratio"])
        scores, means, costs, groups = {}, {}, {}, {}
        for scheme, store in entries:
            module = scheme.get_compression_module()
            w = module.weight.detach().float()
            g = int(scheme.compressors["channel_pruning"].group_size)
            mean = store["input_mean"].float().to(w.device)
            var = (store["input_sq_mean"].float().to(w.device) - mean.pow(2)).clamp_min(0.0)
            s = (var * w.pow(2).sum(0)).reshape(-1, g).sum(1)  # one score per group of g consecutive channels
            scores[scheme.path] = (s - s.mean()) / s.std().clamp_min(1e-12)
            means[scheme.path] = mean
            groups[scheme.path] = g
            costs[scheme.path] = torch.full_like(s, float(g * w.shape[0]))  # weights of the consumer layer in the group
        paths = list(scores)
        all_scores = torch.cat([scores[p].cpu() for p in paths])
        all_costs = torch.cat([costs[p].cpu() for p in paths])
        order = all_scores.argsort()
        cum = all_costs[order].cumsum(0)
        n_prune = int((cum <= ratio * all_costs.sum()).sum())
        threshold = float("-inf") if n_prune == 0 else all_scores[order[n_prune - 1]].item()
        pruned_flat = torch.zeros_like(all_scores, dtype=torch.bool)
        pruned_flat[order[:n_prune]] = True
        keep, offset = {}, 0
        for p in paths:
            n = scores[p].numel()
            keep[p] = (~pruned_flat[offset:offset + n]).repeat_interleave(groups[p]).to(scores[p].device)
            offset += n
        global_store[self.name] = dict(
            keep=keep,
            mean=means,
            ratios={p: 1.0 - keep[p].float().mean().item() for p in paths},
            threshold=threshold,
        )
