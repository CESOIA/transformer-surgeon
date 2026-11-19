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
    (consumer) layers is removed. Layers can be ranked in separate pools through the compressor's ``rank_group``
    label (e.g. attention and MLP), each pool with its own threshold; ``threshold`` is then a dict label -> value. This is the paper's adaptive structure search: layers with unimportant channels
    lose more.

    Result: ``global_store['flap_scores'] = {'keep': {scheme.path: BoolTensor[in]}, 'mean': {scheme.path: Tensor[in]},
    'ratios': {scheme.path: float}, 'threshold': float}``.
    """

    name = "flap_scores"
    requires = ("input_mean", "input_sq_mean")

    def _channel_scores(self, w, store, g):
        mean = store["input_mean"].float().to(w.device)
        var = (store["input_sq_mean"].float().to(w.device) - mean.pow(2)).clamp_min(0.0)
        return (var * w.pow(2).sum(0)).reshape(-1, g).sum(1), mean

    def compute(self, entries, options, global_store):
        ratio = float(options["ratio"])
        scores, means, costs, groups = {}, {}, {}, {}
        for scheme, store in entries:
            module = scheme.get_compression_module()
            w = module.weight.detach().float()
            g = int(scheme.compressors["channel_pruning"].group_size)
            s, mean = self._channel_scores(w, store, g)  # one score per group of g consecutive channels
            scores[scheme.path] = (s - s.mean()) / s.std().clamp_min(1e-12)
            means[scheme.path] = mean
            groups[scheme.path] = g
            costs[scheme.path] = torch.full_like(s, float(g * w.shape[0]))  # weights of the consumer layer in the group
        # Layers sharing a ``rank_group`` label are ranked together, each label with its own threshold for the same ratio.
        labels = {scheme.path: scheme.compressors["channel_pruning"].rank_group for scheme, _ in entries}
        keep, thresholds = {}, {}
        for label in sorted(set(labels.values())):
            paths = [p for p in scores if labels[p] == label]
            all_scores = torch.cat([scores[p].cpu() for p in paths])
            all_costs = torch.cat([costs[p].cpu() for p in paths])
            order = all_scores.argsort()
            cum = all_costs[order].cumsum(0)
            n_prune = int((cum <= ratio * all_costs.sum()).sum())
            thresholds[label] = float("-inf") if n_prune == 0 else all_scores[order[n_prune - 1]].item()
            pruned_flat = torch.zeros_like(all_scores, dtype=torch.bool)
            pruned_flat[order[:n_prune]] = True
            offset = 0
            for p in paths:
                n = scores[p].numel()
                keep[p] = (~pruned_flat[offset:offset + n]).repeat_interleave(groups[p]).to(scores[p].device)
                offset += n
        paths = list(scores)
        threshold = thresholds
        global_store[self.name] = dict(
            keep=keep,
            mean=means,
            ratios={p: 1.0 - keep[p].float().mean().item() for p in paths},
            threshold=threshold,
        )


class MagnitudeScoresSummary(FlapScoresSummary):
    """Uncalibrated counterpart of :class:`FlapScoresSummary`: same standardisation, global cut and pools, but the
    score of a channel group is the squared norm of its weight columns (no activations, so no baseline-mean bias).

    Result: ``global_store['magnitude_scores']`` with the same layout as ``flap_scores`` (``mean`` is all zeros).
    """

    name = "magnitude_scores"
    requires = ()

    def _channel_scores(self, w, store, g):
        return w.pow(2).sum(0).reshape(-1, g).sum(1), torch.zeros(w.shape[1], device=w.device)

    def compute(self, entries, options, global_store):
        entries = [(sch, st) for sch, st in entries if sch.compressors["channel_pruning"]._to_compress()]
        super().compute(entries, options, global_store)  # writes global_store[self.name]
