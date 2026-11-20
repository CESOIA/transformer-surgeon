"""
Inference with a TensorRT engine built from a tsurgeon encoder export (``tsurgeon-encoder/1``).

TensorRTEncoderSession drives the graph contract recorded in the manifest: one static model input
(``pixel_values`` or ``input_ids``) and ``logits`` out, plus one ``rank_<group>`` input per runtime rank
group (``export/runtime_ranks.py``). The active ranks can change between calls (``set_ranks``): the
latency/accuracy operating point of the same engine and the same weights is chosen at run time.

Each distinct rank setting gets its own CUDA graph, captured on first use and replayed afterwards, so a
call is one input copy plus one graph replay. Every rank binds a prefix of the same rank buffer and the
same input/output buffers, so the graphs of all rank settings stay valid side by side and switching is
free once each setting has run once.
"""

import json
import os
from typing import Iterable, Mapping

import torch

from ...runtime_ranks import RANK_INPUT_DTYPE
from .edgellm_int4 import resolve_plugin_libraries
from .engine import TensorRTEngineRunner


class TensorRTEncoderSession:
    """Run an exported encoder engine; ``ranks`` is the initial ``{group: rank}`` (default: the largest
    ranks one optimization profile accepts).

    An engine built with several optimization profiles (``TensorRTExportConfig.rank_profiles``) gets one
    execution context per profile; ``set_ranks`` uses the first profile whose ranges contain the ranks.
    """

    def __init__(self, manifest_path: str, engine_path: str | None = None, *, device: str = "cuda",
                 plugin_libraries: Iterable[str] = (), ranks: Mapping[str, int] | None = None,
                 cuda_graphs: bool = True):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        if self.manifest.get("structure") != "transformer_encoder":
            raise ValueError(f"{manifest_path} is not an encoder export (structure="
                             f"{self.manifest.get('structure')!r}).")
        base = os.path.dirname(os.path.abspath(manifest_path))
        engine_path = engine_path or os.path.join(base, os.path.splitext(self.manifest["onnx_file"])[0] + ".engine")
        self.device = torch.device(device)
        self.rank_specs: dict[str, dict] = dict(self.manifest.get("runtime_ranks", {}))
        self.profiles: list[dict] = self.manifest.get("rank_profiles") or [
            {group: [spec["min"], spec["opt"], spec["max"]] for group, spec in self.rank_specs.items()}]
        first = TensorRTEngineRunner(engine_path, device=self.device,
                                     plugin_libraries=resolve_plugin_libraries(self.manifest, tuple(plugin_libraries)))
        count = first.engine.num_optimization_profiles
        if self.rank_specs and count != len(self.profiles):
            raise RuntimeError(f"The engine has {count} optimization profiles, the manifest records "
                               f"{len(self.profiles)} rank profiles.")
        self._runners = [first] + [None] * (count - 1)
        self._engine_path = engine_path
        self.input_name = next(name for name, spec in self.manifest["inputs"].items()
                               if "runtime_rank_group" not in spec)
        self.input = torch.zeros(self.manifest["inputs"][self.input_name]["shape"],
                                 dtype=first.dtype(self.input_name), device=self.device)
        self.cuda_graphs = cuda_graphs
        # One zero buffer per group, as long as its max rank; a rank r binds its first r entries.
        self._rank_buffers = {group: torch.zeros(spec["max"], dtype=RANK_INPUT_DTYPE, device=self.device)
                              for group, spec in self.rank_specs.items()}
        self.logits = torch.empty(self.manifest["outputs"]["logits"]["shape"], dtype=first.dtype("logits"),
                                  device=self.device)
        self.ranks: dict[str, int] = {}
        self.profile = None
        # Default: the largest ranks one profile accepts.
        widest = max(self.profiles, key=lambda p: sum(bounds[2] for bounds in p.values())) if self.rank_specs else {}
        self.set_ranks({group: bounds[2] for group, bounds in widest.items()} | dict(ranks or {}))

    @property
    def rank_range(self) -> dict[str, tuple[int, int]]:
        """``{group: (min, max)}`` of the exported rank inputs."""
        return {group: (spec["min"], spec["max"]) for group, spec in self.rank_specs.items()}

    def supported_ranks(self, group: str) -> list[tuple[int, int]]:
        """``(min, max)`` ranges of ``group`` that some optimization profile accepts."""
        return sorted({(p[group][0], p[group][2]) for p in self.profiles})

    def _runner(self, profile: int) -> TensorRTEngineRunner:
        if self._runners[profile] is None:
            self._runners[profile] = TensorRTEngineRunner(self._runners[0], profile=profile, device=self.device)
        return self._runners[profile]

    def _profile_for(self, ranks: Mapping[str, int]) -> int:
        for k, profile in enumerate(self.profiles):
            if all(profile[g][0] <= r <= profile[g][2] for g, r in ranks.items()):
                return k
        raise ValueError(f"No optimization profile of the engine accepts ranks {dict(ranks)}; profiles: "
                         f"{self.profiles}.")

    def set_ranks(self, ranks: Mapping[str, int]) -> None:
        """Select the active rank of the given groups (the others keep theirs)."""
        new = {**self.ranks, **{group: int(rank) for group, rank in ranks.items()}}
        unknown = sorted(set(new) - set(self.rank_specs))
        if unknown:
            raise KeyError(f"Unknown runtime rank groups {unknown}; the engine has {sorted(self.rank_specs)}.")
        profile = self._profile_for(new) if self.rank_specs else 0
        if new == self.ranks and profile == self.profile:
            return
        runner = self._runner(profile)
        tensors = {self.input_name: self.input, "logits": self.logits}
        for group, rank in new.items():
            tensors[self.rank_specs[group]["input"]] = self._rank_buffers[group][:rank]
        # Same buffers, new shapes: the graphs captured for other rank settings remain valid.
        runner.bind(tensors, keep_graphs=True)
        self.ranks, self.profile = new, profile

    def _key(self):
        return tuple(sorted(self.ranks.items()))

    def run(self) -> torch.Tensor:
        """Run on the current content of ``self.input``; returns the (reused) logits buffer."""
        runner = self._runners[self.profile]
        if not self.cuda_graphs:
            runner.run()
            return self.logits
        key = self._key()
        if not runner.has_graph(key):
            runner.capture_graph(key)
        runner.replay(key)
        return self.logits

    def __call__(self, x: torch.Tensor, ranks: Mapping[str, int] | None = None) -> torch.Tensor:
        """Logits for ``x`` (copied into the engine input) at ``ranks`` (default: the current ones)."""
        if ranks:
            self.set_ranks(ranks)
        self.input.copy_(x.reshape(self.input.shape))
        return self.run()


__all__ = ["TensorRTEncoderSession"]
