"""
Greedy generation with a TensorRT engine built from a tsurgeon ONNX export.

TensorRTLLMSession drives the graph contract recorded in the export manifest:

  * KV caches are allocated once; with ``cache_inplace`` each present_* output
    is bound to its past_* input buffer (TensorRT updates it in place).
  * prefill() feeds the prompt through the prefill profile in chunks of up to
    max_input_len tokens (or token by token for a decode-only export).
  * decode steps replay one CUDA graph that runs the engine, takes the greedy
    argmax of the logits straight into the next step's input_ids and advances
    pos_id -- no host round trip unless the caller reads the token.
"""

import json
import os
from typing import Iterable

import torch

from .edgellm_int4 import resolve_plugin_libraries
from .engine import TensorRTEngineRunner
from .tensorrt_export import DECODE_PROFILE, PREFILL_PROFILE

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class TensorRTLLMSession:
    def __init__(self, manifest_path: str, engine_path: str | None = None, *,
                 device: str = "cuda", plugin_libraries: Iterable[str] = ()):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        base = os.path.dirname(os.path.abspath(manifest_path))
        engine_path = engine_path or os.path.join(base, os.path.splitext(self.manifest["onnx_file"])[0] + ".engine")
        self.device = torch.device(device)
        self.max_cache_len = int(self.manifest["max_cache_len"])
        self.max_input_len = int(self.manifest["max_input_len"])
        self.eos_token_ids = set(self.manifest.get("eos_token_id") or [])

        plugin_libraries = resolve_plugin_libraries(self.manifest, tuple(plugin_libraries))
        self.decoder = TensorRTEngineRunner(engine_path, profile=DECODE_PROFILE, device=self.device,
                                            plugin_libraries=plugin_libraries)
        self.prefiller = None
        if self.max_input_len > 1 and self.decoder.engine.num_optimization_profiles > PREFILL_PROFILE:
            self.prefiller = TensorRTEngineRunner(self.decoder, profile=PREFILL_PROFILE, device=self.device)

        cache_dtype = _DTYPES[self.manifest["cache_dtype"]]
        self.caches: dict[str, torch.Tensor] = {}
        for layer in self.manifest["caches"]:
            for kind in ("key", "value"):
                spec = layer[kind]
                buf = torch.zeros(spec["shape"], dtype=cache_dtype, device=self.device)
                self.caches[spec["input"]] = buf
                # In place: the output *is* the input buffer. Otherwise the
                # engine writes a fresh tensor that is copied back after a step.
                self.caches[spec["output"]] = buf if self.manifest["cache_inplace"] else torch.empty_like(buf)
        self._cache_pairs = [(l[k]["input"], l[k]["output"]) for l in self.manifest["caches"] for k in ("key", "value")]

        self.input_ids = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.pos_id = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.decoder.bind({"input_ids": self.input_ids, "pos_id": self.pos_id, **self.caches})
        self.logits = self.decoder.allocate_outputs(skip=self.caches)["logits"]
        self._prefill_ids = torch.zeros(max(self.max_input_len, 1), dtype=torch.int64, device=self.device)
        self._prefill_pos = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.position = 0

    # -- state -------------------------------------------------------------------
    def reset(self) -> None:
        """Start a new sequence. Stale cache rows beyond the position are masked."""
        self.position = 0
        self.pos_id.zero_()

    def _sync_caches(self) -> None:
        if not self.manifest["cache_inplace"]:
            for src, dst in self._cache_pairs:
                self.caches[src].copy_(self.caches[dst])

    def _select_token(self) -> None:
        self.input_ids.copy_(torch.argmax(self.logits.float(), dim=-1, keepdim=True).view(1))
        self.pos_id.add_(1)

    # -- prompt --------------------------------------------------------------------
    def prefill(self, token_ids: torch.Tensor) -> int:
        """Run the prompt; leaves input_ids = greedy next token. Returns it."""
        token_ids = torch.as_tensor(token_ids, dtype=torch.int64).view(-1).to(self.device)
        if self.position + token_ids.numel() > self.max_cache_len:
            raise ValueError("prompt exceeds max_cache_len")
        if self.prefiller is None:
            for t in token_ids:
                self.input_ids.copy_(t.view(1))
                self.pos_id.fill_(self.position)
                self.decoder.run()
                self._sync_caches()
                self.position += 1
        else:
            for start in range(0, token_ids.numel(), self.max_input_len):
                chunk = token_ids[start:start + self.max_input_len]
                ids = self._prefill_ids[:chunk.numel()]
                ids.copy_(chunk)
                self._prefill_pos.fill_(self.position)
                self.prefiller.bind({"input_ids": ids, "pos_id": self._prefill_pos, **self.caches, "logits": self.logits})
                self.prefiller.run()
                self._sync_caches()
                self.position += chunk.numel()
        self.pos_id.fill_(self.position - 1)
        self._select_token()  # input_ids <- argmax, pos_id <- position
        return int(self.input_ids.item())

    # -- decode --------------------------------------------------------------------
    def _ensure_decode_graph(self) -> None:
        if self.decoder.has_graph():
            return
        # The warm-up enqueue really runs the pending step: it writes the KV
        # row at pos_id that the first replay rewrites identically, and leaves
        # input_ids/pos_id alone (token selection only runs inside the graph).
        self.decoder.capture_graph(after=lambda: (self._sync_caches(), self._select_token()))

    def step(self) -> None:
        """One decode step on the GPU: consume input_ids at pos_id, produce the next."""
        self._ensure_decode_graph()
        self.decoder.replay()
        self.position += 1

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, *, stop_at_eos: bool = True) -> list[int]:
        """Greedy generation. Returns the generated token ids (prompt excluded)."""
        self.reset()
        if max_new_tokens <= 0:
            return []
        out = [self.prefill(prompt_ids)]
        while len(out) < max_new_tokens and self.position < self.max_cache_len:
            if stop_at_eos and out[-1] in self.eos_token_ids:
                break
            self.step()
            out.append(int(self.input_ids.item()))
        return out


__all__ = ["TensorRTLLMSession"]
