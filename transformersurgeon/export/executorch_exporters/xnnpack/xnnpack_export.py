import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.export import export as torch_export

from ..common import (
    ExecutorchExporterConfig,
    ExecuTorchExportResult,
    finalize_export_result,
    resolve_components_and_wrapper,
    build_quantizer_from_layer_info,
    extract_layer_quant_info,
    inject_scales_into_pt2e_observers,
)


@dataclass
class XNNPACKExportConfig(ExecutorchExporterConfig):
    pass



_CAL_SENTENCES = [
    "The Eiffel Tower is located in Paris, France, and was built in 1889.",
    "Machine learning models learn patterns from large datasets to make predictions.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "Python is widely used for data science, machine learning, and automation tasks.",
    "The capital of Japan is Tokyo, which is one of the most populous cities on Earth.",
    "Quantum computers use quantum mechanical phenomena to perform complex calculations.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
    "Climate change is caused by greenhouse gas emissions from fossil fuels and deforestation.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "Artificial intelligence is transforming industries from healthcare to finance.",
    "Water covers approximately 71 percent of the Earth's surface.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
    "The Amazon rainforest produces 20 percent of the world's oxygen supply.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules.",
]


def _calibrate_pt2e_observers(prepared, model_config, config) -> None:
    """Run real-text calibration passes through the prepared model.

    Random token calibration leaves output activation observers with degenerate
    histograms (e.g. zp=127, clipping all positive values to 0) because the
    distribution of gate_proj / up_proj outputs with uniformly-random token IDs
    is very different from real text.  Tokenizing a small set of fixed sentences
    produces realistic activation distributions for ALL observers (input and output).
    Falls back to random passes if the tokenizer cannot be loaded.
    """
    model_name = getattr(model_config, '_name_or_path', None) if model_config else None
    vocab_size = int(getattr(model_config, 'vocab_size', 32000)) if model_config else 32000
    max_pos = max(1, min(512, getattr(config, 'max_seq_len', 512)))

    token_batches: list[list[int]] = []
    if model_name:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_name)
            for sent in _CAL_SENTENCES:
                ids = tok.encode(sent, add_special_tokens=True)
                token_batches.append(ids)
        except Exception:
            pass  # fall through to random

    with torch.no_grad():
        if token_batches:
            for ids in token_batches:
                for token_id in ids:
                    inp = torch.tensor([token_id], dtype=torch.long)
                    pos = torch.tensor([1], dtype=torch.long)
                    prepared(inp, pos)
        else:
            for _ in range(32):
                rand_ids = torch.randint(0, vocab_size, (1,), dtype=torch.long)
                rand_pos = torch.randint(1, max_pos + 1, (1,), dtype=torch.long)
                prepared(rand_ids, rand_pos)


def export_with_xnnpack(
    model_or_graph: Any,
    *,
    config: XNNPACKExportConfig,
) -> ExecuTorchExportResult:
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    wrapper, model_config, example_inputs = resolve_components_and_wrapper(
        model_or_graph,
        config=config,
    )

    # Detect per-layer compression metadata (hard / soft quantized layers).
    # Quantization is driven entirely by model metadata — config.precision is not consulted.
    layer_info = extract_layer_quant_info(wrapper)

    exported = torch_export(
        wrapper,
        example_inputs,
    )
    export_for_edge = exported
    exported_for_mismatch = None

    if layer_info:
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

        quantizer = build_quantizer_from_layer_info(layer_info)
        prepared = prepare_pt2e(exported.module(), quantizer)

        # Calibration pass: run multiple forward passes with real text so that all
        # observers (including output-activation observers on gate/up/down projections)
        # collect representative statistics.  Random-token passes leave output observers
        # with degenerate histograms (gate/up outputs are mostly negative for random
        # tokens → zp=127 → clips all positive activations to 0).
        _calibrate_pt2e_observers(prepared, model_config, config)

        # Override observer results for hard-quantized layers with exact surgeon scales.
        inject_scales_into_pt2e_observers(prepared, layer_info)

        converted = convert_pt2e(prepared)
        quantized_exported = torch_export(converted, example_inputs)
        export_for_edge = quantized_exported
        exported_for_mismatch = quantized_exported

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    edge = to_edge_transform_and_lower(
        export_for_edge,
        compile_config=EdgeCompileConfig(_check_ir_validity=config.check_ir_validity),
        partitioner=[XnnpackPartitioner()],
    )

    et_program = edge.to_executorch()
    with open(config.output_path, "wb") as f:
        f.write(et_program.buffer)

    # Summarize precision from layer metadata for the result object.
    result_precision = "mixed" if layer_info else "full"

    return finalize_export_result(
        pte_path=config.output_path,
        backend="xnnpack",
        precision=result_precision,
        wrapper=wrapper,
        example_inputs=example_inputs,
        exported_for_mismatch=exported_for_mismatch,
        run_weight_mismatch_check=config.run_weight_mismatch_check,
        weight_mismatch_eps=config.weight_mismatch_eps,
        verbose=config.verbose,
    )
