"""Qualcomm QNN exporter (ExecuTorch, Hexagon HTP)."""

import copy
import importlib.util
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx
import torch.nn as nn
from torch.export import export as torch_export

from ...config import BackendExportConfig
from ...exporter import register_exporter
from ...operations import config_default
from ...pt2e import calibrate_pt2e_observers, inject_scales_into_pt2e_observers
from ...quant_info import extract_layer_quant_info
from ...wrapper import resolve_components_and_wrapper
from ..executorch_base import ExecuTorchExporter
from .passes import (
    _BreakQuantAttrsBeforeFp16Linear,
    _canonicalize_pow_tensor_scalar,
    _decompose_wrap_with_set_grad_enabled,
    _float_attention_nodes,
    _linear_block_sizes,
    _split_float_linear_bias,
    _untie_shared_weights,
    make_split_graph_blocks_pass,
)


def is_qnn_available() -> bool:
    """Whether the QNN ExecuTorch backend can actually be used on this machine.

    ``executorch.backends.qualcomm`` unconditionally imports ``cpuinfo`` at
    package-init time and raises a bare ``ImportError`` if it's missing — an
    undeclared transitive dependency, not a real "toolchain not installed"
    signal. There's no side-effect-free way to probe deeper than that (the
    Qualcomm partitioner itself only fails at lowering time), so this checks
    the two cheap preconditions without importing either package: `executorch`
    and `cpuinfo` must both be resolvable, and one of the QNN SDK env vars the
    backend documents must be set. ``importlib.util.find_spec`` only locates a
    module, it does not execute its ``__init__.py``, so this is safe to call
    even when the Qualcomm toolchain isn't installed.
    """
    if importlib.util.find_spec("executorch") is None:
        return False
    if importlib.util.find_spec("cpuinfo") is None:
        return False
    return bool(os.environ.get("QNN_SDK_ROOT") or os.environ.get("QUALCOMM_SDK_ROOT"))


_QNN_UNAVAILABLE_MSG = (
    "QNN backend unavailable: install the Qualcomm ExecuTorch toolchain "
    "(`pip install py-cpuinfo` plus the Qualcomm QNN SDK) and set "
    "QNN_SDK_ROOT (or QUALCOMM_SDK_ROOT) before exporting with backend='qnn'. "
    "Use `is_qnn_available()` to check this ahead of time."
)


@dataclass
class QNNExportConfig(BackendExportConfig):
    """Qualcomm QNN (HTP) export options. Fields that select an operation take their
    default from ``export/operations.py`` (``quant_scope``, ``convert_linear_to_conv2d``)."""

    backend: str = "qnn"
    soc_model: str = "SM8650"
    is_online_prepare: bool = False
    use_fp16: bool = True
    num_shards: int = 1
    # Rewrite aten.linear into an equivalent 1x1 conv2d before lowering, which is
    # what Qualcomm's own LLM examples do
    # (executorch/examples/qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py
    # calls convert_linear_to_conv2d on the whole decoder).
    #
    # Opt-in, not default, for two reasons. It showed no measurable change in the
    # HTP compiler's DDR bandwidth summary on this model -- it swaps 14 linears
    # for 14 convolutions plus 28 reshape/permute pairs that LayoutTransform is
    # expected to fold, so any win is in kernel selection, which cannot be
    # confirmed without on-device timing. And ExecuTorch 1.3.1's pass has real
    # gaps: see _resolve_linear_to_conv2d for the two it can detect, plus a
    # non-leaf-tensor deepcopy failure that shows up on the full fp32 decoder
    # pipeline and does not reproduce on a synthetic decoder of the same shape.
    # Turn it on once that pass is fixed upstream, or for models it can handle.
    convert_linear_to_conv2d: bool = field(default_factory=lambda: config_default("qnn", "convert_linear_to_conv2d"))
    # Which ops a quantized export annotates:
    #   "linear" - (legacy) only the compressed linears; every other op stays fp16.
    #              Each quantized/float seam splits the graph: 169 HTP partitions +
    #              CPU mul/add/sigmoid on Qwen2.5-0.5B, measured on SM8850 at
    #              ~20 tok/s vs 41 tok/s fp16.
    #   "all"    - vendor recipe (ExecuTorch Qwen2.5 static-LLM recipes): every op
    #              gets 16-bit activations (MinMax, per-tensor), quantized linears
    #              use tsurgeon's exact per-channel scales (W4 -> 16a4w, W8 -> 16a8w)
    #              and linears the model left float are carried as 16a16w, so the
    #              decoder lowers to a single HTP partition.
    quant_scope: str = field(default_factory=lambda: config_default("qnn", "quant_scope"))
    # quant_scope="all" only: a float linear carried as 16a16w computes its bias in
    # int32 with scale act_scale x weight_scale, which 16-bit weights make tiny
    # (largest bias ~ input range x max|w| of the channel): Qwen2.5's q/k biases
    # (up to ~316) overflow it by up to ~65x and are clipped in every layer. True
    # moves the bias of every linear that is not compression-quantized into a
    # separate add (the bias then gets its own 16-bit scale), keeping the 16-bit
    # weights so only the targeted linears change precision.
    split_float_linear_bias: bool = True
    # quant_scope="all" only: quantize the graph (16a16w everywhere) even when no
    # linear is compression-quantized. Gives the activation-only reference point:
    # the cost of HTP's fixed-point graph without any weight compression.
    quantize_float_model: bool = False
    # Keep a deep copy of the quantized torch model the .pte is lowered from
    # (PT2E fake-quant graph for quantized exports) as result.reference_module.
    keep_reference_module: bool = False
    # quant_scope="all" only: keep the attention core (RoPE, KV cache, scores,
    # mask, softmax, A.V) in fp16 instead of 16-bit fixed point.
    float_attention_core: bool = False
    # Token-id sequences for PT2E activation calibration (each decoded from position 0
    # with the KV cache filling up). None = a few built-in sentences. Use real text
    # from the target distribution: static 16-bit ranges come only from these.
    calibration_token_ids: Any = None


def _resolve_qcom_chipset(soc_model: Any):
    try:
        from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import QcomChipset
    except ModuleNotFoundError:
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

    if isinstance(soc_model, QcomChipset):
        return soc_model

    if isinstance(soc_model, str):
        if hasattr(QcomChipset, soc_model):
            return getattr(QcomChipset, soc_model)
        normalized = soc_model.upper()
        if hasattr(QcomChipset, normalized):
            return getattr(QcomChipset, normalized)

        supported = sorted(member.name for member in QcomChipset)
        supported_str = ", ".join(supported)
        raise ValueError(
            f"Unsupported soc_model '{soc_model}'. Supported values: {supported_str}"
        )

    raise TypeError(
        "soc_model must be a QcomChipset value or chipset string (for example 'SM8650')."
    )


def _resolve_linear_to_conv2d(wrapper: nn.Module, config: QNNExportConfig) -> bool:
    """Decide whether ExecuTorch's ConvertLinearToConv2d pass can run on this model.

    The pass gives HTP a faster lowering for the tall/skinny matrices a decode
    step produces, and it is what Qualcomm's own LLM examples use. But as shipped
    in ExecuTorch 1.3.1 it has two preconditions it does not check itself, and
    violating either fails deep inside lowering with an error that says nothing
    about linear layers:

      * **float32 weights only.** ``_reshape_weight`` re-wraps its result in
        ``nn.Parameter`` only when ``weight.dtype == torch.float``; for float16 it
        stores a bare tensor where the export machinery asserts a Parameter
        ("expected torch.nn.Parameter for PARAMETER attr_kind, got
        <class 'torch.Tensor'>").
      * **no tied weights.** When a weight has more than one user — the usual
        embedding/lm_head tie, which the pass's own comment claims to handle — it
        registers a reshaped *fake* tensor as a buffer. That tensor is not a graph
        leaf, so the later ``deepcopy`` of the partitioned submodule raises "Only
        Tensors created explicitly by the user (graph leaves) support the deepcopy
        protocol".

    Both are ExecuTorch-side limitations. Rather than let the export die, detect
    them up front, fall back to plain FullyConnected lowering, and say why.
    Exporting in float32 is the usual QNN flow anyway: HTP's own precision comes
    from ``generate_htp_compiler_spec(use_fp16=...)``, not from the dtype of the
    graph's constants.
    """
    if not config.convert_linear_to_conv2d:
        return False

    linears = [m for m in wrapper.modules() if isinstance(m, nn.Linear)]

    dtypes = {m.weight.dtype for m in linears} - {torch.float32}
    if dtypes:
        warnings.warn(
            "convert_linear_to_conv2d is disabled for this export: ExecuTorch's "
            "ConvertLinearToConv2d pass only supports float32 linear weights, but "
            f"this model has {sorted(str(d) for d in dtypes)}. Re-export with "
            "float_type=torch.float32 to get the conv2d lowering — HTP precision is "
            "controlled by use_fp16 in the compiler spec, not by the graph dtype.",
            stacklevel=2,
        )
        return False

    # The tie that matters in practice is lm_head <-> tok_embeddings, i.e. an
    # nn.Linear sharing storage with an nn.Embedding, so both types are scanned.
    seen: dict[int, str] = {}
    for name, m in wrapper.named_modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        key = id(m.weight)
        other = seen.get(key)
        if other is not None:
            warnings.warn(
                "convert_linear_to_conv2d is disabled for this export: ExecuTorch's "
                "ConvertLinearToConv2d pass mishandles weights shared by more than "
                f"one layer, and {name!r} shares its weight with {other!r} (the "
                "usual embedding/lm_head tie). Untie the weights to get the conv2d "
                "lowering.",
                stacklevel=2,
            )
            return False
        seen[key] = name
    return True


def _layer_predicate(layer_names: frozenset):
    """Quantiser predicate: the node belongs to one of ``layer_names`` (any enclosing module)."""
    def _pred(node: torch.fx.Node) -> bool:
        stack = node.meta.get("nn_module_stack", {})
        return any(qname in layer_names for qname, _ in reversed(list(stack.values())))
    return _pred


@register_exporter
class QNNExporter(ExecuTorchExporter):
    """Trace, sanitise, quantise (PT2E with the QNN quantiser), lower to HTP and write the program."""

    backend = "qnn"
    config_class = QNNExportConfig

    def check_available(self):
        if not is_qnn_available():
            raise RuntimeError(_QNN_UNAVAILABLE_MSG)

    def lower(self, components, config):
        wrapper, model_config, example_inputs = resolve_components_and_wrapper(components)
        layer_info = extract_layer_quant_info(wrapper)

        # HTP runs block-quantised (LPBQ) weights only on convolutions, so such exports
        # lower linears as 1x1 conv2d; that pass cannot handle tied weights: untie them.
        lpbq = any(i.get("group_size") is not None for i in layer_info.values())
        if lpbq:
            _untie_shared_weights(wrapper)

        with torch.inference_mode():
            exported = torch_export(wrapper, example_inputs)
        module = exported.module()
        _decompose_wrap_with_set_grad_enabled(module)
        _canonicalize_pow_tensor_scalar(module)

        if config.quantize_float_model and config.quant_scope != "all":
            raise ValueError("QNNExportConfig.quantize_float_model requires quant_scope='all'.")
        quantize = bool(layer_info) or config.quantize_float_model
        model_for_edge, exported_for_mismatch = module, None
        if quantize:
            model_for_edge, exported_for_mismatch = self._quantize(module, layer_info, model_config,
                                                                   example_inputs, config)

        # Lowering may rewrite the module in place: keep an untouched copy if requested.
        reference_module = copy.deepcopy(model_for_edge) if config.keep_reference_module else None

        try:
            from executorch.backends.qualcomm.utils.utils import (
                generate_htp_compiler_spec,
                generate_qnn_executorch_compiler_spec,
                to_edge_transform_and_lower_to_qnn,
            )
        except ImportError as e:
            raise RuntimeError(_QNN_UNAVAILABLE_MSG) from e

        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=_resolve_qcom_chipset(config.soc_model),
            backend_options=generate_htp_compiler_spec(use_fp16=config.use_fp16,
                                                       use_multi_contexts=(config.num_shards > 1)),
            online_prepare=config.is_online_prepare,
        )
        edge = to_edge_transform_and_lower_to_qnn(
            model_for_edge,
            example_inputs,
            compiler_spec,
            convert_linear_to_conv2d=lpbq or _resolve_linear_to_conv2d(wrapper, config),
            **self._edge_pass_options(wrapper, config, quantize),
        )
        self.write_program(edge, config, alloc_graph_input=True, alloc_graph_output=True)
        result = self.finalize(config, precision="mixed" if quantize else "full", wrapper=wrapper,
                               example_inputs=example_inputs, model_config=model_config,
                               exported_for_mismatch=exported_for_mismatch)
        # The torch model the .pte was lowered from: host-side reference for on-device parity.
        result.reference_module = reference_module
        return result

    def _build_quantizer(self, module, layer_info, config):
        from executorch.backends.qualcomm.quantizer.quantizer import ModuleQConfig, QnnQuantizer, QuantDtype
        from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchBackendType

        # Compression precision -> QNN quantisation dtype (activations x weights).
        prec_to_qdtype = {
            4: QuantDtype.use_16a4w,
            8: QuantDtype.use_16a8w if config.quant_scope == "all" else QuantDtype.use_8a8w,
        }
        quantizer = QnnQuantizer(backend=QnnExecuTorchBackendType.kHtpBackend,
                                 soc_model=_resolve_qcom_chipset(config.soc_model))

        layers_by_qdtype: dict = defaultdict(set)
        block_layers: dict[str, int] = {}
        for name, info in layer_info.items():
            if info.get("group_size") is not None:
                # HTP block quantisation (LPBQ): 4-bit weights, group scale = channel
                # scale x a 4-bit integer (quantisation group_scale_bits=4).
                if info["precision"] != 4 or info.get("group_scale_bits") != 4:
                    raise NotImplementedError(
                        f"QNN export: '{name}' is per-group quantized with free group scales "
                        f"(precision={info['precision']}, group_scale_bits={info.get('group_scale_bits')}). "
                        "HTP block quantization (LPBQ) holds 4-bit weights with group scales = channel "
                        "scale x a 4-bit integer: quantize with precision=4, granularity='per_group', "
                        "group_scale_bits=4.")
                block_layers[name] = int(info["group_size"])
                layers_by_qdtype[QuantDtype.use_16a4w_block].add(name)
                continue
            qdtype = prec_to_qdtype.get(info["precision"])
            if qdtype is not None:
                layers_by_qdtype[qdtype].add(name)

        act_observer = None
        if config.quant_scope == "all":
            from torchao.quantization.pt2e import MinMaxObserver
            act_observer = MinMaxObserver
        if config.quant_scope == "all":
            # Every other operation (and any float linear) gets 16-bit activations and weights.
            quantizer.set_default_quant_config(quant_dtype=QuantDtype.use_16a16w, is_linear_per_channel=True,
                                               act_observer=act_observer)
            if config.float_attention_core:
                quantizer.add_discard_nodes(_float_attention_nodes(module))
            if config.split_float_linear_bias:
                split = _split_float_linear_bias(module, frozenset(layer_info))
                if split:
                    print(f"  QNN: bias split from {len(split)} float (16a16w) linears, e.g. {split[:3]}")
        else:
            # Unmatched nodes stay float: a default config that annotates nothing.
            float_config = ModuleQConfig()
            float_config.use_per_channel_weight_quant_ops = {}
            float_config.quant_config = None
            quantizer.default_quant_config = float_config

        # Same order as the vendor recipe: default config first, then per-layer configs.
        quantizer.set_submodule_qconfig_list([
            (_layer_predicate(frozenset(layers)),
             ModuleQConfig(quant_dtype=qdtype, is_linear_per_channel=True, act_observer=act_observer))
            for qdtype, layers in layers_by_qdtype.items()
        ])
        if block_layers:
            quantizer.set_block_size_map(_linear_block_sizes(module, block_layers))
        return quantizer

    def _quantize(self, module, layer_info, model_config, example_inputs, config):
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        prepared = prepare_pt2e(module, self._build_quantizer(module, layer_info, config))
        calibrate_pt2e_observers(prepared, model_config, config, example_inputs=example_inputs)
        inject_scales_into_pt2e_observers(prepared, layer_info)
        converted = convert_pt2e(prepared)
        with torch.inference_mode():
            quantized_exported = torch_export(converted, example_inputs)
        return converted, quantized_exported

    def _edge_pass_options(self, wrapper, config, quantize):
        """Extra keyword arguments of ``to_edge_transform_and_lower_to_qnn`` (sharding, float16 fix-up)."""
        if config.num_shards <= 1 and not quantize:
            return {}
        from executorch.backends.qualcomm._passes import LayoutTransform, TagQuantIO
        from executorch.backends.qualcomm._passes.qnn_pass_manager import get_capture_program_passes
        from executorch.backends.qualcomm._passes.utils import get_passes_dependency_for_capture_program
        from executorch.backends.qualcomm.utils.constants import (
            QCOM_PASS_ACTIVATE_KEY,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
        )

        options = {}
        passes_job = get_capture_program_passes()
        dep_table = get_passes_dependency_for_capture_program()
        if config.num_shards > 1:
            # llama.fallback marks shard boundaries and must stay out of QNN partitions.
            options["skip_node_op_set"] = {"llama.fallback.default"}
            try:
                n_layers = len(wrapper.decoder.blocks)
            except AttributeError:
                raise RuntimeError("num_shards > 1 requires wrapper.decoder.blocks "
                                   "(TransformerDecoder with a 'blocks' ModuleList)")
            shard_layers = list(range(0, n_layers, n_layers // config.num_shards))
            print(f"  Graph sharding: {config.num_shards} shards, layer boundaries at {shard_layers}")
            passes_job[make_split_graph_blocks_pass(shard_layers)] = {
                QCOM_PASS_ACTIVATE_KEY: True,
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {"shard_layers": shard_layers},
            }
        if quantize:
            # Mixed precision: after FoldQDQ the quantisation annotation survives on
            # activations feeding float16 linears, which makes their FC op invalid. Insert
            # a float16 cast before them (after LayoutTransform, before TagQuantIO).
            passes_job[_BreakQuantAttrsBeforeFp16Linear] = {
                QCOM_PASS_ACTIVATE_KEY: True,
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {},
            }
            dep_table[_BreakQuantAttrsBeforeFp16Linear] = [LayoutTransform]
            dep_table[TagQuantIO] = list(dep_table.get(TagQuantIO, [LayoutTransform])) + [
                _BreakQuantAttrsBeforeFp16Linear]
        options["passes_job"] = passes_job
        options["dep_table"] = dep_table
        return options


__all__ = ["QNNExportConfig", "QNNExporter", "is_qnn_available"]
