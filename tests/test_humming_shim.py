"""Contract tests for the top-level ``humming`` package that vLLM consumes.

These run on CPU: kernel launches are intercepted and every CUDA-dependent
path is exercised only up to the point where chord would compile or launch.
They complement ``tests/test_humming_facade.py``, which pins the ``chord``
facade; this file pins the ``humming`` import root that frameworks with a
hardcoded upstream name resolve, including the vLLM-specific w2/w13 block-M
alignment on the published tuning tables.
"""

from __future__ import annotations

import importlib
import json
import types

import pytest
import torch

# vLLM's lazy facade (vllm/utils/humming.py) resolves these module paths.
_FACADE_EXPORTS = {
    "dtypes": "humming.dtypes",
    "DataType": "humming.dtypes:DataType",
    "GemmType": "humming.config:GemmType",
    "WeightScaleType": "humming.config:WeightScaleType",
    "HummingMethod": "humming.layer:HummingMethod",
    "HummingLayerMeta": "humming.layer:HummingLayerMeta",
    "BaseInputSchema": "humming.schema:BaseInputSchema",
    "BaseWeightSchema": "humming.schema:BaseWeightSchema",
    "HummingInputSchema": "humming.schema:HummingInputSchema",
    "HummingWeightSchema": "humming.schema:HummingWeightSchema",
    "quantize_weight": "humming.utils.weight:quantize_weight",
    "AWQWeightSchema": "humming.schema:AWQWeightSchema",
    "BitnetWeightSchema": "humming.schema:BitnetWeightSchema",
    "ModeloptMxfp8WeightSchema": "humming.schema.modelopt:ModeloptMxfp8WeightSchema",
    "ModeloptNvfp4InputSchema": "humming.schema.modelopt:ModeloptNvfp4InputSchema",
    "ModeloptNvfp4WeightSchema": "humming.schema.modelopt:ModeloptNvfp4WeightSchema",
    "CompressedTensorsInputSchema": "humming.schema:CompressedTensorsInputSchema",
    "CompressedTensorsWeightSchema": "humming.schema:CompressedTensorsWeightSchema",
    "Fp8InputSchema": "humming.schema:Fp8InputSchema",
    "Fp8WeightSchema": "humming.schema.fp8:Fp8WeightSchema",
    "Mxfp4WeightSchema": "humming.schema:Mxfp4WeightSchema",
    "GptOssMxfp4WeightSchema": "humming.schema:GptOssMxfp4WeightSchema",
    "GPTQWeightSchema": "humming.schema:GPTQWeightSchema",
}


# vLLM 0.29's facade switched to Humming's functional interface; these are the
# additional paths its humming MoE experts and weight loader resolve.
_FUNCTIONAL_EXPORTS = {
    "LayerConfig": "humming.config:LayerConfig",
    "humming_forward": "humming.forward:humming_forward",
    "may_quant_input": "humming.forward:may_quant_input",
    "prepare_layer_config": "humming.transform:prepare_layer_config",
    "transform_humming_tensors": "humming.transform:transform_humming_tensors",
    "get_heuristics_config": "humming.tune:get_heuristics_config",
}


def test_facade_surface_resolves() -> None:
    for name, spec in {**_FACADE_EXPORTS, **_FUNCTIONAL_EXPORTS}.items():
        if ":" in spec:
            module_path, attr = spec.split(":", 1)
            assert hasattr(importlib.import_module(module_path), attr), name
        else:
            importlib.import_module(spec)


def test_dtypes_cover_vllm_lookup_tables() -> None:
    import chord.dtypes
    import humming.dtypes
    from chord_kernels.operator import dtypes as operator_dtypes

    for name in (
        "uint2",
        "uint3",
        "uint4",
        "uint8",
        "int4",
        "int8",
        "float16",
        "bfloat16",
        "float32",
        "float8e4m3",
        "float8e5m2",
        "float8e8m0",
        "float4e2m1",
    ):
        assert isinstance(getattr(humming.dtypes, name), humming.dtypes.DataType)
    # All three import roots must hand out the same DataType instances, or
    # dtype-keyed lookup maps built on one root miss metadata from another.
    assert humming.dtypes.uint4 is operator_dtypes.uint4
    assert humming.dtypes.bfloat16 is operator_dtypes.bfloat16
    assert humming.dtypes.uint4 is chord.dtypes.uint4


def test_enums_match_vllm_conventions() -> None:
    from humming.config import GemmType, WeightScaleType

    assert GemmType.INDEXED.value == "indexed"
    assert GemmType.GROUPED_CONTIGUOUS.value == "grouped_contiguous"
    assert GemmType.GROUPED_MASKED.value == "grouped_masked"
    assert "group" in str(WeightScaleType.GROUP).lower()
    assert WeightScaleType.GROUP != WeightScaleType.GROUP_TENSOR


def test_weight_schema_defaults_match_chord_contract() -> None:
    from humming import dtypes
    from humming.schema import BaseWeightSchema, HummingWeightSchema

    schema = BaseWeightSchema.from_config(
        {"quant_method": "humming", "b_dtype": "uint4"}
    )
    assert isinstance(schema, HummingWeightSchema)
    assert schema.b_dtype is dtypes.uint4
    assert schema.has_zero_point is False

    # The _MoeWNA16HummingWeightSchema adapter in vLLM constructs the schema
    # directly from these keyword names; group-32 selects the GROUP scale type
    # and defaults the scale dtype to BF16.
    direct = HummingWeightSchema(
        b_dtype=dtypes.DataType.from_str("uint4"),
        weight_scale_group_size=32,
        has_zero_point=False,
    )
    from humming.config import WeightScaleType

    assert direct.weight_scale_type is WeightScaleType.GROUP
    assert direct.bs_dtype is dtypes.bfloat16
    with pytest.raises(NotImplementedError):
        direct.requant_tensors({}, direct, torch.bfloat16)
    with pytest.raises(NotImplementedError):
        BaseWeightSchema.from_config({"quant_method": "awq"})


def test_compressed_tensors_schema_from_kimi_style_config() -> None:
    """The dict shape vLLM's compressed_tensors_get_config hands out."""
    from humming.schema import BaseWeightSchema, CompressedTensorsWeightSchema

    checkpoint_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": 32,
        "num_bits": 4,
        "observer": "minmax",
        "strategy": "group",
        "symmetric": True,
        "type": "int",
    }
    schema = BaseWeightSchema.from_config(checkpoint_config)
    assert isinstance(schema, CompressedTensorsWeightSchema)

    # Per-expert tensor allocations use the humming N-major checkpoint layout.
    attrs = schema.get_padded_tensors_attrs(
        shape_n=4096, shape_k=7168, param_dtype=torch.bfloat16, num_experts=48
    )
    assert attrs["weight_packed"]["shape"] == (48, 4096, 7168 // 8)
    assert attrs["weight_packed"]["dtype"] is torch.int32
    assert attrs["weight_scale"]["shape"] == (48, 4096, 7168 // 32)
    assert attrs["weight_scale"]["dtype"] is torch.bfloat16

    from humming.schema import HummingWeightSchema

    converted_schema, tensors = schema.convert_humming(
        tensors={
            "weight_packed": torch.zeros((48, 4096, 7168 // 8), dtype=torch.int32),
            "weight_scale": torch.ones((48, 4096, 7168 // 32), dtype=torch.bfloat16),
        },
        shape_n_stacks=[4096],
        shape_k_stacks=[7168],
        param_dtype=torch.bfloat16,
        num_experts=48,
    )
    assert isinstance(converted_schema, HummingWeightSchema)
    assert converted_schema.b_dtype is not None
    assert converted_schema.b_dtype.num_bits == 4
    assert not converted_schema.b_dtype.is_signed
    assert converted_schema.weight_scale_group_size == 32
    assert converted_schema.bs_dtype is not None
    assert converted_schema.bs_dtype.num_bits == 16
    assert set(tensors) == {"weight", "weight_scale"}

    with pytest.raises(NotImplementedError):
        BaseWeightSchema.from_config(dict(checkpoint_config, symmetric=False))


def test_input_schema_is_bf16_passthrough() -> None:
    from humming import dtypes
    from humming.schema import BaseInputSchema, HummingInputSchema

    schema = HummingInputSchema()
    assert schema.a_dtype is None
    assert schema.input_scale_group_size == 0
    converted, leftover = schema.convert_humming(
        tensors={"weight": 1},
        shape_n_stacks=[1],
        shape_k_stacks=[1],
        param_dtype=None,
    )
    assert converted is schema and leftover == {}
    assert schema.get_fallback_input_dtype(dtypes.int8) is dtypes.bfloat16
    assert isinstance(
        BaseInputSchema.from_config({"quant_method": "humming"}), HummingInputSchema
    )
    with pytest.raises(NotImplementedError):
        BaseInputSchema.from_config({"quant_method": "fp8", "type": "float"})


class _HostMoeConfig:
    def __init__(self, ep_size: int = 8, tp_size: int = 1) -> None:
        self.ep_size = ep_size
        self.tensor_parallel_size = tp_size
        self.moe_parallel_config = types.SimpleNamespace(
            ep_size=ep_size, tp_size=tp_size
        )


def _packed_params(layer, sublayer: str, shape_n: int, shape_k: int, experts: int):
    setattr(
        layer,
        f"{sublayer}_weight",
        torch.nn.Parameter(
            torch.zeros((experts, shape_k // 16, shape_n * 2), dtype=torch.int32),
            requires_grad=False,
        ),
    )
    setattr(
        layer,
        f"{sublayer}_weight_scale",
        torch.nn.Parameter(
            torch.zeros((experts, shape_k // 32, shape_n), dtype=torch.bfloat16),
            requires_grad=False,
        ),
    )


def _make_host_layer(
    *,
    num_experts: int = 48,
    w13_shape: tuple[int, int] = (4096, 7168),
    w2_shape: tuple[int, int] = (7168, 2048),
) -> torch.nn.Module:
    """Simulate vLLM's RoutedExperts after kernel-layout transformation."""
    layer = torch.nn.Module()
    layer.moe_config = _HostMoeConfig()
    _packed_params(layer, "w13", *w13_shape, num_experts)
    _packed_params(layer, "w2", *w2_shape, num_experts)
    return layer


_PROFILE = "blackwell_decode_ep8"


def _prepare_both_sublayers(method, layer) -> None:
    method.prepare_layer_meta(
        layer=layer,
        shape_n=4096,
        shape_k=7168,
        num_experts=48,
        sublayer_name="w13",
        profile=_PROFILE,
    )
    method.prepare_layer_meta(
        layer=layer,
        shape_n=7168,
        shape_k=2048,
        num_experts=48,
        sublayer_name="w2",
        profile=_PROFILE,
    )


def _covering_row(rows, valid_shape_m: int) -> dict:
    for lower, upper, row in rows:
        if valid_shape_m > lower and valid_shape_m <= upper:
            return row
    raise AssertionError(f"no row covers {valid_shape_m}")


# Brackets where the Blackwell decode tables tune w13 and w2 with different
# native block_m (e.g. m in (272, 320]: gate stays at 8, down moves to 16).
_SAMPLE_M = [1, 64, 200, 272, 273, 300, 320, 321, 500, 736, 737, 1424, 1425, 3000]


def test_humming_method_aliased_from_chord_facade() -> None:
    import chord.layer as chord_layer
    from humming.layer import (
        HummingLayer,
        HummingLayerMeta,
        HummingMethod,
    )

    from chord_kernels.operator.layer import IndexedW4A16Layer, IndexedW4A16Method

    assert HummingLayer is IndexedW4A16Layer is chord_layer.HummingLayer
    assert issubclass(HummingMethod, IndexedW4A16Method)
    assert HummingLayerMeta is chord_layer.HummingLayerMeta


def test_w2_tuning_rows_follow_w13_routing_block_m() -> None:
    from humming.layer import HummingMethod

    layer = _make_host_layer()
    _prepare_both_sublayers(HummingMethod, layer)
    rows13 = HummingMethod.get_default_tuning_configs(
        layer=layer, gemm_type="indexed", sublayer_name="w13"
    )
    aligned2 = HummingMethod.get_default_tuning_configs(
        layer=layer, gemm_type="indexed", sublayer_name="w2"
    )
    # The native (unaligned) w2 table the chord router would publish on its own.
    from chord_kernels.operator.layer import IndexedW4A16Method

    native2 = IndexedW4A16Method.get_default_tuning_configs(
        layer=layer, gemm_type="indexed", sublayer_name="w2"
    )

    divergence_seen = False
    for m in _SAMPLE_M:
        row13 = _covering_row(rows13, m)
        native_row2 = _covering_row(native2, m)
        aligned_row2 = _covering_row(aligned2, m)
        if row13["block_shape"][0] != native_row2["block_shape"][0]:
            divergence_seen = True
        # Routing follows w13, so the published w2 table must agree on M...
        assert aligned_row2["block_shape"][0] == row13["block_shape"][0]
        # ...while keeping the projection's own N/K tile and stream-K choice.
        assert aligned_row2["block_shape"][1:] == native_row2["block_shape"][1:]
        assert aligned_row2["use_stream_k"] == native_row2["use_stream_k"]
    assert divergence_seen, "the sample must hit a natively divergent bracket"


def test_forward_layer_consumes_vllm_call_shape(monkeypatch) -> None:
    import chord_kernels.operator.dispatch as dispatch_module
    from humming.config import GemmType
    from humming.layer import HummingMethod

    layer = _make_host_layer()
    _prepare_both_sublayers(HummingMethod, layer)

    rows13 = HummingMethod.get_default_tuning_configs(
        layer=layer, gemm_type=GemmType.INDEXED, sublayer_name="w13"
    )
    rows2 = HummingMethod.get_default_tuning_configs(
        layer=layer, gemm_type=GemmType.INDEXED, sublayer_name="w2"
    )
    # vLLM serializes the tables once and replays them per layer call.
    tuning13 = json.dumps(rows13)
    tuning2 = json.dumps(rows2)
    compute_config = json.dumps(
        {"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "indexed"}
    )

    calls = []
    sentinel = object()

    def fake_indexed(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(dispatch_module, "w4a16_indexed", fake_indexed)

    hidden_states = torch.zeros((4, 7168), dtype=torch.bfloat16)
    act_output = torch.zeros((4, 2048), dtype=torch.bfloat16)
    sorted_ids = torch.zeros((8,), dtype=torch.int32)
    expert_ids = torch.zeros((1,), dtype=torch.int32)
    num_tokens_padded = torch.tensor([8], dtype=torch.int32)

    inputs, input_scale = HummingMethod.may_quant_input(
        layer=layer, inputs=hidden_states, quanted_input=None, sublayer_name="w13"
    )
    assert inputs is hidden_states and input_scale is None
    out = HummingMethod.forward_layer(
        layer=layer,
        inputs=inputs,
        input_scale=input_scale,
        outputs=None,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        compute_config=compute_config,
        valid_shape_m=32,
        top_k=4,
        tuning_config=tuning13,
        sublayer_name="w13",
    )
    assert out is sentinel

    out2 = HummingMethod.forward_layer(
        layer=layer,
        inputs=act_output,
        input_scale=None,
        outputs=None,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        compute_config=compute_config,
        valid_shape_m=16,
        top_k=1,
        tuning_config=tuning2,
        sublayer_name="w2",
    )
    assert out2 is sentinel

    assert len(calls) == 2
    (args13, kwargs13), (args2, kwargs2) = calls
    prepared13, prepared2 = args13[1], args2[1]
    # _parameter_tensor unwraps to Parameter.data; identity holds by storage.
    assert prepared13.packed.data_ptr() == layer.w13_weight.data_ptr()
    assert prepared13.scale.data_ptr() == layer.w13_weight_scale.data_ptr()
    assert prepared2.packed.data_ptr() == layer.w2_weight.data_ptr()
    assert args13[5] == 4 and args2[5] == 1  # top_k passed through
    assert kwargs13["validate_routing"] is False
    # The JSON tuning row keeps the routing-aligned block-M at the selected m.
    assert kwargs13["config"].block_m == _covering_row(rows13, 32)["block_shape"][0]
    assert kwargs2["config"].block_m == _covering_row(rows2, 16)["block_shape"][0]
    # This host registers no ``locks`` buffer, so the launcher supplies its own.
    assert kwargs13["locks"] is None and kwargs2["locks"] is None

    # vLLM registers one (``layer.locks``) the way upstream Humming layers do;
    # forward_layer hands it to the launch, as upstream's forward_layer does.
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32))
    calls.clear()
    HummingMethod.forward_layer(
        layer=layer,
        inputs=inputs,
        input_scale=None,
        outputs=None,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        compute_config=compute_config,
        valid_shape_m=32,
        top_k=4,
        tuning_config=tuning13,
        sublayer_name="w13",
    )
    assert calls[0][1]["locks"] is layer.locks

    # One left on another device is not handed over; the launcher's own serves.
    layer.locks = torch.zeros(1024, dtype=torch.int32, device="meta")
    calls.clear()
    HummingMethod.forward_layer(
        layer=layer,
        inputs=inputs,
        input_scale=None,
        outputs=None,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        compute_config=compute_config,
        valid_shape_m=32,
        top_k=4,
        tuning_config=tuning13,
        sublayer_name="w13",
    )
    assert calls[0][1]["locks"] is None


def test_forward_layer_rejects_unsupported_paths() -> None:
    from humming.config import GemmType
    from humming.layer import HummingMethod

    layer = _make_host_layer()
    _prepare_both_sublayers(HummingMethod, layer)

    inputs = torch.zeros((4, 7168), dtype=torch.bfloat16)
    routing = dict(
        sorted_ids=torch.zeros((8,), dtype=torch.int32),
        expert_ids=torch.zeros((1,), dtype=torch.int32),
        num_tokens_padded=torch.tensor([8], dtype=torch.int32),
        sublayer_name="w13",
    )

    with pytest.raises(ValueError, match="does not accept an input scale"):
        HummingMethod.forward_layer(
            layer=layer, inputs=inputs, input_scale=inputs, top_k=4, **routing
        )
    with pytest.raises(ValueError, match="sorted_ids/expert_ids routing only"):
        HummingMethod.forward_layer(
            layer=layer,
            inputs=inputs,
            top_k=4,
            expert_layout=torch.zeros((1,), dtype=torch.int64),
            **routing,
        )
    with pytest.raises(ValueError, match="use_batch_invariant"):
        HummingMethod.forward_layer(
            layer=layer,
            inputs=inputs,
            top_k=4,
            compute_config=json.dumps(
                {"use_batch_invariant": True, "gemm_type": "indexed"}
            ),
            **routing,
        )
    with pytest.raises(ValueError, match="indexed"):
        HummingMethod.get_default_tuning_configs(
            layer=layer, gemm_type=GemmType.GROUPED_CONTIGUOUS, sublayer_name="w13"
        )


def test_auto_profile_recovers_shard_axis_from_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prepare_layer_meta('auto') lands on TP8 for TP8 projection shapes.

    vLLM passes only shape_n/shape_k of the local partition; TP8 slices
    moe_intermediate, so (512, 7168)/(7168, 256) is unreachable by EP8.  The
    published EP8/TP8 shape pairs are disjoint, which is what makes the axis
    recoverable without a chord-specific argument.  Both are H200 tables, so
    'auto' is resolved as on SM90 whatever GPU runs the suite.
    """
    from humming.layer import HummingMethod

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_, **__: (9, 0))

    tp_layer = torch.nn.Module()
    meta = HummingMethod.prepare_layer_meta(
        layer=tp_layer, shape_n=512, shape_k=7168, num_experts=384
    )
    assert meta.profile.name == "h200_tp8"

    ep_layer = torch.nn.Module()
    meta = HummingMethod.prepare_layer_meta(
        layer=ep_layer, shape_n=4096, shape_k=7168, num_experts=48
    )
    assert meta.profile.name == "h200_prefill_ep8"

    # A stated axis contradicting a published shape is a config error, not a
    # silent fallback to either table.
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        HummingMethod.prepare_layer_meta(
            layer=torch.nn.Module(),
            shape_n=4096,
            shape_k=7168,
            num_experts=48,
            tensor_parallel_size=8,
        )


def test_humming_input_schema_used_by_moe_wna16_path() -> None:
    # vllm.../oracle/int_wna16.py instantiates HummingInputSchema() bare and
    # get_humming_moe_quant_config reads .a_dtype to decide activation quant.
    from humming.schema import HummingInputSchema

    schema = HummingInputSchema()
    assert schema.a_dtype is None or schema.a_dtype.num_bits >= 16


def _kimi_schemas():
    from humming.schema import BaseWeightSchema, HummingInputSchema

    weight_schema = BaseWeightSchema.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "type": "int",
            "num_bits": 4,
            "strategy": "group",
            "group_size": 32,
            "symmetric": True,
        }
    )
    return weight_schema, HummingInputSchema()


def _functional_config(monkeypatch, shape_n, shape_k, num_experts, capability=(9, 0)):
    """prepare_layer_config as vLLM 0.29 calls it, on a pinned capability."""
    from humming.transform import prepare_layer_config

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_, **__: capability)
    weight_schema, input_schema = _kimi_schemas()
    weight_schema, _ = weight_schema.convert_humming(
        tensors={
            "weight_packed": torch.zeros((num_experts, shape_n, shape_k // 8), dtype=torch.int32),
            "weight_scale": torch.ones((num_experts, shape_n, shape_k // 32)),
        },
        shape_n_stacks=[shape_n],
        shape_k_stacks=[shape_k],
        param_dtype=torch.bfloat16,
        num_experts=num_experts,
    )
    return prepare_layer_config(
        shape_n=shape_n,
        shape_k=shape_k,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        input_schema=input_schema,
        weight_schema=weight_schema,
        has_bias=False,
        num_experts=num_experts,
        torch_dtype=torch.bfloat16,
    )


def test_prepare_layer_config_exposes_the_fields_vllm_reads(monkeypatch) -> None:
    """vLLM sizes its workspace buffers from these LayerConfig attributes."""
    from humming import dtypes

    config = _functional_config(monkeypatch, 4096, 7168, 48)
    assert (config.shape_n, config.shape_k, config.num_experts) == (4096, 7168, 48)
    # No padding: buffers vLLM sizes as shape - pad_shape stay the real dims.
    assert config.pad_shape_n == 0 and config.pad_shape_k == 0
    assert config.a_dtype is dtypes.bfloat16 and config.c_dtype is dtypes.bfloat16
    assert config.as_dtype is None and config.input_scale_group_size == 0
    assert config.meta.profile.name == "h200_prefill_ep8"
    assert dtypes.torch_dtype_map[config.c_dtype] is torch.bfloat16

    monkeypatch.setenv("CHORD_SM90_DECODE", "1")
    assert _functional_config(monkeypatch, 4096, 7168, 48).meta.profile.name == "h200_decode_ep8"
    # The shard axis comes from the shapes, as on the layer path.
    assert _functional_config(monkeypatch, 512, 7168, 384).meta.profile.name == "h200_tp8"
    blackwell = _functional_config(monkeypatch, 4096, 7168, 48, capability=(10, 3))
    assert blackwell.meta.profile.name == "blackwell_decode_ep8"


def test_prepare_layer_config_fails_closed_outside_w4a16(monkeypatch) -> None:
    from humming import dtypes
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.transform import prepare_layer_config

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_, **__: (9, 0))

    def prepare(weight_schema, input_schema=None, num_experts=48, **kwargs):
        return prepare_layer_config(
            shape_n=4096, shape_k=7168, weight_schema=weight_schema,
            input_schema=input_schema or HummingInputSchema(),
            num_experts=num_experts, torch_dtype=torch.bfloat16, **kwargs,
        )

    group128 = HummingWeightSchema(b_dtype=dtypes.uint4, weight_scale_group_size=128)
    with pytest.raises(NotImplementedError, match="group-32"):
        prepare(group128)
    w4a16 = HummingWeightSchema(b_dtype=dtypes.uint4, weight_scale_group_size=32)
    with pytest.raises(NotImplementedError, match="BF16 activations"):
        prepare(w4a16, HummingInputSchema(a_dtype=dtypes.float8e4m3))
    with pytest.raises(NotImplementedError, match="dense"):
        prepare(w4a16, num_experts=0)
    with pytest.raises(NotImplementedError, match="bias"):
        prepare(w4a16, has_bias=True)
    monkeypatch.setenv("CHORD_USE_GROUPED", "1")
    with pytest.raises(NotImplementedError, match="grouped"):
        prepare(w4a16)


def test_get_heuristics_config_follows_w13_routing_block_m(monkeypatch) -> None:
    """vLLM routes both projections with the w13 table's block-M."""
    from humming.config import GemmType
    from humming.tune import get_heuristics_config

    w13 = _functional_config(monkeypatch, 4096, 7168, 48)
    w2 = _functional_config(monkeypatch, 7168, 2048, 48)
    rows13 = get_heuristics_config(layer_config=w13, gemm_type=GemmType.INDEXED)
    rows2 = get_heuristics_config(layer_config=w2, gemm_type=GemmType.INDEXED)
    json.dumps(rows13)  # vLLM serializes the table once.
    for routed_m in (8, 160, 1024, 4096, 16384, 1 << 20):
        r13 = _covering_row(rows13, routed_m)
        r2 = _covering_row(rows2, routed_m)
        assert r2["block_shape"][0] == r13["block_shape"][0], routed_m
        assert get_heuristics_config(w2, shape_m=routed_m, gemm_type="indexed") == r2
    for unsupported in ({"gemm_type": GemmType.GROUPED_CONTIGUOUS},
                        {"gemm_type": "indexed", "use_f16_accum": True},
                        {"gemm_type": "indexed", "use_batch_invariant": True}):
        with pytest.raises(NotImplementedError):
            get_heuristics_config(layer_config=w13, **unsupported)


def test_humming_forward_uses_chord_tile_at_the_routing_block_m(monkeypatch) -> None:
    """Only block-M is taken from the framework's (edited) row; the tile is
    chord's own, and the experts' lock buffer reaches the launch."""
    import chord_kernels.operator.dispatch as dispatch_module
    from humming.forward import humming_forward, may_quant_input
    from humming.tune import get_heuristics_config

    w13 = _functional_config(monkeypatch, 4096, 7168, 48)
    rows = get_heuristics_config(layer_config=w13, gemm_type="indexed")
    edited = json.loads(json.dumps(rows))
    for _, _, row in edited:  # what vLLM's _fixup_moe_tuning_config does
        row["block_shape"][2] = min(row["block_shape"][2], 128)
        row["warp_shape"][1] = 32
    calls = []
    monkeypatch.setattr(
        dispatch_module, "w4a16_indexed",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "out",
    )
    inputs = torch.zeros((20, 7168), dtype=torch.bfloat16)
    assert may_quant_input(w13, inputs=inputs, input_scale=None) == (inputs, None)
    locks = torch.zeros(1024, dtype=torch.int32)
    out = humming_forward(
        w13,
        inputs=inputs,
        weight=torch.zeros((48, 7168 // 16, 4096 * 2), dtype=torch.int32),
        weight_scale=torch.zeros((48, 7168 // 32, 4096), dtype=torch.bfloat16),
        outputs=None,
        locks=locks,
        sorted_ids=torch.zeros((64,), dtype=torch.int32),
        expert_ids=torch.zeros((8,), dtype=torch.int32),
        num_tokens_padded=torch.tensor([64], dtype=torch.int32),
        compute_config=json.dumps({"gemm_type": "indexed"}),
        valid_shape_m=160,
        top_k=8,
        tuning_config=json.dumps(edited),
    )
    assert out == "out"
    (args, kwargs), = calls
    expected = w13.meta.kernel_config(160)
    assert kwargs["config"].block_shape == expected.block_shape
    assert kwargs["config"].block_m == _covering_row(edited, 160)["block_shape"][0]
    assert kwargs["locks"] is locks and kwargs["validate_routing"] is False
    with pytest.raises(NotImplementedError):
        may_quant_input(w13, inputs=inputs, input_scale=torch.ones(1))
