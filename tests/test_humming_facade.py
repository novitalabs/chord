"""Contract tests for the ``chord`` humming-compatible facade.

These lock the drop-in surface a serving framework's Humming integration
touches, so renaming something inside ``chord_kernels`` fails here rather than
at model load time in vLLM.
"""

from __future__ import annotations

import pytest
import torch

import chord
import chord.dtypes
import chord.ops
from chord import config as chord_config
from chord import layer as chord_layer
from chord_kernels.operator.layer import IndexedW4A16Layer, IndexedW4A16Method

# The names vLLM's Humming adapter imports from ``humming.layer``.
_LAYER_NAMES = (
    "HummingLayer",
    "HummingLayerMeta",
    "HummingLayerMethod",
    "HummingMethod",
    "HummingModule",
)

# The classmethod delegation contract on the method class.
_METHOD_API = (
    "prepare_layer_meta",
    "transform_humming_layer",
    "forward_layer",
    "get_default_tuning_configs",
    "may_quant_input",
    "may_hadamard_quant_input",
    "may_set_param",
)


def test_layer_module_exports_humming_names() -> None:
    for name in _LAYER_NAMES:
        assert hasattr(chord_layer, name), f"chord.layer is missing {name}"


def test_humming_names_alias_the_real_classes() -> None:
    assert chord_layer.HummingLayer is IndexedW4A16Layer
    assert chord_layer.HummingLayerMethod is IndexedW4A16Method
    assert chord_layer.HummingMethod is IndexedW4A16Method
    assert chord_layer.HummingModule is torch.nn.Module


@pytest.mark.parametrize("name", _METHOD_API)
def test_method_delegation_contract(name: str) -> None:
    attr = getattr(chord_layer.HummingLayerMethod, name, None)
    assert attr is not None, f"HummingLayerMethod is missing {name}"
    assert callable(attr)


def test_layer_exposes_humming_metas() -> None:
    layer = chord_layer.HummingLayer(
        shape_n=128, shape_k=64, num_experts=1, profile="h200_decode_ep8"
    )
    assert hasattr(layer, "humming_metas")
    assert isinstance(layer.humming_metas, dict)


def test_prepare_layer_meta_returns_upstream_shaped_meta() -> None:
    layer = chord_layer.HummingLayer(
        shape_n=128, shape_k=64, num_experts=1, profile="h200_decode_ep8"
    )
    meta = chord_layer.HummingLayerMethod.prepare_layer_meta(
        layer, shape_n=128, shape_k=64, num_experts=1, sublayer_name="w13"
    )
    assert isinstance(meta, chord_layer.HummingLayerMeta)
    assert meta.weight_name == "w13_weight"
    assert meta.weight_scale_name == "w13_weight_scale"


def test_prepare_layer_meta_registers_under_sublayer_name() -> None:
    """The adapter reads back its metas from the layer, keyed per sublayer."""
    layer = torch.nn.Module()
    metas = {
        sub: chord_layer.HummingLayerMethod.prepare_layer_meta(
            layer, shape_n=n, shape_k=k, num_experts=8, sublayer_name=sub
        )
        for sub, (n, k) in {"w13": (512, 1024), "w2": (1024, 256)}.items()
    }
    assert set(layer.humming_metas) == {"w13", "w2"}
    assert layer.humming_metas["w13"] is metas["w13"]
    assert (metas["w2"].shape_n, metas["w2"].shape_k) == (1024, 256)


def test_tensors_attrs_give_adapter_checkpoint_shapes() -> None:
    """An adapter allocates its checkpoint parameters from these attrs.

    This runs before any weight transform, so the shapes here are what the
    framework registers -- they must match the packed layout
    ``transform_humming_layer`` later validates.
    """
    layer = torch.nn.Module()
    meta = chord_layer.HummingLayerMethod.prepare_layer_meta(
        layer, shape_n=512, shape_k=1024, num_experts=8, sublayer_name="w13"
    )
    attrs = meta.get_tensors_attrs()

    assert attrs["weight"]["shape"] == (8, 512, 1024 // 8)
    assert attrs["weight"]["dtype"] is torch.int32
    assert attrs["weight"]["extra_attrs"]["packed_factor"] == 8
    assert attrs["weight_scale"]["shape"] == (8, 512, 32)
    assert attrs["weight_scale"]["dtype"] is torch.bfloat16

    # The reported specs really are allocatable as parameters.
    for spec in attrs.values():
        torch.empty(spec["shape"], dtype=spec["dtype"])


def test_ops_module_exports_launcher_bootstrap() -> None:
    assert callable(chord.ops.init_humming_launcher)
    assert chord.ops.init_humming_launcher is chord.ops.init_launcher


def test_unpack_weight_rejects_out_of_scope_bit_width() -> None:
    with pytest.raises(ValueError, match="num_bits=4"):
        chord.ops.unpack_weight(torch.zeros((1, 4), dtype=torch.int32), num_bits=8)


def test_dtypes_expose_w4a16_schema() -> None:
    assert chord.dtypes.torch_dtype_map[chord.dtypes.bfloat16] is torch.bfloat16
    assert chord.dtypes.uint4.num_bits == 4


def test_default_f16_dtype_is_bfloat16() -> None:
    assert chord_layer.get_default_f16_torch_dtype() is torch.bfloat16


class TestShardAxisReachability:
    """TP8 must be reachable from an adapter that passes only upstream args.

    A framework adapter calls ``prepare_layer_meta`` with the shapes it always
    passes and no chord-specific arguments.  Since each shard axis has its own
    tuned schedule, landing on the wrong one would silently pack the weight
    against the wrong table -- so these pin the shape-driven axis recovery.
    """

    @staticmethod
    def _meta(shape_n: int, shape_k: int, num_experts: int, **kwargs):
        return chord_layer.HummingLayerMethod.prepare_layer_meta(
            torch.nn.Module(),
            shape_n=shape_n,
            shape_k=shape_k,
            num_experts=num_experts,
            **kwargs,
        )

    @pytest.mark.parametrize(("shape_n", "shape_k"), [(512, 7168), (7168, 256)])
    def test_tp8_shapes_reach_the_tp8_profile(
        self, shape_n: int, shape_k: int
    ) -> None:
        meta = self._meta(shape_n, shape_k, 384)
        assert meta.profile.name == "h200_tp8"
        assert meta.profile.shard_axis == "tp"
        assert meta.profile.mode == "mix"

    @pytest.mark.parametrize(("shape_n", "shape_k"), [(4096, 7168), (7168, 2048)])
    def test_ep8_shapes_keep_the_ep8_profile(
        self, shape_n: int, shape_k: int
    ) -> None:
        meta = self._meta(shape_n, shape_k, 32)
        assert meta.profile.shard_axis == "ep"
        assert meta.profile.expert_parallel_size == 8

    def test_unpublished_shapes_default_to_ep8(self) -> None:
        """An unrecognised shape must not be guessed into a tuned profile."""
        meta = self._meta(2048, 2048, 32)
        assert meta.profile.shard_axis == "ep"

    def test_explicit_size_selects_tp8_for_unpublished_shapes(self) -> None:
        """A stated axis still works for a model chord has no table for."""
        meta = self._meta(2048, 2048, 32, tensor_parallel_size=8)
        assert meta.profile.name == "h200_tp8"

    def test_explicit_profile_overrides_the_inferred_axis(self) -> None:
        meta = self._meta(512, 7168, 384, profile="h200_prefill_ep8")
        assert meta.profile.name == "h200_prefill_ep8"

    @pytest.mark.parametrize(
        ("shape_n", "shape_k", "tensor_parallel_size"),
        [(4096, 7168, 8), (512, 7168, 1)],
    )
    def test_size_contradicting_published_shapes_is_rejected(
        self, shape_n: int, shape_k: int, tensor_parallel_size: int
    ) -> None:
        """Silently honouring either side would use the wrong tuned table."""
        with pytest.raises(ValueError, match="shard axis"):
            self._meta(
                shape_n,
                shape_k,
                384,
                tensor_parallel_size=tensor_parallel_size,
            )

    def test_tp8_ignores_the_pd_role_bit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mix profile serves both phases, so CHORD_SM90_DECODE cannot apply."""
        monkeypatch.setenv("CHORD_SM90_DECODE", "1")
        assert self._meta(512, 7168, 384).profile.name == "h200_tp8"

    def test_tp8_ignores_the_grouped_switch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each grouped kernel serves one phase, so none can back a mix weight."""
        monkeypatch.setenv("CHORD_USE_GROUPED", "1")
        meta = self._meta(512, 7168, 384)
        assert meta.profile.name == "h200_tp8"
        assert meta.backend == "indexed"

    def test_tp8_metadata_selects_the_tp8_tuning_table(self) -> None:
        """The point of the axis: TP8 shapes must get TP8's schedule.

        Compared at equal tokens-per-expert so the two tables are asked the same
        question.  TP8's windows are flatter (capped at block-M 128) than EP8's
        padding model, so a TP8 layer landing on the EP8 profile would launch a
        tile its shapes were never tuned for.
        """
        tp8 = self._meta(512, 7168, 384)
        ep8 = self._meta(4096, 7168, 384)
        routed_m = 384 * 300
        assert tp8.kernel_config(routed_m).block_m == 128
        assert ep8.kernel_config(routed_m).block_m == 168


class TestBackendPolicy:
    """``sm90_w4a16_decode_backend`` must report chord's own constants."""

    @staticmethod
    def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "CHORD_SM90_DECODE",
            "CHORD_USE_GROUPED",
            "CHORD_SM90_EP8_MIN_EXPERTS",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_prefill_native(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear(monkeypatch)
        assert (
            chord_config.sm90_w4a16_decode_backend(num_experts=48)
            == chord_config.SM90_W4A16_WGMMA
        )

    def test_decode_native_ep8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("CHORD_SM90_DECODE", "1")
        assert (
            chord_config.sm90_w4a16_decode_backend(num_experts=48)
            == chord_config.SM90_W4A16_SWAP_AB
        )

    def test_decode_native_below_ep8_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("CHORD_SM90_DECODE", "1")
        assert (
            chord_config.sm90_w4a16_decode_backend(num_experts=8)
            == chord_config.SM90_W4A16_WGMMA
        )

    def test_decode_grouped_is_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("CHORD_SM90_DECODE", "1")
        monkeypatch.setenv("CHORD_USE_GROUPED", "1")
        assert (
            chord_config.sm90_w4a16_decode_backend(num_experts=48)
            == chord_config.SM90_W4A16_DEEPGEMM_MASKED
        )

    def test_prefill_grouped_is_contiguous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("CHORD_USE_GROUPED", "1")
        assert (
            chord_config.sm90_w4a16_decode_backend(num_experts=48)
            == chord_config.SM90_W4A16_DEEPGEMM_CONTIGUOUS
        )

    def test_out_of_scope_signature_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear(monkeypatch)
        monkeypatch.setenv("CHORD_USE_GROUPED", "1")
        assert (
            chord_config.sm90_w4a16_decode_backend(num_experts=48, b_num_bits=8)
            == chord_config.SM90_W4A16_WGMMA
        )
