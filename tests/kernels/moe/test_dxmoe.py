# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts and single-GPU real vLLM + CUDA Graph tests for DxMoE (ABI 12)."""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from vllm.config import (
    CUDAGraphMode,
    KernelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.dxmoe_experts import (
    DxMoEExperts,
    DxMoELayerRuntimeState,
    DxMoEMoEQuantConfig,
)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.dxmoe import (
    DxMoEConfig,
    DxMoEMoEMethod,
    prepare_shared_ffn_rotation,
)
from vllm.utils import dxmoe as vllm_dxmoe


def _sample_dxmoe_meta(
    *,
    layers: tuple[int, ...] = (1, 2),
    hidden_size: int = 256,
    inter_size: int = 256,
    num_experts: int = 4,
    top_k: int = 2,
    w_bits: tuple[int, ...] = (4, 8, 4, 8),
    hadamard_mode: str = "none",
    tau_entry: tuple[float, float] = (0.05, 0.5),
    tau_down: tuple[float, float] = (0.08, 0.8),
) -> dict:
    layers_dict = {}
    for l_idx in layers:
        experts = {}
        for e_idx in range(num_experts):
            bw = w_bits[e_idx]
            experts[str(e_idx)] = {
                "gate": {"w_bits": bw, "beta": 0.5 + 0.1 * e_idx},
                "up": {"w_bits": bw, "beta": 0.5 + 0.1 * e_idx},
                "entry": {"w_bits": bw, "beta": 0.5 + 0.1 * e_idx},
                "down": {"w_bits": bw, "beta": 0.7 + 0.1 * e_idx},
            }
        layers_dict[str(l_idx)] = {
            "hidden_size": hidden_size,
            "moe_intermediate_size": inter_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "experts": experts,
            "thresholds": {
                "5.0": {
                    "tau_entry": list(tau_entry),
                    "tau_down": list(tau_down),
                }
            },
        }
    return {
        "format": "scaling-moe.dxmoe.meta",
        "version": 1,
        "quantization": {"group_size": -1, "sym": True},
        "hadamard": {
            "mode": hadamard_mode,
            "block_size": 128 if hadamard_mode == "block128" else 0,
            "signs_file": "dxmoe_signs.safetensors" if hadamard_mode == "block128" else None,
        },
        "layers": layers_dict,
    }


def _make_moe_config(
    *,
    num_experts: int = 4,
    top_k: int = 2,
    hidden_size: int = 256,
    inter_size: int = 256,
    moe_backend: str = "dxmoe",
    routing_method: RoutingMethodType = RoutingMethodType.RenormalizeNaive,
    activation: MoEActivation = MoEActivation.SILU,
    swiglu_limit: float | None = None,
) -> FusedMoEConfig:
    p_cfg = FusedMoEParallelConfig.make_no_parallel()
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=inter_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=activation,
        device="cpu",
        routing_method=routing_method,
        moe_parallel_config=p_cfg,
        in_dtype=torch.bfloat16,
        moe_backend=moe_backend,  # type: ignore[arg-type]
        swiglu_limit=swiglu_limit,
    )


def test_dxmoe_abi12_rejects_mismatched_dso_before_arch_or_prepare(
    tmp_path: Path,
) -> None:
    assert vllm_dxmoe.DXMOE_ABI_VERSION == 12
    dummy_so = tmp_path / "libsmoe_dxmoe_short.so"
    dummy_so.write_bytes(b"stub")

    for bad_abi in (0, 9, 10, 11, 13):
        calls: list[str] = []

        class _FakeLib:
            def __init__(self) -> None:
                self.smoe_dxmoe_abi_version = self._make("abi", bad_abi)
                self.smoe_dxmoe_arch = self._make("arch", 80)
                self.smoe_dxmoe_prepare = self._make("prepare", 0)
                self.smoe_dxmoe_fused_run = self._make("fused_run", 0)

            @staticmethod
            def _make(name: str, ret: int):
                def _fn(*_args):
                    calls.append(name)
                    return ret

                _fn.argtypes = None
                _fn.restype = None
                return _fn

        with (
            patch.object(ctypes, "CDLL", return_value=_FakeLib()),
            pytest.raises(
                RuntimeError,
                match=rf"DxMoE ABI mismatch.*expected 12, got {bad_abi}",
            ),
        ):
            vllm_dxmoe._bind_dxmoe_dso(dummy_so, expected_arch=80)
        assert calls == ["abi"]


def test_dxmoe_abi12_descriptor_sizes_and_key_offsets() -> None:
    assert ctypes.sizeof(vllm_dxmoe.NativeReorderDesc) == 296
    assert vllm_dxmoe.NativeReorderDesc.in_row_to_call.offset == 56
    assert vllm_dxmoe.NativeReorderDesc.out_row_to_call.offset == 64
    assert vllm_dxmoe.NativeReorderDesc.out_call_to_row.offset == 72
    assert vllm_dxmoe.NativeReorderDesc.pool_data.offset == 80
    assert vllm_dxmoe.NativeReorderDesc.workspace.offset == 208
    assert vllm_dxmoe.NativeReorderDesc.is_padding.offset == 240
    assert vllm_dxmoe.NativeReorderDesc.num_tokens.offset == 248
    assert vllm_dxmoe.NativeReorderDesc.compact_outputs.offset == 284
    assert vllm_dxmoe.NativeReorderDesc.num_valid_tokens.offset == 288
    assert vllm_dxmoe.NativeReorderDesc.preserve_prior_status.offset == 292

    assert ctypes.sizeof(vllm_dxmoe.NativeOmegaDesc) == 80
    assert vllm_dxmoe.NativeOmegaDesc.batch_size.offset == 48
    assert vllm_dxmoe.NativeOmegaDesc.mask_past_seq_end.offset == 68
    assert vllm_dxmoe.NativeOmegaDesc.pos_dtype.offset == 72

    assert ctypes.sizeof(vllm_dxmoe.NativePolicyDesc) == 208
    assert vllm_dxmoe.NativePolicyDesc.call_indices.offset == 40
    assert vllm_dxmoe.NativePolicyDesc.out_expert_ids.offset == 96
    assert vllm_dxmoe.NativePolicyDesc.is_padding.offset == 128
    assert vllm_dxmoe.NativePolicyDesc.tau1.offset == 136
    assert vllm_dxmoe.NativePolicyDesc.scalar_omega.offset == 152
    assert vllm_dxmoe.NativePolicyDesc.routing_dtype.offset == 196
    assert vllm_dxmoe.NativePolicyDesc.num_valid_tokens.offset == 200

    assert ctypes.sizeof(vllm_dxmoe.NativeUnpermuteDesc) == 80
    assert vllm_dxmoe.NativeUnpermuteDesc.status.offset == 40
    assert vllm_dxmoe.NativeUnpermuteDesc.sticky_status.offset == 48
    assert vllm_dxmoe.NativeUnpermuteDesc.num_tokens.offset == 56
    assert vllm_dxmoe.NativeUnpermuteDesc.out_dtype.offset == 76


def test_unquantized_w16_moe_rejects_explicit_dxmoe_backend() -> None:
    moe_cfg = _make_moe_config(moe_backend="dxmoe")
    with pytest.raises(ValueError, match="moe_backend='dxmoe' is not supported for unquantized MoE"):
        select_unquantized_moe_backend(moe_cfg)


def test_dxmoe_config_rejects_missing_layer_w16_w8a4_and_invalid_deployment() -> None:
    from vllm.model_executor.layers.linear import RowParallelLinear

    base_meta = _sample_dxmoe_meta(layers=(1,))

    # 1) Missing RoutedExperts layer MUST raise ValueError (no UnquantizedFusedMoEMethod fallback!).
    cfg = DxMoEConfig(base_meta)
    dummy_routed = RoutedExperts.__new__(RoutedExperts)
    dummy_routed.moe_config = _make_moe_config()
    assert isinstance(cfg.get_quant_method(dummy_routed, "model.layers.1.mlp.experts"), DxMoEMoEMethod)
    with pytest.raises(ValueError, match="missing RoutedExperts layer 2"):
        cfg.get_quant_method(dummy_routed, "model.layers.2.mlp.experts")
    with pytest.raises(ValueError, match="cannot parse layer index"):
        cfg.get_quant_method(dummy_routed, "model.mlp.experts")

    # Dense / shared LinearBase layers still cleanly get UnquantizedLinearMethod.
    dummy_linear = RowParallelLinear.__new__(RowParallelLinear)
    assert isinstance(cfg.get_quant_method(dummy_linear, "model.layers.1.self_attn.q_proj"), UnquantizedLinearMethod)

    # 2) Accept W2/W3/W4/W8 and reject unsupported static weight bitwidths (e.g. W1, W5, W6, W16).
    ok_w2348 = _sample_dxmoe_meta(w_bits=(2, 3, 4, 8))
    resolved_ok = DxMoEConfig(ok_w2348)._ensure_resolved()
    assert resolved_ok[1].w_bits_entry == (2, 3, 4, 8)
    assert resolved_ok[1].w_bits_down == (2, 3, 4, 8)

    for bad_bw in (1, 5, 6, 16):
        bad_meta = _sample_dxmoe_meta(w_bits=(4, bad_bw, 4, 8))
        with pytest.raises(ValueError, match="supports W2/W3/W4/W8 static weights only"):
            DxMoEConfig(bad_meta)._ensure_resolved()

    # 3) Reject W8 + apply_coupling=False in BOTH static_a4 AND dxmoe (tau1 > 0).
    for mode in ("static_a4", "dxmoe"):
        m_w8a4 = _sample_dxmoe_meta(w_bits=(4, 8, 4, 8), tau_entry=(0.05, 0.5))
        m_w8a4["schedule_mode"] = mode
        m_w8a4["apply_coupling"] = False
        with pytest.raises(ValueError, match="W8A4"):
            DxMoEConfig(m_w8a4)._ensure_resolved()

    # 4) Reject NaN / Inf / negative beta and NaN / Inf / negative / inverted tau.
    for bad_beta in (float("nan"), float("inf"), -0.5):
        m_b = copy.deepcopy(base_meta)
        m_b["layers"]["1"]["experts"]["0"]["entry"]["beta"] = bad_beta
        with pytest.raises(ValueError, match="beta must be finite and non-negative"):
            DxMoEConfig(m_b)._ensure_resolved()

    for bad_tau in ([float("nan"), 1.0], [0.0, float("inf")], [-0.1, 1.0], [1.5, 0.5]):
        m_t = copy.deepcopy(base_meta)
        m_t["layers"]["1"]["thresholds"]["5.0"]["tau_entry"] = bad_tau
        with pytest.raises(ValueError, match="invalid tau_entry"):
            DxMoEConfig(m_t)._ensure_resolved()

    # 5) Startup deployment checks: reject max_num_seqs != 1, tp > 1, and moe_backend='triton'.
    bad_sched_cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_model_len=256, is_encoder_decoder=False
        )
    )
    with set_current_vllm_config(bad_sched_cfg):
        with pytest.raises(ValueError, match="max_num_seqs == 1"):
            DxMoEConfig(base_meta)._ensure_resolved()

    bad_tp_cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_model_len=256, is_encoder_decoder=False
        ),
    )
    bad_tp_cfg.parallel_config.tensor_parallel_size = 2
    with set_current_vllm_config(bad_tp_cfg):
        with pytest.raises(ValueError, match="single-GPU execution"):
            DxMoEConfig(base_meta)._ensure_resolved()

    bad_backend_cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_model_len=256, is_encoder_decoder=False
        ),
        kernel_config=KernelConfig(moe_backend="triton"),
    )
    with set_current_vllm_config(bad_backend_cfg):
        with pytest.raises(ValueError, match="moe_backend in \\('auto', 'dxmoe'\\)"):
            DxMoEConfig(base_meta)._ensure_resolved()


def test_dxmoe_shared_ffn_rotation_is_explicit_and_uses_shared_mid_signs(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "rotated"
    model_dir.mkdir()
    meta = _sample_dxmoe_meta(
        layers=(1,), hidden_size=256, inter_size=128, hadamard_mode="block128"
    )
    meta["layers"]["1"]["shared_ffn_rotated"] = True
    (model_dir / "dxmoe_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    gen = torch.Generator(device="cpu").manual_seed(123)
    s_in = torch.where(
        torch.randint(0, 2, (256,), generator=gen) == 0, -1, 1
    ).to(torch.int8)
    s_mid = torch.where(
        torch.randint(0, 2, (128,), generator=gen) == 0, -1, 1
    ).to(torch.int8)
    s_mid_shared = torch.where(
        torch.randint(0, 2, (256,), generator=gen) == 0, -1, 1
    ).to(torch.int8)
    save_file(
        {
            "layer.1.s_in": s_in,
            "layer.1.s_mid": s_mid,
            "layer.1.s_mid_shared": s_mid_shared,
        },
        str(model_dir / "dxmoe_signs.safetensors"),
    )

    rotation = prepare_shared_ffn_rotation(
        DxMoEConfig({"_model_dir": str(model_dir)}),
        "model.layers.1.mlp.shared_experts",
        hidden_size=256,
        intermediate_size=256,
    )
    assert rotation is not None
    x = torch.randn((3, 256), generator=gen, dtype=torch.bfloat16)
    actual = rotation.intermediate(x)
    base = torch.ones((1, 1), dtype=torch.float64)
    while base.shape[0] < 128:
        base = torch.cat(
            (torch.cat((base, base), dim=1), torch.cat((base, -base), dim=1)),
            dim=0,
        )
    had = (base / (128.0**0.5)).to(torch.float32)
    expected = (
        (x.float() * s_mid_shared.float()).reshape(3, 2, 128) @ had
    ).reshape_as(x).to(x.dtype)
    assert torch.equal(actual, expected)

    # Qwen's scalar gate must still observe the original, unrotated input.
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    class _PairLinear(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("weight", weight)

        def forward(self, value: torch.Tensor):
            return F.linear(value, self.weight), None

    class _SiluAndMul(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            gate, up = value.chunk(2, dim=-1)
            return F.silu(gate) * up

    class _RecordingGate(_PairLinear):
        last_input: torch.Tensor | None = None

        def forward(self, value: torch.Tensor):
            self.last_input = value
            return super().forward(value)

    qwen = Qwen2MoeMLP.__new__(Qwen2MoeMLP)
    torch.nn.Module.__init__(qwen)
    qwen.gate_up_proj = _PairLinear(torch.randn((512, 256), generator=gen))
    qwen.down_proj = _PairLinear(torch.randn((256, 256), generator=gen))
    qwen.act_fn = _SiluAndMul()
    qwen.expert_gate = _RecordingGate(torch.randn((1, 256), generator=gen))
    qwen._dxmoe_shared_rotation = rotation
    x_f32 = torch.randn((2, 256), generator=gen)
    qwen_out = qwen(x_f32)
    x_rot = rotation.input(x_f32)
    gate_up = F.linear(x_rot, qwen.gate_up_proj.weight)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden_rot = rotation.intermediate(F.silu(gate) * up)
    expected_qwen = F.linear(hidden_rot, qwen.down_proj.weight)
    expected_qwen *= torch.sigmoid(F.linear(x_f32, qwen.expert_gate.weight))
    assert torch.equal(qwen_out, expected_qwen)
    assert qwen.expert_gate.last_input is x_f32

    legacy_meta = copy.deepcopy(meta)
    legacy_meta["layers"]["1"].pop("shared_ffn_rotated")
    (model_dir / "dxmoe_meta.json").write_text(
        json.dumps(legacy_meta), encoding="utf-8"
    )
    legacy = prepare_shared_ffn_rotation(
        DxMoEConfig({"_model_dir": str(model_dir)}),
        "model.layers.1.mlp.shared_experts",
        hidden_size=256,
        intermediate_size=256,
    )
    assert legacy is None

    save_file(
        {"layer.1.s_in": s_in, "layer.1.s_mid": s_mid},
        str(model_dir / "dxmoe_signs.safetensors"),
    )
    (model_dir / "dxmoe_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="s_mid_shared is missing"):
        prepare_shared_ffn_rotation(
            DxMoEConfig({"_model_dir": str(model_dir)}),
            "model.layers.1.mlp.shared_experts",
            hidden_size=256,
            intermediate_size=256,
        )

def test_dxmoe_method_rejects_topk_mismatch_and_nonstandard_routers() -> None:
    spec = DxMoEConfig(_sample_dxmoe_meta(layers=(1,), top_k=2))._ensure_resolved()[1]

    def _make_dummy_layer(moe_cfg: FusedMoEConfig, **overrides) -> RoutedExperts:
        lyr = RoutedExperts.__new__(RoutedExperts)
        lyr.moe_config = moe_cfg
        lyr.top_k = moe_cfg.experts_per_token
        lyr.scoring_func = "softmax"
        lyr.use_grouped_topk = False
        lyr.num_expert_group = None
        lyr.topk_group = None
        lyr.e_score_correction_bias = None
        lyr.custom_routing_function = None
        lyr.apply_router_weight_on_input = False
        for k, v in overrides.items():
            setattr(lyr, k, v)
        return lyr

    # top_k mismatch (experts_per_token=4 vs spec.top_k=2)
    m_bad_k = _make_moe_config(top_k=4)
    method_bad_k = DxMoEMoEMethod(spec, m_bad_k)
    with pytest.raises(ValueError, match="top_k mismatch"):
        method_bad_k._validate_layer_contract(_make_dummy_layer(m_bad_k))

    # Non-standard routers: sigmoid, grouped topk, correction bias, custom router, swiglu_limit
    base_moe = _make_moe_config(top_k=2)
    method_ok = DxMoEMoEMethod(spec, base_moe)
    method_ok._validate_layer_contract(_make_dummy_layer(base_moe))
    # Degenerate n_group=1, topk_group=1 (DeepSeek-V2-Lite) stays valid global Softmax TopK
    method_ok._validate_layer_contract(
        _make_dummy_layer(base_moe, use_grouped_topk=True, num_expert_group=1, topk_group=1)
    )

    for ov_kwargs in (
        {"scoring_func": "sigmoid"},
        {"use_grouped_topk": True, "num_expert_group": 4, "topk_group": 2},
        {"use_grouped_topk": True, "num_expert_group": 1, "topk_group": 2},
        {"e_score_correction_bias": torch.zeros(4)},
        {"custom_routing_function": lambda *a: a},
        {"apply_router_weight_on_input": True},
    ):
        with pytest.raises(ValueError, match="unsupported MoE routing"):
            method_ok._validate_layer_contract(_make_dummy_layer(base_moe, **ov_kwargs))

    m_sigmoid = _make_moe_config(top_k=2, routing_method=RoutingMethodType.Sigmoid)
    with pytest.raises(ValueError, match="unsupported MoE routing"):
        DxMoEMoEMethod(spec, m_sigmoid)._validate_layer_contract(_make_dummy_layer(m_sigmoid))

    m_clamp = _make_moe_config(top_k=2, swiglu_limit=7.0)
    with pytest.raises(ValueError, match="unclamped gated SiLU-and-mul"):
        DxMoEMoEMethod(spec, m_clamp)._validate_layer_contract(_make_dummy_layer(m_clamp))


def test_dxmoe_external_out_thresholds_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Base metadata without calibrated thresholds (`thresholds={}` as exported by quantize_dxmoe_hf.py)
    meta = _sample_dxmoe_meta(layers=(1, 2))
    for lk in meta["layers"]:
        meta["layers"][lk]["thresholds"] = {}

    # Exact schema written by `scripts/calibrate_dxmoe_thresholds.py --out-thresholds`
    out_thresholds_path = tmp_path / "calibrated_thresholds.json"
    out_thresholds_payload = {
        "format": "scaling-moe.dxmoe.thresholds",
        "version": 1,
        "model": "/tmp/mock_dxmoe_model",
        "elapsed_s": 1.23,
        "threshold_calibration": {"threshold_pool": "per_stage", "quantile_mode": "effective"},
        "layers": {
            "1": {
                "thresholds": {
                    "5.0": {"tau_entry": [0.125, 1.5], "tau_down": [0.25, 2.75]},
                }
            },
            "2": {
                "thresholds": {
                    "5.0": {"tau_entry": [0.0625, 0.875], "tau_down": [0.5, 3.25]},
                }
            },
        },
    }
    out_thresholds_path.write_text(json.dumps(out_thresholds_payload), encoding="utf-8")

    monkeypatch.setenv("VLLM_DXMOE_THRESHOLDS_JSON", str(out_thresholds_path))
    monkeypatch.setenv("VLLM_DXMOE_THRESHOLD_KEY", "5.0")
    specs = DxMoEConfig(meta)._ensure_resolved()
    assert specs[1].tau_entry == (0.125, 1.5)
    assert specs[1].tau_down == (0.25, 2.75)
    assert specs[2].tau_entry == (0.0625, 0.875)
    assert specs[2].tau_down == (0.5, 3.25)

    # Corrupt format header -> rejected immediately
    bad_path = tmp_path / "bad_thresholds.json"
    bad_doc = copy.deepcopy(out_thresholds_payload)
    bad_doc["format"] = "wrong.format"
    bad_path.write_text(json.dumps(bad_doc), encoding="utf-8")
    monkeypatch.setenv("VLLM_DXMOE_THRESHOLDS_JSON", str(bad_path))
    with pytest.raises(ValueError, match="scaling-moe.dxmoe.thresholds"):
        DxMoEConfig(meta)._ensure_resolved()


def _hadamard128_ref(
    x: torch.Tensor, signs: torch.Tensor, *, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    target_dtype = x.dtype if out_dtype is None else out_dtype
    v = (x.float() * signs.float()).reshape(*x.shape[:-1], -1, 128)
    d = 1
    while d < 128:
        v = v.reshape(*v.shape[:-1], -1, 2, d)
        a = v[..., 0, :]
        b = v[..., 1, :]
        v = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], -1, 128)
        d *= 2
    return (v.reshape_as(x) * (128.0**-0.5)).to(dtype=target_dtype)


def _fake_quant_act_row(row_f32: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 16:
        return row_f32.to(torch.float16).float()
    qmax = 7.0 if bits == 4 else 127.0
    amax = float(row_f32.abs().amax().item())
    if amax <= 0.0:
        return torch.zeros_like(row_f32)
    scale = amax / qmax
    inv_scale = qmax / amax
    q = torch.round(row_f32 * inv_scale).clamp(-qmax, qmax)
    return q * scale


def _dequant_packed_w(packed: vllm_dxmoe.DxMoELayerWeights, expert: int, w_bits: int) -> torch.Tensor:
    n, k = packed.n, packed.k
    row_bytes = vllm_dxmoe.dxmoe_packed_row_bytes(k, w_bits)
    base_off = int(packed.w_ptrs_dev[expert].item()) - int(packed.codes_arena.data_ptr())
    raw = packed.codes_arena[base_off : base_off + n * row_bytes].reshape(n, row_bytes)
    if w_bits == 2:
        vals = torch.stack(
            ((raw >> 0) & 0x03, (raw >> 2) & 0x03, (raw >> 4) & 0x03, (raw >> 6) & 0x03),
            dim=-1,
        ).reshape(n, k)
        q = (vals.to(torch.int16) - 2).float()
    elif w_bits == 3:
        tri = raw.reshape(n, k // 8, 3).to(torch.int32)
        u24 = tri[:, :, 0] | (tri[:, :, 1] << 8) | (tri[:, :, 2] << 16)
        vals = torch.stack([(u24 >> (3 * i)) & 0x07 for i in range(8)], dim=-1).reshape(n, k)
        q = (vals.to(torch.int16) - 4).float()
    elif w_bits == 4:
        lo = (raw & 0x0F).to(torch.int16) - 8
        hi = (raw >> 4).to(torch.int16) - 8
        q = torch.stack((lo, hi), dim=-1).reshape(n, k).float()
    else:
        assert w_bits == 8
        q = (raw.to(torch.int16) - 128).float()
    s = packed.scales[expert].float().unsqueeze(-1)
    return q * s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single-GPU test requires CUDA")
def test_gpu_repack_error_and_fp16_bf16_layer_parity_and_entry_guards() -> None:
    cap = torch.cuda.get_device_capability(0)
    if cap not in ((8, 0), (8, 6)):
        pytest.skip(f"DxMoE native kernel requires SM80/SM86, got {cap}")

    device = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(20261123)
    e_cnt, h_dim, i_dim, topk = 4, 256, 256, 2
    w_bits = (4, 8, 4, 8)

    # Synthesize symmetric G=-1 quantized weights (exported as BF16 float tensors)
    def _synth_float_w(out_features: int, in_features: int) -> torch.Tensor:
        out = torch.empty((e_cnt, out_features, in_features), dtype=torch.bfloat16)
        for e, b in enumerate(w_bits):
            qmin, qmax = (-8, 7) if b == 4 else (-128, 127)
            q = torch.randint(qmin, qmax + 1, (out_features, in_features), generator=gen).float()
            amax = torch.linspace(0.04, 0.16, out_features)
            s = (amax / (7.0 if b == 4 else 127.0)).to(torch.float16).float().unsqueeze(-1)
            out[e] = (q * s).to(torch.bfloat16)
        return out.to(device)

    w13_bf16 = _synth_float_w(2 * i_dim, h_dim)
    w2_bf16 = _synth_float_w(h_dim, i_dim)
    w13_packed = vllm_dxmoe.pack_dxmoe_weight_channel(w13_bf16, w_bits)
    w2_packed = vllm_dxmoe.pack_dxmoe_weight_channel(w2_bf16, w_bits)

    # Verify repack relative reconstruction error against exported BF16 weights is tiny
    for e, b in enumerate(w_bits):
        rec13 = _dequant_packed_w(w13_packed, e, b)
        rel_err13 = (rec13 - w13_bf16[e].float()).norm() / w13_bf16[e].float().norm()
        assert float(rel_err13.item()) < (1.5e-2 if b == 4 else 5e-3)

    s_in = (torch.randint(0, 2, (h_dim,), generator=gen) * 2 - 1).to(dtype=torch.int8, device=device)
    s_mid = (torch.randint(0, 2, (i_dim,), generator=gen) * 2 - 1).to(dtype=torch.int8, device=device)
    beta_e = torch.tensor([0.5, 1.2, 0.8, 1.5], dtype=torch.float64, device=device)
    beta_d = torch.tensor([0.6, 1.1, 0.9, 1.4], dtype=torch.float64, device=device)
    tau_e = (0.043, 0.77)
    tau_d = (0.037, 1.13)

    state = DxMoELayerRuntimeState(
        layer_idx=1,
        hidden_size=h_dim,
        inter_size=i_dim,
        num_experts=e_cnt,
        top_k=topk,
        w13=w13_packed,
        w2=w2_packed,
        s_in=s_in,
        s_mid=s_mid,
        beta_entry=beta_e,
        beta_down=beta_d,
        tau_entry=tau_e,
        tau_down=tau_d,
        schedule_mode="dxmoe",
        apply_coupling_entry=True,
        apply_coupling_down=True,
        use_omega=False,
        sink_tokens=4,
        local_window=32,
        pipelined_code=1,
    )
    moe_cfg = _make_moe_config(num_experts=e_cnt, top_k=topk, hidden_size=h_dim, inter_size=i_dim)
    moe_cfg.device = device
    experts_op = DxMoEExperts(moe_config=moe_cfg, quant_config=DxMoEMoEQuantConfig(state))
    dummy_w = torch.empty(0, device=device)

    for act_dtype in (torch.float16, torch.bfloat16):
        x = (torch.randn((12, h_dim), generator=gen) * 0.7).to(device=device, dtype=act_dtype)
        logits = torch.randn((12, e_cnt), generator=gen).to(device=device, dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        rw_f32, ids_i64 = torch.topk(probs, k=topk, dim=-1)
        rw = (rw_f32 / rw_f32.sum(dim=-1, keepdim=True)).to(dtype=act_dtype).contiguous()
        ids = ids_i64.to(dtype=torch.int32).contiguous()
        out = torch.empty_like(x)

        experts_op.apply(
            output=out,
            hidden_states=x,
            w1=dummy_w,
            w2=dummy_w,
            topk_weights=rw,
            topk_ids=ids,
            activation=MoEActivation.SILU,
            global_num_experts=e_cnt,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=dummy_w,
            workspace2=dummy_w,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        vllm_dxmoe.check_dxmoe_workspaces_healthy(device)

        # Exact protocol PyTorch reference (including FP16 pool conversion for A16)
        xt = _hadamard128_ref(x, s_in)
        ref_acc = torch.zeros((12, h_dim), dtype=torch.float32, device=device)
        for t in range(12):
            am1 = float(xt[t].float().abs().amax().item())
            for s in range(topk):
                e = int(ids[t, s].item())
                g = float(rw[t, s].float().item())
                sc1 = (g * g) * float(beta_e[e].item()) * (am1 * am1)
                ab1 = 4 if sc1 < tau_e[0] else (8 if sc1 < tau_e[1] else 16)
                if w_bits[e] == 8 and ab1 < 8:
                    ab1 = 8
                xa = _fake_quant_act_row(xt[t].float(), ab1)
                dw13 = _dequant_packed_w(w13_packed, e, w_bits[e])
                if ab1 == 16:
                    dw13 = dw13.to(torch.float16).float()
                gu = (xa @ dw13.T).to(torch.float16)
                hm = _hadamard128_ref(
                    (F.silu(gu[:i_dim].float()) * gu[i_dim:].float()).unsqueeze(0),
                    s_mid,
                    out_dtype=act_dtype,
                )[0]
                am2 = float(hm.float().abs().amax().item())
                sc2 = (g * g) * float(beta_d[e].item()) * (am2 * am2)
                ab2 = 4 if sc2 < tau_d[0] else (8 if sc2 < tau_d[1] else 16)
                if w_bits[e] == 8 and ab2 < 8:
                    ab2 = 8
                ha = _fake_quant_act_row(hm.float(), ab2)
                dw2 = _dequant_packed_w(w2_packed, e, w_bits[e])
                if ab2 == 16:
                    dw2 = dw2.to(torch.float16).float()
                y_call = (ha @ dw2.T).to(torch.float16).float()
                ref_acc[t] += g * y_call
        ref_cast = ref_acc.to(act_dtype)
        rel_l2 = (out.float() - ref_cast.float()).norm() / ref_cast.float().norm()
        cos_sim = F.cosine_similarity(out.float().reshape(1, -1), ref_cast.float().reshape(1, -1)).item()
        assert float(rel_l2.item()) < 5e-3
        assert cos_sim > 0.9999
        torch.testing.assert_close(out, ref_cast, rtol=3.5e-2, atol=9e-2)

        # Also verify pure static_a16 FP16/BF16 conversion parity to tight tolerance
        state_a16 = dataclasses.replace(state, schedule_mode="static_a16")
        op_a16 = DxMoEExperts(moe_config=moe_cfg, quant_config=DxMoEMoEQuantConfig(state_a16))
        out_a16 = torch.empty_like(x)
        op_a16.apply(
            output=out_a16,
            hidden_states=x,
            w1=dummy_w,
            w2=dummy_w,
            topk_weights=rw,
            topk_ids=ids,
            activation=MoEActivation.SILU,
            global_num_experts=e_cnt,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=dummy_w,
            workspace2=dummy_w,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        ref_a16 = torch.zeros((12, h_dim), dtype=torch.float32, device=device)
        for t in range(12):
            xa16 = xt[t].to(torch.float16).float()
            for s in range(topk):
                e = int(ids[t, s].item())
                g = float(rw[t, s].float().item())
                dw13 = _dequant_packed_w(w13_packed, e, w_bits[e]).to(torch.float16).float()
                gu = (xa16 @ dw13.T).to(torch.float16)
                hm = _hadamard128_ref(
                    (F.silu(gu[:i_dim].float()) * gu[i_dim:].float()).unsqueeze(0),
                    s_mid,
                    out_dtype=act_dtype,
                )[0].to(torch.float16).float()
                dw2 = _dequant_packed_w(w2_packed, e, w_bits[e]).to(torch.float16).float()
                ref_a16[t] += g * (hm @ dw2.T).to(torch.float16).float()
        torch.testing.assert_close(out_a16, ref_a16.to(act_dtype), rtol=8e-3, atol=1.5e-2)

    # Bare-pointer entry guards: reject overlapping x/output and 1D / wrong-K topk_ids!
    x_guard = torch.randn((4, h_dim), device=device, dtype=torch.bfloat16)
    ids_guard = torch.zeros((4, topk), device=device, dtype=torch.int32)
    rw_guard = torch.full((4, topk), 0.5, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="non-overlapping"):
        experts_op.apply(
            output=x_guard, hidden_states=x_guard, w1=dummy_w, w2=dummy_w,
            topk_weights=rw_guard, topk_ids=ids_guard, activation=MoEActivation.SILU,
            global_num_experts=e_cnt, expert_map=None, a1q_scale=None, a2_scale=None,
            workspace13=dummy_w, workspace2=dummy_w, expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
    with pytest.raises(ValueError, match="topk_ids must be contiguous 2D"):
        experts_op.apply(
            output=torch.empty_like(x_guard), hidden_states=x_guard, w1=dummy_w, w2=dummy_w,
            topk_weights=rw_guard, topk_ids=ids_guard.view(-1), activation=MoEActivation.SILU,
            global_num_experts=e_cnt, expert_map=None, a1q_scale=None, a2_scale=None,
            workspace13=dummy_w, workspace2=dummy_w, expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
    with pytest.raises(ValueError, match="topk_ids must be contiguous 2D"):
        experts_op.apply(
            output=torch.empty_like(x_guard), hidden_states=x_guard, w1=dummy_w, w2=dummy_w,
            topk_weights=torch.full((4, topk + 1), 0.5, device=device, dtype=torch.bfloat16),
            topk_ids=torch.zeros((4, topk + 1), device=device, dtype=torch.int32),
            activation=MoEActivation.SILU, global_num_experts=e_cnt, expert_map=None,
            a1q_scale=None, a2_scale=None, workspace13=dummy_w, workspace2=dummy_w,
            expert_tokens_meta=None, apply_router_weight_on_input=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single-GPU test requires CUDA")
def test_gpu_real_vllm_router_neg1_padding_single_cudagraph_multi_replay_and_active_bad_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real vLLM weight loader + native `ops.topk_softmax` `-1` padding + 3-replay single CUDA Graph!"""
    cap = torch.cuda.get_device_capability(0)
    if cap not in ((8, 0), (8, 6)):
        pytest.skip(f"DxMoE native kernel requires SM80/SM86, got {cap}")

    from vllm.model_executor.layers.fused_moe.router.router_factory import (
        create_fused_moe_router,
    )
    from vllm.model_executor.model_loader import get_model

    # Keep VLLM_MOE_SKIP_PADDING=1 (default) so native `ops.topk_softmax` emits real `-1` on padded rows!
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    monkeypatch.setenv("VLLM_DXMOE_SINK_TOKENS", "4")
    monkeypatch.setenv("VLLM_DXMOE_LOCAL_WINDOW", "4")

    model_dir = tmp_path / "mini_ds2_dxmoe"
    model_dir.mkdir(parents=True)
    h_dim, i_dim, e_cnt, topk = 256, 256, 8, 2
    gen = torch.Generator(device="cpu").manual_seed(20261124)

    hf_cfg = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "torch_dtype": "bfloat16",
        "vocab_size": 512,
        "hidden_size": h_dim,
        "intermediate_size": 512,
        "moe_intermediate_size": i_dim,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "kv_lora_rank": 512,
        "q_lora_rank": None,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": e_cnt,
        "num_experts_per_tok": topk,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "scoring_func": "softmax",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    (model_dir / "config.json").write_text(json.dumps(hf_cfg, indent=2), encoding="utf-8")

    # Mixed W4/W8 experts across 2 MoE layers (layers 1 and 2) with Hadamard block128
    w_bits = (4, 8, 4, 8, 4, 8, 4, 4)
    # Choose thresholds that split omega=1.0 vs omega=2.0 scores into distinct A4/A8/A16 pools!
    meta = _sample_dxmoe_meta(
        layers=(1, 2),
        hidden_size=h_dim,
        inter_size=i_dim,
        num_experts=e_cnt,
        top_k=topk,
        w_bits=w_bits,
        hadamard_mode="block128",
        tau_entry=(0.02, 0.12),
        tau_down=(0.02, 0.12),
    )
    (model_dir / "dxmoe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    signs_tensors = {}
    for l_idx in (1, 2):
        signs_tensors[f"layer.{l_idx}.s_in"] = (torch.randint(0, 2, (h_dim,), generator=gen) * 2 - 1).to(torch.int8)
        signs_tensors[f"layer.{l_idx}.s_mid"] = (torch.randint(0, 2, (i_dim,), generator=gen) * 2 - 1).to(torch.int8)
    save_file(signs_tensors, str(model_dir / "dxmoe_signs.safetensors"))

    # Populate all DeepseekV2ForCausalLM weights and save to model.safetensors
    state_dict: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": (torch.randn((512, h_dim), generator=gen) * 0.03).to(torch.bfloat16),
        "model.norm.weight": torch.ones((h_dim,), dtype=torch.bfloat16),
        "lm_head.weight": (torch.randn((512, h_dim), generator=gen) * 0.03).to(torch.bfloat16),
    }
    for l_idx in range(3):
        pfx = f"model.layers.{l_idx}"
        state_dict[f"{pfx}.input_layernorm.weight"] = torch.ones((h_dim,), dtype=torch.bfloat16)
        state_dict[f"{pfx}.post_attention_layernorm.weight"] = torch.ones((h_dim,), dtype=torch.bfloat16)
        state_dict[f"{pfx}.self_attn.q_proj.weight"] = (torch.randn((4 * 192, h_dim), generator=gen) * 0.03).to(torch.bfloat16)
        state_dict[f"{pfx}.self_attn.kv_a_proj_with_mqa.weight"] = (torch.randn((512 + 64, h_dim), generator=gen) * 0.03).to(torch.bfloat16)
        state_dict[f"{pfx}.self_attn.kv_a_layernorm.weight"] = torch.ones((512,), dtype=torch.bfloat16)
        state_dict[f"{pfx}.self_attn.kv_b_proj.weight"] = (torch.randn((4 * (128 + 128), 512), generator=gen) * 0.03).to(torch.bfloat16)
        state_dict[f"{pfx}.self_attn.o_proj.weight"] = (torch.randn((h_dim, 4 * 128), generator=gen) * 0.03).to(torch.bfloat16)
        if l_idx == 0:
            for proj, sh in (("gate_proj", (512, h_dim)), ("up_proj", (512, h_dim)), ("down_proj", (h_dim, 512))):
                state_dict[f"{pfx}.mlp.{proj}.weight"] = (torch.randn(sh, generator=gen) * 0.03).to(torch.bfloat16)
        else:
            state_dict[f"{pfx}.mlp.gate.weight"] = (torch.randn((e_cnt, h_dim), generator=gen) * 0.2).to(torch.bfloat16)
            for proj, sh in (("gate_proj", (i_dim, h_dim)), ("up_proj", (i_dim, h_dim)), ("down_proj", (h_dim, i_dim))):
                state_dict[f"{pfx}.mlp.shared_experts.{proj}.weight"] = (torch.randn(sh, generator=gen) * 0.03).to(torch.bfloat16)
            for e in range(e_cnt):
                for proj, sh in (("gate_proj", (i_dim, h_dim)), ("up_proj", (i_dim, h_dim)), ("down_proj", (h_dim, i_dim))):
                    state_dict[f"{pfx}.mlp.experts.{e}.{proj}.weight"] = (torch.randn(sh, generator=gen) * 0.05).to(torch.bfloat16)
    save_file(state_dict, str(model_dir / "model.safetensors"))

    import socket
    from vllm.config import ModelConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    vllm_cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir),
            tokenizer=str(model_dir),
            dtype="bfloat16",
            quantization="dxmoe",
            max_model_len=256,
            enforce_eager=True,
            skip_tokenizer_init=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            max_model_len=256,
            is_encoder_decoder=False,
        ),
        kernel_config=KernelConfig(moe_backend="dxmoe"),
    )
    device = torch.device("cuda:0")
    if not is_workspace_manager_initialized():
        init_workspace_manager(device)
    need_cleanup = False
    try:
        with set_current_vllm_config(vllm_cfg):
            if not model_parallel_is_initialized():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    free_port = sock.getsockname()[1]
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=f"tcp://127.0.0.1:{free_port}",
                    local_rank=0,
                    backend="nccl",
                )
                need_cleanup = True
                ensure_model_parallel_initialized(1, 1)
            model = get_model(vllm_config=vllm_cfg)
            router_probe = create_fused_moe_router(
                top_k=topk, global_num_experts=e_cnt, renormalize=True, scoring_func="softmax"
            )
        model.eval()

        # Extract the 2 real loaded DeepseekV2MoE blocks (layers 1 & 2) with real vLLM router + DxMoEExperts!
        moe_layer_1 = model.model.layers[1].mlp
        moe_layer_2 = model.model.layers[2].mlp
        assert isinstance(moe_layer_1.experts.routed_experts.quant_method, DxMoEMoEMethod)
        assert isinstance(moe_layer_2.experts.routed_experts.quant_method, DxMoEMoEMethod)

        # First verify native router indeed emits `-1` on padded rows when is_padding is set!
        m_cap = 16
        static_x = torch.zeros((m_cap, h_dim), dtype=torch.bfloat16, device=device)
        static_pos = torch.zeros((m_cap,), dtype=torch.int64, device=device)
        static_seq_lens = torch.zeros((1,), dtype=torch.int32, device=device)
        static_pad = torch.ones((m_cap,), dtype=torch.bool, device=device)
        with set_forward_context(
            None, vllm_cfg, num_tokens=m_cap, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            is_padding=static_pad, positions=static_pos, seq_lens=static_seq_lens,
        ):
            probe_logits = torch.randn((m_cap, e_cnt), dtype=torch.float32, device=device)
            _, probe_ids = router_probe.select_experts(static_x, probe_logits)
            assert bool((probe_ids == -1).all()), "Expected native vLLM router to emit -1 on padded rows"

        def _two_layer_routed_chain(x_in: torch.Tensor) -> torch.Tensor:
            logits1, _ = moe_layer_1.gate(x_in)
            rw1, ids1 = moe_layer_1.experts.router.select_experts(x_in, logits1)
            h1 = moe_layer_1.experts.routed_experts.quant_method.apply(
                moe_layer_1.experts.routed_experts, x_in, rw1, ids1, None, None
            )
            logits2, _ = moe_layer_2.gate(h1)
            rw2, ids2 = moe_layer_2.experts.router.select_experts(h1, logits2)
            h2 = moe_layer_2.experts.routed_experts.quant_method.apply(
                moe_layer_2.experts.routed_experts, h1, rw2, ids2, None, None
            )
            return h2

        # Warmup & capture ONE single CUDA Graph at M_cap = 16
        capture_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(capture_stream):
            with set_forward_context(
                None, vllm_cfg, num_tokens=m_cap, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                is_padding=static_pad, positions=static_pos, seq_lens=static_seq_lens,
            ):
                _ = _two_layer_routed_chain(static_x)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            with set_forward_context(
                None, vllm_cfg, num_tokens=m_cap, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                is_padding=static_pad, positions=static_pos, seq_lens=static_seq_lens,
            ):
                graph_out = _two_layer_routed_chain(static_x)

        replays = (
            (5, 0, 101),
            (11, 0, 202),
            (7, 64, 303),
        )
        for m_valid, past_len, seed in replays:
            r_gen = torch.Generator(device="cpu").manual_seed(seed)
            x_valid = (torch.randn((m_valid, h_dim), generator=r_gen) * 0.8).to(device=device, dtype=torch.bfloat16)

            static_x.fill_(-99.0)
            static_x[:m_valid].copy_(x_valid)
            static_pos.zero_()
            static_pos[:m_valid].copy_(
                torch.arange(past_len, past_len + m_valid, dtype=torch.int64, device=device)
            )
            static_seq_lens[0] = past_len + m_valid
            static_pad[:m_valid].fill_(False)
            static_pad[m_valid:].fill_(True)

            with torch.cuda.stream(capture_stream):
                graph.replay()
            capture_stream.synchronize()
            vllm_dxmoe.check_dxmoe_workspaces_healthy(device)

            eager_pos = torch.arange(past_len, past_len + m_valid, dtype=torch.int64, device=device)
            eager_seq_lens = torch.tensor([past_len + m_valid], dtype=torch.int32, device=device)
            eager_pad = torch.zeros((m_valid,), dtype=torch.bool, device=device)
            with set_forward_context(
                None, vllm_cfg, num_tokens=m_valid, cudagraph_runtime_mode=CUDAGraphMode.NONE,
                is_padding=eager_pad, positions=eager_pos, seq_lens=eager_seq_lens,
            ):
                eager_out = _two_layer_routed_chain(x_valid)
            vllm_dxmoe.check_dxmoe_workspaces_healthy(device)

            torch.testing.assert_close(graph_out[:m_valid], eager_out, rtol=1e-3, atol=1e-3)
            assert bool((graph_out[m_valid:] == 0).all()), f"Non-zero padded tail on m_valid={m_valid}"

        with pytest.raises(NotImplementedError, match="requires persistent GPU positions and seq_lens"):
            with set_forward_context(
                None, vllm_cfg, num_tokens=8, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                is_padding=static_pad[:8], positions=None, seq_lens=None,
            ):
                _ = _two_layer_routed_chain(static_x[:8])

        routed_l1 = moe_layer_1.experts.routed_experts
        experts_l1 = routed_l1.quant_method.moe_kernel.fused_experts
        dummy_w = torch.empty(0, device=device)
        for bad_id in (-1, e_cnt + 5):
            x_bad = torch.randn((6, h_dim), dtype=torch.bfloat16, device=device)
            out_bad = torch.zeros_like(x_bad)
            ids_bad = torch.tensor(
                [[0, 1], [2, bad_id], [3, 4], [1, 2], [-1, -1], [-1, -1]],
                dtype=torch.int32,
                device=device,
            )
            rw_bad = torch.full((6, topk), 0.5, dtype=torch.bfloat16, device=device)
            pad_6 = torch.tensor([False, False, False, False, True, True], dtype=torch.bool, device=device)
            pos_6 = torch.arange(6, dtype=torch.int64, device=device)
            seq_6 = torch.tensor([4], dtype=torch.int32, device=device)

            with set_forward_context(
                None, vllm_cfg, num_tokens=6, cudagraph_runtime_mode=CUDAGraphMode.NONE,
                is_padding=pad_6, positions=pos_6, seq_lens=seq_6,
            ):
                experts_l1.apply(
                    output=out_bad, hidden_states=x_bad, w1=dummy_w, w2=dummy_w,
                    topk_weights=rw_bad, topk_ids=ids_bad, activation=MoEActivation.SILU,
                    global_num_experts=e_cnt, expert_map=None, a1q_scale=None, a2_scale=None,
                    workspace13=dummy_w, workspace2=dummy_w, expert_tokens_meta=None,
                    apply_router_weight_on_input=False,
                )
            assert bool(torch.isnan(out_bad[:4]).all())
            assert bool((out_bad[4:] == 0).all())
            with pytest.raises(RuntimeError, match="latched error status=2"):
                vllm_dxmoe.check_dxmoe_workspaces_healthy(device)
            vllm_dxmoe.check_dxmoe_workspaces_healthy(device)
    finally:
        if need_cleanup:
            cleanup_dist_env_and_memory()


def _pack_3bit_blocks_32(codes_32: torch.Tensor) -> torch.Tensor:
    assert codes_32.shape[-1] == 32
    c = codes_32.to(dtype=torch.int64) & 0x7
    w0 = torch.zeros(c.shape[:-1], dtype=torch.int64)
    w1 = torch.zeros(c.shape[:-1], dtype=torch.int64)
    w2 = torch.zeros(c.shape[:-1], dtype=torch.int64)
    for i in range(10):
        w0 |= c[..., i] << (3 * i)
    w0 |= (c[..., 10] & 0x3) << 30
    w1 |= (c[..., 10] >> 2) & 0x1
    for j in range(10):
        w1 |= c[..., 11 + j] << (1 + 3 * j)
    w1 |= (c[..., 21] & 0x1) << 31
    w2 |= (c[..., 21] >> 1) & 0x3
    for j in range(10):
        w2 |= c[..., 22 + j] << (2 + 3 * j)
    words_u64 = torch.stack((w0, w1, w2), dim=-1)
    words_s64 = torch.where(words_u64 >= (1 << 31), words_u64 - (1 << 32), words_u64)
    return words_s64.to(dtype=torch.int32)


def _pack_sym_gptq_matrix(
    q_signed: torch.Tensor,
    scales_1d: torch.Tensor,
    bits: int,
    *,
    group_size: int = -1,
) -> dict[str, torch.Tensor]:
    """Pack signed `[-2^(b-1), 2^(b-1)-1]` codes `[N, K]` into standard HF GPTQ tensors."""
    assert bits in (2, 3, 4, 8)
    n_out, k_in = int(q_signed.shape[0]), int(q_signed.shape[1])
    zero_val = 1 << (bits - 1)
    mask = (1 << bits) - 1
    q_u = (q_signed.to(dtype=torch.int32) + zero_val) & mask
    num_groups = 1 if group_size == -1 else k_in // group_size
    if scales_1d.ndim == 1:
        scales_gn = scales_1d.to(dtype=torch.float16).unsqueeze(0).expand(num_groups, n_out).contiguous()
    else:
        scales_gn = scales_1d.to(dtype=torch.float16).contiguous()

    if bits == 3:
        assert k_in % 32 == 0 and n_out % 32 == 0
        # [N, K] -> [K // 32, 32, N] -> [N, K // 32, 32] -> [N, K // 32, 3] -> [(K * 3) // 32, N]
        q_blocks = q_u.reshape(n_out, k_in // 32, 32)
        qweight = _pack_3bit_blocks_32(q_blocks).reshape(n_out, (k_in * 3) // 32).transpose(0, 1).contiguous()
        qz_row = torch.tensor(vllm_dxmoe.GPTQ_SYM_QZEROS_W3, dtype=torch.int32).repeat(n_out // 32)
        qzeros = qz_row.unsqueeze(0).expand(num_groups, (n_out * 3) // 32).contiguous().clone()
    else:
        pack_factor = 32 // bits
        q_kn = q_u.transpose(0, 1).contiguous().reshape(k_in // pack_factor, pack_factor, n_out)
        qweight = torch.zeros((k_in // pack_factor, n_out), dtype=torch.int32)
        for i in range(pack_factor):
            qweight |= q_kn[:, i, :] << (i * bits)
        qz_val = vllm_dxmoe.GPTQ_SYM_QZEROS[bits]
        qzeros = torch.full((num_groups, n_out // pack_factor), qz_val, dtype=torch.int32)

    g_idx = (
        torch.zeros((k_in,), dtype=torch.int32)
        if group_size == -1
        else torch.div(torch.arange(k_in, dtype=torch.int32), group_size, rounding_mode="floor")
    )
    return {"qweight": qweight, "scales": scales_gn, "qzeros": qzeros, "g_idx": g_idx}


def test_dxmoe_gptq_repack_is_bit_lossless_and_rejects_grouped_or_asymmetric() -> None:
    gen = torch.Generator(device="cpu").manual_seed(2026)
    n_out, k_in = 256, 128
    for bits in (2, 3, 4, 8):
        zero = 1 << (bits - 1)
        lo = -(1 << (bits - 1))
        hi = 1 << (bits - 1)
        if bits == 8:
            lo = -127
        q_signed = torch.randint(lo, hi, (n_out, k_in), generator=gen, dtype=torch.int16)
        if bits == 3:
            # Explicitly exercise every 3-bit code (-4..3) on cross-word slots 10 and 21!
            for slot_mod in (10, 21):
                for blk in range(k_in // 32):
                    q_signed[:, blk * 32 + slot_mod] = ((torch.arange(n_out) + blk + slot_mod) % 8 - 4).to(
                        torch.int16
                    )
        scales = (torch.rand((n_out,), generator=gen, dtype=torch.float32) * 0.04 + 0.005).to(
            torch.float16
        )
        pkg = _pack_sym_gptq_matrix(q_signed, scales, bits, group_size=-1)
        repacked = vllm_dxmoe.repack_gptq_qweight_shard(
            pkg["qweight"],
            k_expected=k_in,
            n_expected=n_out,
            w_bits=bits,
            target_device=torch.device("cpu"),
        )
        s_checked = vllm_dxmoe.validate_gptq_scales_shard(
            pkg["scales"], n_expected=n_out, target_device=torch.device("cpu")
        )
        vllm_dxmoe.validate_gptq_qzeros_shard(
            pkg["qzeros"], n_expected=n_out, w_bits=bits
        )

        u8_expected = (q_signed + zero).to(dtype=torch.uint8)
        if bits == 2:
            g = u8_expected.reshape(n_out, k_in // 4, 4)
            expected_codes = (
                g[:, :, 0] | (g[:, :, 1] << 2) | (g[:, :, 2] << 4) | (g[:, :, 3] << 6)
            ).contiguous()
        elif bits == 3:
            g = u8_expected.reshape(n_out, k_in // 8, 8).to(torch.int32)
            p24 = sum((g[:, :, i] << (3 * i)) for i in range(8))
            expected_codes = torch.stack(
                ((p24 & 0xFF).to(torch.uint8), ((p24 >> 8) & 0xFF).to(torch.uint8), ((p24 >> 16) & 0xFF).to(torch.uint8)),
                dim=-1,
            ).reshape(n_out, (k_in * 3) // 8).contiguous()
        elif bits == 4:
            expected_codes = (u8_expected[:, 0::2] | (u8_expected[:, 1::2] << 4)).contiguous()
        else:
            expected_codes = u8_expected.contiguous()
        assert torch.equal(repacked, expected_codes)
        assert torch.equal(s_checked, scales)

    # Verify arena byte layout for heterogeneous [2, 3, 4, 8] experts
    bits_clean, offsets, total_bytes = vllm_dxmoe.compute_dxmoe_arena_layout(
        4, n_out, k_in, (2, 3, 4, 8)
    )
    assert bits_clean == (2, 3, 4, 8)
    assert offsets == (
        0,
        n_out * (k_in // 4),
        n_out * (k_in // 4) + n_out * ((k_in * 3) // 8),
        n_out * (k_in // 4) + n_out * ((k_in * 3) // 8) + n_out * (k_in // 2),
    )
    assert total_bytes == offsets[-1] + n_out * k_in

    # Reject group_size != -1 scales and asymmetric qzeros (including W2 and W3)
    pkg_g64 = _pack_sym_gptq_matrix(
        torch.randint(-8, 8, (n_out, k_in), generator=gen, dtype=torch.int16),
        torch.full((2, n_out), 0.02, dtype=torch.float16),
        4,
        group_size=64,
    )
    with pytest.raises(ValueError, match="group_size=-1"):
        vllm_dxmoe.validate_gptq_scales_shard(
            pkg_g64["scales"], n_expected=n_out, target_device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="group_size=-1"):
        vllm_dxmoe.validate_gptq_qzeros_shard(
            pkg_g64["qzeros"], n_expected=n_out, w_bits=4
        )
    bad_qz = torch.full((1, n_out // 8), 0x66666666, dtype=torch.int32)
    with pytest.raises(ValueError, match="symmetric gptq-v1 qzeros"):
        vllm_dxmoe.validate_gptq_qzeros_shard(bad_qz, n_expected=n_out, w_bits=4)
    bad_qz2 = torch.full((1, n_out // 16), 0x55555554, dtype=torch.int32)
    with pytest.raises(ValueError, match="symmetric gptq-v1 qzeros"):
        vllm_dxmoe.validate_gptq_qzeros_shard(bad_qz2, n_expected=n_out, w_bits=2)
    for word_idx in (0, 1, 2):
        bad_qz3 = (
            torch.tensor(vllm_dxmoe.GPTQ_SYM_QZEROS_W3, dtype=torch.int32)
            .repeat(n_out // 32)
            .reshape(1, -1)
            .clone()
        )
        bad_qz3[0, word_idx] ^= 0x8
        with pytest.raises(ValueError, match="symmetric gptq-v1 qzeros"):
            vllm_dxmoe.validate_gptq_qzeros_shard(bad_qz3, n_expected=n_out, w_bits=3)


def test_dxmoe_adaptive_shared_experts_wna16_vs_fp16_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify adaptive shared_experts dispatch: FP16 -> UnquantizedLinearMethod, GPTQ -> AutoGPTQLinearMethod (strictly WNA16)."""
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod

    # Even if VLLM_MARLIN_INPUT_DTYPE is set to int8 in env, DxMoE MUST keep shared_experts WNA16!
    monkeypatch.setenv("VLLM_MARLIN_INPUT_DTYPE", "int8")
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_gptq.verify_marlin_supported",
        lambda **_kw: None,
    )

    h_dim, i_dim, shared_i = 256, 256, 256
    meta_doc = _sample_dxmoe_meta(layers=(1,), hidden_size=h_dim, inter_size=i_dim, num_experts=2, top_k=1, w_bits=(4, 8))

    for case_name, quantize_shared, shared_gs in (
        ("shared_fp16", False, -1),
        ("shared_gptq_g_neg1", True, -1),
        ("shared_gptq_g128", True, 128),
    ):
        case_dir = tmp_path / case_name
        case_dir.mkdir()
        (case_dir / "dxmoe_meta.json").write_text(json.dumps(meta_doc), encoding="utf-8")
        qcfg_doc = {
            "bits": 4,
            "group_size": -1,
            "sym": True,
            "desc_act": False,
            "quant_method": "gptq",
            "checkpoint_format": "gptq",
            "modules_in_block_to_quantize": ["mlp.experts."]
            + (["mlp.shared_experts."] if quantize_shared else []),
            "lm_head": False,
        }
        if quantize_shared and shared_gs != -1:
            qcfg_doc["dynamic"] = {r"+:.*mlp\.shared_experts\..*": {"bits": 4, "group_size": shared_gs}}
        (case_dir / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v2", "quantization_config": qcfg_doc}),
            encoding="utf-8",
        )

        st_tensors: dict[str, torch.Tensor] = {
            "model.layers.1.self_attn.q_proj.weight": torch.zeros((h_dim, h_dim), dtype=torch.float16),
        }
        for e_idx, bw in enumerate((4, 8)):
            for proj, (n_d, k_d) in (
                ("gate_proj", (i_dim, h_dim)),
                ("up_proj", (i_dim, h_dim)),
                ("down_proj", (h_dim, i_dim)),
            ):
                pkg = _pack_sym_gptq_matrix(
                    torch.zeros((n_d, k_d), dtype=torch.int16),
                    torch.ones((n_d,), dtype=torch.float16),
                    bw,
                    group_size=-1,
                )
                for suf, t in pkg.items():
                    st_tensors[f"model.layers.1.mlp.experts.{e_idx}.{proj}.{suf}"] = t

        if quantize_shared:
            for proj, (n_d, k_d) in (
                ("gate_proj", (shared_i, h_dim)),
                ("up_proj", (shared_i, h_dim)),
                ("down_proj", (h_dim, shared_i)),
            ):
                n_grp = 1 if shared_gs == -1 else k_d // shared_gs
                pkg_s = _pack_sym_gptq_matrix(
                    torch.zeros((n_d, k_d), dtype=torch.int16),
                    torch.ones((n_grp, n_d), dtype=torch.float16),
                    4,
                    group_size=shared_gs,
                )
                for suf, t in pkg_s.items():
                    st_tensors[f"model.layers.1.mlp.shared_experts.{proj}.{suf}"] = t
        else:
            st_tensors["model.layers.1.mlp.shared_experts.gate_proj.weight"] = torch.zeros(
                (shared_i, h_dim), dtype=torch.float16
            )
            st_tensors["model.layers.1.mlp.shared_experts.up_proj.weight"] = torch.zeros(
                (shared_i, h_dim), dtype=torch.float16
            )
            st_tensors["model.layers.1.mlp.shared_experts.down_proj.weight"] = torch.zeros(
                (h_dim, shared_i), dtype=torch.float16
            )
        save_file(st_tensors, str(case_dir / "model.safetensors"))

        dx_cfg = DxMoEConfig({"_model_dir": str(case_dir), **qcfg_doc})
        dx_cfg.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

        dummy_linear = object.__new__(MergedColumnParallelLinear)
        dummy_down = object.__new__(RowParallelLinear)
        attn_m = dx_cfg.get_quant_method(dummy_linear, "model.layers.1.self_attn.q_proj")
        assert isinstance(attn_m, UnquantizedLinearMethod)

        sh_gu_m = dx_cfg.get_quant_method(
            dummy_linear, "model.layers.1.mlp.shared_experts.gate_up_proj"
        )
        sh_dn_m = dx_cfg.get_quant_method(
            dummy_down, "model.layers.1.mlp.shared_experts.down_proj"
        )
        if not quantize_shared:
            assert isinstance(sh_gu_m, UnquantizedLinearMethod)
            assert isinstance(sh_dn_m, UnquantizedLinearMethod)
        else:
            assert isinstance(sh_gu_m, AutoGPTQLinearMethod)
            assert isinstance(sh_dn_m, AutoGPTQLinearMethod)
            assert sh_gu_m.input_dtype is None
            assert sh_dn_m.input_dtype is None
            assert sh_gu_m.quant_config.group_size == shared_gs

        moe_cfg = _make_moe_config(num_experts=2, top_k=1, hidden_size=h_dim, inter_size=i_dim)
        routed_stub = object.__new__(RoutedExperts)
        routed_stub.moe_config = moe_cfg
        routed_m = dx_cfg.get_quant_method(
            routed_stub, "model.layers.1.mlp.experts.routed_experts"
        )
        assert isinstance(routed_m, DxMoEMoEMethod)
        assert routed_m.checkpoint_is_gptq is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single-GPU test requires CUDA")
@pytest.mark.parametrize("quantize_shared", [False, True])
def test_dxmoe_vllm_gptq_checkpoint_e2e_fp16_and_wna16_shared_experts(
    tmp_path: Path, quantize_shared: bool
) -> None:
    """Load real DeepseekV2ForCausalLM from GPTQ-packed DxMoE checkpoint with FP16 or W4A16 shared_experts."""
    cap = torch.cuda.get_device_capability(0)
    if cap not in ((8, 0), (8, 6)):
        pytest.skip(f"DxMoE native kernel requires SM80/SM86, got {cap}")

    import socket
    from vllm.config import ModelConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
    from vllm.model_executor.model_loader import get_model
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    h_dim, dense_i, moe_i, e_cnt, topk = 256, 256, 256, 4, 2
    w_bits_l1 = (4, 8, 4, 8)
    model_dir = tmp_path / ("ds2_gptq_sh_w4" if quantize_shared else "ds2_gptq_sh_fp16")
    model_dir.mkdir()

    gen = torch.Generator(device="cpu").manual_seed(515)
    s_in_1 = torch.where(torch.randint(0, 2, (h_dim,), generator=gen) == 0, -1, 1).to(torch.int8)
    s_mid_1 = torch.where(torch.randint(0, 2, (moe_i,), generator=gen) == 0, -1, 1).to(torch.int8)
    save_file(
        {"layer.1.s_in": s_in_1, "layer.1.s_mid": s_mid_1},
        str(model_dir / "dxmoe_signs.safetensors"),
    )

    qcfg_doc: dict = {
        "bits": 4,
        "group_size": -1,
        "sym": True,
        "desc_act": False,
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
        "modules_in_block_to_quantize": ["mlp.experts."]
        + (["mlp.shared_experts."] if quantize_shared else []),
        "lm_head": False,
        "dynamic": {
            r"+:^model\.layers\.1\.mlp\.experts\.(1|3)\.(gate_proj|up_proj|down_proj)$": {"bits": 8},
            **({r"+:.*mlp\.shared_experts\..*": {"bits": 4, "group_size": 128}} if quantize_shared else {}),
        },
    }
    hf_cfg = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "vocab_size": 128,
        "hidden_size": h_dim,
        "intermediate_size": dense_i,
        "moe_intermediate_size": moe_i,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "n_shared_experts": 1,
        "n_routed_experts": e_cnt,
        "num_experts_per_tok": topk,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "q_lora_rank": None,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "scoring_func": "softmax",
        "topk_method": "greedy",
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 256,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "quantization_config": qcfg_doc,
    }
    (model_dir / "config.json").write_text(json.dumps(hf_cfg, indent=2), encoding="utf-8")

    st: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": (torch.randn((128, h_dim), generator=gen) * 0.02).to(torch.bfloat16),
        "model.norm.weight": torch.ones((h_dim,), dtype=torch.bfloat16),
        "lm_head.weight": (torch.randn((128, h_dim), generator=gen) * 0.02).to(torch.bfloat16),
    }
    for l_idx in range(2):
        pfx = f"model.layers.{l_idx}"
        st[f"{pfx}.input_layernorm.weight"] = torch.ones((h_dim,), dtype=torch.bfloat16)
        st[f"{pfx}.post_attention_layernorm.weight"] = torch.ones((h_dim,), dtype=torch.bfloat16)
        st[f"{pfx}.self_attn.q_proj.weight"] = (torch.randn((2 * 192, h_dim), generator=gen) * 0.02).to(torch.bfloat16)
        st[f"{pfx}.self_attn.kv_a_proj_with_mqa.weight"] = (torch.randn((512 + 64, h_dim), generator=gen) * 0.02).to(torch.bfloat16)
        st[f"{pfx}.self_attn.kv_a_layernorm.weight"] = torch.ones((512,), dtype=torch.bfloat16)
        st[f"{pfx}.self_attn.kv_b_proj.weight"] = (torch.randn((2 * (128 + 128), 512), generator=gen) * 0.02).to(torch.bfloat16)
        st[f"{pfx}.self_attn.o_proj.weight"] = (torch.randn((h_dim, 2 * 128), generator=gen) * 0.02).to(torch.bfloat16)

    # Layer 0 dense MLP
    for proj, (n_d, k_d) in (("gate_proj", (dense_i, h_dim)), ("up_proj", (dense_i, h_dim)), ("down_proj", (h_dim, dense_i))):
        st[f"model.layers.0.mlp.{proj}.weight"] = (torch.randn((n_d, k_d), generator=gen) * 0.02).to(torch.bfloat16)

    # Layer 1 MoE router + mixed W4/W8 GPTQ routed experts + shared_experts
    st["model.layers.1.mlp.gate.weight"] = (torch.randn((e_cnt, h_dim), generator=gen) * 0.2).to(torch.bfloat16)
    experts_meta = {}
    routed_deq: dict[tuple[int, str], torch.Tensor] = {}
    for e_idx, bw in enumerate(w_bits_l1):
        lo, hi = (-8, 8) if bw == 4 else (-127, 128)
        exp_entry = {}
        for slot, proj, (n_d, k_d) in (
            ("gate", "gate_proj", (moe_i, h_dim)),
            ("up", "up_proj", (moe_i, h_dim)),
            ("down", "down_proj", (h_dim, moe_i)),
        ):
            q_s = torch.randint(lo, hi, (n_d, k_d), generator=gen, dtype=torch.int16)
            sc = (torch.rand((n_d,), generator=gen, dtype=torch.float32) * 0.01 + 0.002).to(torch.float16)
            pkg = _pack_sym_gptq_matrix(q_s, sc, bw, group_size=-1)
            for suf, t in pkg.items():
                st[f"model.layers.1.mlp.experts.{e_idx}.{proj}.{suf}"] = t
            w_deq = q_s.to(torch.float32) * sc.to(torch.float32).unsqueeze(-1)
            routed_deq[(e_idx, slot)] = w_deq
            beta = float(w_deq.to(torch.float64).square().sum().item() / 12.0)
            exp_entry[slot] = {"w_bits": bw, "beta": beta}
        exp_entry["entry"] = {
            "w_bits": bw,
            "beta": 0.5 * (exp_entry["gate"]["beta"] + exp_entry["up"]["beta"]),
        }
        experts_meta[str(e_idx)] = exp_entry

    shared_deq: dict[str, torch.Tensor] = {}
    for slot, proj, (n_d, k_d) in (
        ("gate", "gate_proj", (moe_i, h_dim)),
        ("up", "up_proj", (moe_i, h_dim)),
        ("down", "down_proj", (h_dim, moe_i)),
    ):
        if quantize_shared:
            q_s = torch.randint(-8, 8, (n_d, k_d), generator=gen, dtype=torch.int16)
            sc_g = (torch.rand((k_d // 128, n_d), generator=gen, dtype=torch.float32) * 0.01 + 0.002).to(torch.float16)
            pkg_sh = _pack_sym_gptq_matrix(q_s, sc_g, 4, group_size=128)
            for suf, t in pkg_sh.items():
                st[f"model.layers.1.mlp.shared_experts.{proj}.{suf}"] = t
            w_deq = (
                q_s.to(torch.float32).reshape(n_d, k_d // 128, 128)
                * sc_g.transpose(0, 1).to(torch.float32).unsqueeze(-1)
            ).reshape(n_d, k_d)
            shared_deq[slot] = w_deq.to(torch.bfloat16)
        else:
            w_fp = (torch.randn((n_d, k_d), generator=gen) * 0.03).to(torch.bfloat16)
            st[f"model.layers.1.mlp.shared_experts.{proj}.weight"] = w_fp
            shared_deq[slot] = w_fp

    save_file(st, str(model_dir / "model.safetensors"))
    meta_doc = {
        "format": "scaling-moe.dxmoe.meta",
        "version": 1,
        "base_model_type": "deepseek_v2",
        "quantization": {"group_size": -1, "sym": True, "checkpoint_format": "gptq"},
        "hadamard": {"mode": "block128", "block_size": 128, "seed": 42, "signs_file": "dxmoe_signs.safetensors"},
        "layers": {
            "1": {
                "hidden_size": h_dim,
                "moe_intermediate_size": moe_i,
                "num_experts": e_cnt,
                "top_k": topk,
                "experts": experts_meta,
                "thresholds": {"5.0": {"tau_entry": [0.02, 0.25], "tau_down": [0.03, 0.35]}},
            }
        },
    }
    (model_dir / "dxmoe_meta.json").write_text(json.dumps(meta_doc, indent=2), encoding="utf-8")

    vllm_cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir),
            tokenizer=str(model_dir),
            dtype="bfloat16",
            quantization="dxmoe",
            max_model_len=128,
            enforce_eager=True,
            skip_tokenizer_init=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=128,
            max_model_len=128,
            is_encoder_decoder=False,
        ),
        kernel_config=KernelConfig(moe_backend="dxmoe"),
    )
    device = torch.device("cuda:0")
    if not is_workspace_manager_initialized():
        init_workspace_manager(device)
    need_cleanup = False
    try:
        with set_current_vllm_config(vllm_cfg):
            if not model_parallel_is_initialized():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    free_port = sock.getsockname()[1]
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=f"tcp://127.0.0.1:{free_port}",
                    local_rank=0,
                    backend="nccl",
                )
                need_cleanup = True
                ensure_model_parallel_initialized(1, 1)
            model = get_model(vllm_config=vllm_cfg)
        model.eval()

        moe_mlp = model.model.layers[1].mlp
        routed_qm = moe_mlp.experts.routed_experts.quant_method
        assert isinstance(routed_qm, DxMoEMoEMethod)
        assert routed_qm.checkpoint_is_gptq is True
        # Proxy qweight/scales/qzeros params removed after loading; 0-byte w13/w2 placeholders remain
        assert moe_mlp.experts.routed_experts.w13_weight.numel() == 0
        assert not hasattr(moe_mlp.experts.routed_experts, "w13_qweight")

        sh_gu_qm = moe_mlp.shared_experts.gate_up_proj.quant_method
        if quantize_shared:
            assert isinstance(sh_gu_qm, AutoGPTQLinearMethod)
            assert sh_gu_qm.input_dtype is None
        else:
            assert isinstance(sh_gu_qm, UnquantizedLinearMethod)

        x_in = (torch.randn((8, h_dim), generator=gen) * 0.5).to(device=device, dtype=torch.bfloat16)
        pos = torch.arange(8, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([8], dtype=torch.int32, device=device)
        pad = torch.zeros((8,), dtype=torch.bool, device=device)
        with (
            torch.no_grad(),
            set_forward_context(
                None,
                vllm_cfg,
                num_tokens=8,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                is_padding=pad,
                positions=pos,
                seq_lens=seq_lens,
            ),
        ):
            out_moe = moe_mlp(x_in)
        vllm_dxmoe.check_dxmoe_workspaces_healthy(device)
        assert out_moe.shape == (8, h_dim)
        assert bool(torch.isfinite(out_moe).all())
    finally:
        if need_cleanup:
            cleanup_dist_env_and_memory()


def test_dxmoe_w2_w3_shared_experts_dispatch_and_float_repack(tmp_path: Path) -> None:
    """CPU contract: W2/W3 GPTQ shared_experts dispatch to DxMoEWNA16LinearMethod and reject group_size != -1."""
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.quantization.dxmoe import DxMoEWNA16LinearMethod

    h_dim, i_dim, shared_i = 256, 256, 256
    for w_bits in (2, 3):
        meta_doc = _sample_dxmoe_meta(
            layers=(1,), hidden_size=h_dim, inter_size=i_dim, num_experts=2, top_k=1, w_bits=(w_bits, 3 if w_bits == 2 else 2)
        )
        case_dir = tmp_path / f"shared_w{w_bits}_g_neg1"
        case_dir.mkdir()
        (case_dir / "dxmoe_meta.json").write_text(json.dumps(meta_doc), encoding="utf-8")
        qcfg_doc = {
            "bits": w_bits,
            "group_size": -1,
            "sym": True,
            "desc_act": False,
            "quant_method": "gptq",
            "checkpoint_format": "gptq",
            "modules_in_block_to_quantize": ["mlp.experts.", "mlp.shared_experts."],
            "lm_head": False,
        }
        (case_dir / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v2", "quantization_config": qcfg_doc}),
            encoding="utf-8",
        )

        st_tensors: dict[str, torch.Tensor] = {
            "model.layers.1.self_attn.q_proj.weight": torch.zeros((h_dim, h_dim), dtype=torch.float16),
        }
        for e_idx, bw in enumerate((w_bits, 3 if w_bits == 2 else 2)):
            for proj, (n_d, k_d) in (
                ("gate_proj", (i_dim, h_dim)),
                ("up_proj", (i_dim, h_dim)),
                ("down_proj", (h_dim, i_dim)),
            ):
                pkg = _pack_sym_gptq_matrix(
                    torch.zeros((n_d, k_d), dtype=torch.int16),
                    torch.ones((n_d,), dtype=torch.float16),
                    bw,
                    group_size=-1,
                )
                for suf, t in pkg.items():
                    st_tensors[f"model.layers.1.mlp.experts.{e_idx}.{proj}.{suf}"] = t

        for proj, (n_d, k_d) in (
            ("gate_proj", (shared_i, h_dim)),
            ("up_proj", (shared_i, h_dim)),
            ("down_proj", (h_dim, shared_i)),
        ):
            pkg_s = _pack_sym_gptq_matrix(
                torch.zeros((n_d, k_d), dtype=torch.int16),
                torch.ones((1, n_d), dtype=torch.float16),
                w_bits,
                group_size=-1,
            )
            for suf, t in pkg_s.items():
                st_tensors[f"model.layers.1.mlp.shared_experts.{proj}.{suf}"] = t
        save_file(st_tensors, str(case_dir / "model.safetensors"))

        dx_cfg = DxMoEConfig({"_model_dir": str(case_dir), **qcfg_doc})
        dx_cfg.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        dummy_linear = object.__new__(MergedColumnParallelLinear)
        dummy_down = object.__new__(RowParallelLinear)
        sh_gu_m = dx_cfg.get_quant_method(dummy_linear, "model.layers.1.mlp.shared_experts.gate_up_proj")
        sh_dn_m = dx_cfg.get_quant_method(dummy_down, "model.layers.1.mlp.shared_experts.down_proj")
        assert isinstance(sh_gu_m, DxMoEWNA16LinearMethod) and sh_gu_m.w_bits == w_bits and sh_gu_m.input_dtype is None
        assert isinstance(sh_dn_m, DxMoEWNA16LinearMethod) and sh_dn_m.w_bits == w_bits and sh_dn_m.input_dtype is None

        # Reject group_size=128 on W2/W3 non-routed linear
        bad_dir = tmp_path / f"shared_w{w_bits}_g128"
        bad_dir.mkdir()
        (bad_dir / "dxmoe_meta.json").write_text(json.dumps(meta_doc), encoding="utf-8")
        bad_qcfg = {**qcfg_doc, "dynamic": {r"+:.*mlp\.shared_experts\..*": {"bits": w_bits, "group_size": 128}}}
        (bad_dir / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v2", "quantization_config": bad_qcfg}),
            encoding="utf-8",
        )
        bad_st = dict(st_tensors)
        for proj, (n_d, k_d) in (
            ("gate_proj", (shared_i, h_dim)),
            ("up_proj", (shared_i, h_dim)),
            ("down_proj", (h_dim, shared_i)),
        ):
            pkg_bad = _pack_sym_gptq_matrix(
                torch.zeros((n_d, k_d), dtype=torch.int16),
                torch.ones((k_d // 128, n_d), dtype=torch.float16),
                w_bits,
                group_size=128,
            )
            for suf, t in pkg_bad.items():
                bad_st[f"model.layers.1.mlp.shared_experts.{proj}.{suf}"] = t
        save_file(bad_st, str(bad_dir / "model.safetensors"))
        bad_dx_cfg = DxMoEConfig({"_model_dir": str(bad_dir), **bad_qcfg})
        bad_dx_cfg.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        with pytest.raises(ValueError, match="requires channel-wise group_size=-1"):
            bad_dx_cfg.get_quant_method(dummy_linear, "model.layers.1.mlp.shared_experts.gate_up_proj")

    # Verify float matrix packing for W2 and W3 (--export-format float path)
    gen = torch.Generator(device="cpu").manual_seed(909)
    for w_bits, lo, hi, zero in ((2, -2, 2, 2), (3, -4, 4, 4)):
        q_s = torch.randint(lo, hi, (128, 256), generator=gen, dtype=torch.int16)
        sc = (torch.rand((128,), generator=gen, dtype=torch.float32) * 0.05 + 0.01).to(torch.float16)
        w_f = (q_s.float() * sc.float().unsqueeze(-1)).to(torch.bfloat16)
        packed_codes, fit_sc = vllm_dxmoe.pack_dxmoe_float_matrix(w_f, w_bits)
        assert packed_codes.shape == (128, vllm_dxmoe.dxmoe_packed_row_bytes(256, w_bits))
        dummy_lw = vllm_dxmoe.finalize_dxmoe_weight_channel(
            num_experts=1,
            n=128,
            k=256,
            w_bits_list=(w_bits,),
            codes_arena=packed_codes.reshape(-1),
            scales=fit_sc.unsqueeze(0),
            w_bits_dev=torch.tensor([w_bits], dtype=torch.int32),
            expert_offsets=(0,),
        )
        rec = _dequant_packed_w(dummy_lw, 0, w_bits)
        rel_err = (rec - w_f.float()).norm() / w_f.float().norm()
        assert float(rel_err.item()) < (2.5e-2 if w_bits == 2 else 1.5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single-GPU test requires CUDA")
def test_gpu_dxmoe_wna16_linear_w2_w3_parity_padding_and_cudagraph() -> None:
    """Verify native W2A16/W3A16 DxMoEWNA16LinearMethod across aligned & padded shapes, FP16/BF16, and CUDAGraph."""
    cap = torch.cuda.get_device_capability(0)
    if cap not in ((8, 0), (8, 6)):
        pytest.skip(f"DxMoE native kernel requires SM80/SM86, got {cap}")

    from vllm.model_executor.layers.quantization.dxmoe import DxMoEWNA16LinearMethod

    device = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(20260401)

    for w_bits, lo, hi in ((2, -2, 2), (3, -4, 4)):
        # Test both exact (k=256, n=512) and padded (k=160 -> 256, n=192 -> 256) shapes!
        for k_in, n_half in ((256, 256), (160, 96)):
            n_total = 2 * n_half
            layer = torch.nn.Module()
            method = DxMoEWNA16LinearMethod(w_bits=w_bits, prefix=f"test.w{w_bits}.gate_up_proj")
            method.create_weights(
                layer,
                input_size_per_partition=k_in,
                output_partition_sizes=[n_half, n_half],
                input_size=k_in,
                output_size=n_total,
                params_dtype=torch.float16,
            )
            layer.to(device)

            q0 = torch.randint(lo, hi, (n_half, k_in), generator=gen, dtype=torch.int16)
            q1 = torch.randint(lo, hi, (n_half, k_in), generator=gen, dtype=torch.int16)
            if w_bits == 3:
                for blk in range(k_in // 32):
                    q0[:, blk * 32 + 10] = ((torch.arange(n_half) + blk) % 8 - 4).to(torch.int16)
                    q1[:, blk * 32 + 21] = ((torch.arange(n_half) * 3 + blk) % 8 - 4).to(torch.int16)
            sc0 = (torch.rand((n_half,), generator=gen, dtype=torch.float32) * 0.04 + 0.005).to(torch.float16)
            sc1 = (torch.rand((n_half,), generator=gen, dtype=torch.float32) * 0.04 + 0.005).to(torch.float16)
            pkg0 = _pack_sym_gptq_matrix(q0, sc0, w_bits, group_size=-1)
            pkg1 = _pack_sym_gptq_matrix(q1, sc1, w_bits, group_size=-1)

            for shard_id, pkg in ((0, pkg0), (1, pkg1)):
                layer.qweight.weight_loader(layer.qweight, pkg["qweight"].to(device), shard_id)
                layer.scales.weight_loader(layer.scales, pkg["scales"].to(device), shard_id)
                layer.qzeros.weight_loader(layer.qzeros, pkg["qzeros"].to(device), shard_id)
                layer.g_idx.weight_loader(layer.g_idx, pkg["g_idx"].to(device))

            method.process_weights_after_loading(layer)
            assert not hasattr(layer, "qweight") and hasattr(layer, "_dxmoe_wna16_codes")

            w_deq = torch.cat(
                (q0.float() * sc0.float().unsqueeze(-1), q1.float() * sc1.float().unsqueeze(-1)), dim=0
            ).to(device=device, dtype=torch.float16)

            for act_dtype in (torch.float16, torch.bfloat16):
                x = (torch.randn((19, k_in), generator=gen, dtype=torch.float32) * 0.8).to(
                    device=device, dtype=act_dtype
                )
                got = method.apply(layer, x)
                ref = F.linear(x.to(torch.float16), w_deq).to(act_dtype)
                vllm_dxmoe.check_dxmoe_workspaces_healthy(device)
                torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)

            # CUDAGraph capture & replay on padded shape
            if k_in == 160:
                gx = (torch.randn((11, k_in), generator=gen, dtype=torch.float32) * 0.7).to(
                    device=device, dtype=torch.bfloat16
                )
                s = torch.cuda.Stream(device=device)
                with torch.cuda.stream(s):
                    _ = method.apply(layer, gx)
                s.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    g_out = method.apply(layer, gx)
                nx = (torch.randn((11, k_in), generator=gen, dtype=torch.float32) * 1.1).to(
                    device=device, dtype=torch.bfloat16
                )
                gx.copy_(nx)
                with torch.cuda.stream(s):
                    g.replay()
                s.synchronize()
                vllm_dxmoe.check_dxmoe_workspaces_healthy(device)
                ref_g = F.linear(gx.to(torch.float16), w_deq).to(torch.bfloat16)
                torch.testing.assert_close(g_out, ref_g, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single-GPU test requires CUDA")
def test_gpu_dxmoe_experts_heterogeneous_w2_w3_w4_w8_layer_parity() -> None:
    """Verify vLLM DxMoEExperts across heterogeneous (2, 3, 4, 8) static weights and all schedule modes."""
    cap = torch.cuda.get_device_capability(0)
    if cap not in ((8, 0), (8, 6)):
        pytest.skip(f"DxMoE native kernel requires SM80/SM86, got {cap}")

    device = torch.device("cuda:0")
    gen = torch.Generator(device="cpu").manual_seed(20260402)
    e_cnt, h_dim, i_dim, topk = 4, 256, 256, 3
    w_bits = (2, 3, 4, 8)

    def _build_gptq_packed_channel(n_dim: int, k_dim: int) -> vllm_dxmoe.DxMoELayerWeights:
        codes_arena, scales, w_bits_dev, offsets = vllm_dxmoe.allocate_dxmoe_weight_buffers(
            e_cnt, n_dim, k_dim, w_bits, device=device
        )
        for e, b in enumerate(w_bits):
            lo = -(1 << (b - 1)) if b < 8 else -127
            hi = 1 << (b - 1)
            q_s = torch.randint(lo, hi, (n_dim, k_dim), generator=gen, dtype=torch.int16)
            if b == 3:
                for blk in range(k_dim // 32):
                    q_s[:, blk * 32 + 10] = ((torch.arange(n_dim) + blk) % 8 - 4).to(torch.int16)
                    q_s[:, blk * 32 + 21] = ((torch.arange(n_dim) * 5 + blk) % 8 - 4).to(torch.int16)
            sc = (torch.rand((n_dim,), generator=gen, dtype=torch.float32) * (0.04 / b) + 0.003).to(torch.float16)
            pkg = _pack_sym_gptq_matrix(q_s, sc, b, group_size=-1)
            repacked = vllm_dxmoe.repack_gptq_qweight_shard(
                pkg["qweight"].to(device), k_expected=k_dim, n_expected=n_dim, w_bits=b, target_device=device
            )
            vllm_dxmoe.validate_gptq_qzeros_shard(pkg["qzeros"], n_expected=n_dim, w_bits=b)
            nbytes = repacked.numel()
            codes_arena[offsets[e] : offsets[e] + nbytes].copy_(repacked.view(-1))
            scales[e].copy_(sc.to(device))
        return vllm_dxmoe.finalize_dxmoe_weight_channel(
            num_experts=e_cnt,
            n=n_dim,
            k=k_dim,
            w_bits_list=w_bits,
            codes_arena=codes_arena,
            scales=scales,
            w_bits_dev=w_bits_dev,
            expert_offsets=offsets,
        )

    w13_packed = _build_gptq_packed_channel(2 * i_dim, h_dim)
    w2_packed = _build_gptq_packed_channel(h_dim, i_dim)
    s_in = (torch.randint(0, 2, (h_dim,), generator=gen) * 2 - 1).to(dtype=torch.int8, device=device)
    s_mid = (torch.randint(0, 2, (i_dim,), generator=gen) * 2 - 1).to(dtype=torch.int8, device=device)
    beta_e = torch.tensor([0.6, 1.1, 0.9, 1.4], dtype=torch.float64, device=device)
    beta_d = torch.tensor([0.7, 1.0, 1.2, 0.8], dtype=torch.float64, device=device)
    tau_e = (0.04, 0.35)
    tau_d = (0.02, 0.25)

    moe_cfg = _make_moe_config(num_experts=e_cnt, top_k=topk, hidden_size=h_dim, inter_size=i_dim)
    moe_cfg.device = device
    dummy_w = torch.empty(0, device=device)

    for sched_mode in ("dxmoe", "static_a4", "static_a8", "static_a16"):
        coup = sched_mode in ("dxmoe", "static_a4")
        state = DxMoELayerRuntimeState(
            layer_idx=1,
            hidden_size=h_dim,
            inter_size=i_dim,
            num_experts=e_cnt,
            top_k=topk,
            w13=w13_packed,
            w2=w2_packed,
            s_in=s_in,
            s_mid=s_mid,
            beta_entry=beta_e,
            beta_down=beta_d,
            tau_entry=tau_e,
            tau_down=tau_d,
            schedule_mode=sched_mode,
            apply_coupling_entry=coup,
            apply_coupling_down=coup,
            use_omega=False,
            sink_tokens=4,
            local_window=32,
            pipelined_code=1,
        )
        op = DxMoEExperts(moe_config=moe_cfg, quant_config=DxMoEMoEQuantConfig(state))

        for act_dtype in (torch.float16, torch.bfloat16):
            x = (torch.randn((16, h_dim), generator=gen) * 0.75).to(device=device, dtype=act_dtype)
            probs = torch.softmax(torch.randn((16, e_cnt), generator=gen).to(device=device), dim=-1)
            rw_f32, ids_i64 = torch.topk(probs, k=topk, dim=-1)
            rw = (rw_f32 / rw_f32.sum(dim=-1, keepdim=True)).to(dtype=act_dtype).contiguous()
            ids = ids_i64.to(dtype=torch.int32).contiguous()
            out = torch.empty_like(x)

            op.apply(
                output=out,
                hidden_states=x,
                w1=dummy_w,
                w2=dummy_w,
                topk_weights=rw,
                topk_ids=ids,
                activation=MoEActivation.SILU,
                global_num_experts=e_cnt,
                expert_map=None,
                a1q_scale=None,
                a2_scale=None,
                workspace13=dummy_w,
                workspace2=dummy_w,
                expert_tokens_meta=None,
                apply_router_weight_on_input=False,
            )
            vllm_dxmoe.check_dxmoe_workspaces_healthy(device)

            xt = _hadamard128_ref(x, s_in)
            ref_acc = torch.zeros((16, h_dim), dtype=torch.float32, device=device)
            for t in range(16):
                am1 = float(xt[t].float().abs().amax().item())
                for s in range(topk):
                    e = int(ids[t, s].item())
                    g = float(rw[t, s].float().item())
                    if sched_mode == "dxmoe":
                        sc1 = (g * g) * float(beta_e[e].item()) * (am1 * am1)
                        ab1 = 4 if sc1 < tau_e[0] else (8 if sc1 < tau_e[1] else 16)
                    else:
                        ab1 = int(sched_mode.split("_a")[1])
                    if coup and w_bits[e] == 8 and ab1 < 8:
                        ab1 = 8
                    xa = _fake_quant_act_row(xt[t].float(), ab1)
                    dw13 = _dequant_packed_w(w13_packed, e, w_bits[e])
                    if ab1 == 16:
                        dw13 = dw13.to(torch.float16).float()
                    gu = (xa @ dw13.T).to(torch.float16)
                    hm = _hadamard128_ref(
                        (F.silu(gu[:i_dim].float()) * gu[i_dim:].float()).unsqueeze(0),
                        s_mid,
                        out_dtype=act_dtype,
                    )[0]
                    am2 = float(hm.float().abs().amax().item())
                    if sched_mode == "dxmoe":
                        sc2 = (g * g) * float(beta_d[e].item()) * (am2 * am2)
                        ab2 = 4 if sc2 < tau_d[0] else (8 if sc2 < tau_d[1] else 16)
                    else:
                        ab2 = int(sched_mode.split("_a")[1])
                    if coup and w_bits[e] == 8 and ab2 < 8:
                        ab2 = 8
                    ha = _fake_quant_act_row(hm.float(), ab2)
                    dw2 = _dequant_packed_w(w2_packed, e, w_bits[e])
                    if ab2 == 16:
                        dw2 = dw2.to(torch.float16).float()
                    ref_acc[t] += g * (ha @ dw2.T).to(torch.float16).float()
            ref_cast = ref_acc.to(act_dtype)
            rel_l2 = (out.float() - ref_cast.float()).norm() / ref_cast.float().norm()
            assert float(rel_l2.item()) < 8e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Single-GPU test requires CUDA")
def test_gpu_real_offline_export_and_vllm_generation_w2_w3_deepseek_and_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: offline W2/W3 GPTQ export + threshold calibration -> real vLLM loading & token generation!

    Covers both DeepseekV2ForCausalLM (rotated W2/W3 shared_experts with s_mid_shared) and
    Qwen2MoeForCausalLM (rotated W2/W3 shared_expert + unrotated shared_expert_gate).
    """
    cap = torch.cuda.get_device_capability(0)
    if cap not in ((8, 0), (8, 6)):
        pytest.skip(f"DxMoE native kernel requires SM80/SM86, got {cap}")

    import os
    import socket
    import sys

    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from safetensors import safe_open
    from transformers import (
        DeepseekV2Config,
        DeepseekV2ForCausalLM,
        Qwen2MoeConfig,
        Qwen2MoeForCausalLM,
    )
    from scaling_moe.dxmoe import (
        calibrate_dxmoe_thresholds,
        load_dxmoe_hf_causal_lm,
        load_dxmoe_shared_mid_signs,
        load_dxmoe_signs,
        quantize_model_dxmoe_hf,
        save_dxmoe_meta,
    )
    from vllm import LLM, SamplingParams
    from vllm.config import ModelConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )
    from vllm.inputs import TokensPrompt
    from vllm.model_executor.layers.quantization.dxmoe import (
        DxMoESharedFFNRotation,
        DxMoEWNA16LinearMethod,
    )
    from vllm.model_executor.model_loader import get_model
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    if os.environ.get("DXMOE_SHORT_LIB") and not os.environ.get("VLLM_DXMOE_SHORT_LIB"):
        monkeypatch.setenv("VLLM_DXMOE_SHORT_LIB", os.environ["DXMOE_SHORT_LIB"])
    if os.environ.get("DXMOE_LONG_LIB") and not os.environ.get("VLLM_DXMOE_LONG_LIB"):
        monkeypatch.setenv("VLLM_DXMOE_LONG_LIB", os.environ["DXMOE_LONG_LIB"])
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def _save_per_expert_base_model(model_obj: torch.nn.Module, cfg_obj, out_base: Path, n_exp: int, i_moe: int) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        out_base.mkdir(parents=True)
        cfg_obj.architectures = [type(model_obj).__name__]
        cfg_obj.save_pretrained(out_base)
        vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
        for idx in range(4, int(cfg_obj.vocab_size)):
            vocab[f"tok{idx}"] = idx
        tok_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tok_backend.pre_tokenizer = Whitespace()
        fast_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok_backend,
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
        )
        fast_tok.save_pretrained(out_base)

        sd = model_obj.state_dict()
        disk_sd: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.endswith(".mlp.experts.gate_up_proj"):
                stem = k[: -len(".gate_up_proj")]
                for e in range(n_exp):
                    disk_sd[f"{stem}.{e}.gate_proj.weight"] = v[e, :i_moe, :].contiguous()
                    disk_sd[f"{stem}.{e}.up_proj.weight"] = v[e, i_moe:, :].contiguous()
            elif k.endswith(".mlp.experts.down_proj"):
                stem = k[: -len(".down_proj")]
                for e in range(n_exp):
                    disk_sd[f"{stem}.{e}.down_proj.weight"] = v[e].contiguous()
            else:
                disk_sd[k] = v.contiguous()
        save_file(disk_sd, str(out_base / "model.safetensors"))

    def _quantize_and_calibrate_w23(base_dir: Path, quant_dir: Path, vocab_sz: int, seed: int) -> dict:
        g_cal = torch.Generator().manual_seed(seed)
        calib_batches = [torch.randint(0, vocab_sz, (1, 24), generator=g_cal, dtype=torch.long) for _ in range(4)]
        meta = quantize_model_dxmoe_hf(
            base_dir,
            quant_dir,
            calibration_batches=calib_batches,
            group_size=-1,
            sym=True,
            candidate_w_bits=(2, 3),
            budget_bpw=2.5,
            hadamard_mode="block128",
            export_format="gptq",
            device="cuda:0",
            seed=seed,
        )
        signs = load_dxmoe_signs(quant_dir, meta)
        shared_mid = load_dxmoe_shared_mid_signs(quant_dir, meta)
        hf_q_model = load_dxmoe_hf_causal_lm(quant_dir, dtype=torch.bfloat16).to("cuda:0")
        cal_meta = calibrate_dxmoe_thresholds(
            hf_q_model,
            meta,
            calib_batches,
            signs_by_layer=signs,
            shared_mid_signs=shared_mid,
            threshold_pool="joint",
        )
        del hf_q_model
        torch.cuda.empty_cache()
        save_dxmoe_meta(quant_dir / "dxmoe_meta.json", cal_meta)
        return cal_meta

    # --- 1) Miniature DeepseekV2ForCausalLM (W2/W3 routed + W2/W3 rotated shared_experts with s_mid_shared) ---
    torch.manual_seed(4101)
    ds_cfg = DeepseekV2Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=2,  # shared intermediate = 512 != 256 -> exercises s_mid_shared!
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scoring_func="softmax",
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    ds_base_model = DeepseekV2ForCausalLM(ds_cfg).to(dtype=torch.bfloat16).eval()
    ds_base_dir = tmp_path / "ds2_base"
    ds_quant_dir = tmp_path / "ds2_w23_gptq"
    _save_per_expert_base_model(ds_base_model, ds_cfg, ds_base_dir, n_exp=4, i_moe=256)
    del ds_base_model

    ds_meta = _quantize_and_calibrate_w23(ds_base_dir, ds_quant_dir, vocab_sz=128, seed=71)
    ds_l1 = ds_meta["layers"]["1"]
    assert ds_l1["shared_ffn_rotated"] is True
    ds_wbits = {
        ds_l1["experts"][str(e)][slot]["w_bits"]
        for e in range(4)
        for slot in ("entry", "down")
    }
    assert ds_wbits == {2, 3}, f"Expected both W2 and W3 in ILP allocation, got {ds_wbits}"
    with safe_open(str(ds_quant_dir / "dxmoe_signs.safetensors"), framework="pt", device="cpu") as sr:
        assert "layer.1.s_mid_shared" in sr.keys()

    # --- 2) Miniature Qwen2MoeForCausalLM (W2/W3 routed + W2/W3 rotated shared_expert + unrotated shared_expert_gate) ---
    torch.manual_seed(4102)
    qw_cfg = Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=512,  # != 256 -> exercises s_mid_shared!
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    qw_base_model = Qwen2MoeForCausalLM(qw_cfg).to(dtype=torch.bfloat16).eval()
    qw_base_dir = tmp_path / "qwen2_base"
    qw_quant_dir = tmp_path / "qwen2_w23_gptq"
    _save_per_expert_base_model(qw_base_model, qw_cfg, qw_base_dir, n_exp=4, i_moe=256)
    del qw_base_model

    qw_meta = _quantize_and_calibrate_w23(qw_base_dir, qw_quant_dir, vocab_sz=128, seed=73)
    for lk in ("0", "1"):
        assert qw_meta["layers"][lk]["shared_ffn_rotated"] is True
    qw_wbits = {
        qw_meta["layers"][lk]["experts"][str(e)][slot]["w_bits"]
        for lk in ("0", "1")
        for e in range(4)
        for slot in ("entry", "down")
    }
    assert qw_wbits == {2, 3}, f"Expected both W2 and W3 in Qwen ILP allocation, got {qw_wbits}"

    # Verify live Qwen2MoeSparseMoeBlock module wiring & unrotated shared_expert_gate input in vLLM!
    device = torch.device("cuda:0")
    if not is_workspace_manager_initialized():
        init_workspace_manager(device)
    qw_vllm_cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(qw_quant_dir),
            tokenizer=str(qw_quant_dir),
            dtype="bfloat16",
            quantization="dxmoe",
            max_model_len=64,
            enforce_eager=True,
            skip_tokenizer_init=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=64,
            max_model_len=64,
            is_encoder_decoder=False,
        ),
        kernel_config=KernelConfig(moe_backend="dxmoe"),
    )
    need_cleanup = False
    try:
        with set_current_vllm_config(qw_vllm_cfg):
            if not model_parallel_is_initialized():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    free_port = sock.getsockname()[1]
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=f"tcp://127.0.0.1:{free_port}",
                    local_rank=0,
                    backend="nccl",
                )
                need_cleanup = True
                ensure_model_parallel_initialized(1, 1)
            qw_vllm_model = get_model(vllm_config=qw_vllm_cfg)
        qw_vllm_model.eval()

        q_block = qw_vllm_model.model.layers[0].mlp
        assert isinstance(q_block.experts.routed_experts.quant_method, DxMoEMoEMethod)
        assert isinstance(q_block.shared_expert.gate_up_proj.quant_method, DxMoEWNA16LinearMethod)
        assert isinstance(q_block.shared_expert.down_proj.quant_method, DxMoEWNA16LinearMethod)
        assert q_block.shared_expert.gate_up_proj.quant_method.w_bits in (2, 3)
        assert isinstance(q_block.shared_expert._dxmoe_shared_rotation, DxMoESharedFFNRotation)
        assert isinstance(q_block.shared_expert_gate.quant_method, UnquantizedLinearMethod)

        captured_gate_inputs: list[torch.Tensor] = []
        captured_gu_inputs: list[torch.Tensor] = []
        h_gate = q_block.shared_expert_gate.register_forward_pre_hook(
            lambda _m, args: captured_gate_inputs.append(args[0].detach().clone())
        )
        h_gu = q_block.shared_expert.gate_up_proj.register_forward_pre_hook(
            lambda _m, args: captured_gu_inputs.append(args[0].detach().clone())
        )
        try:
            x_probe = torch.randn((6, 256), dtype=torch.bfloat16, device=device)
            pos_p = torch.arange(6, dtype=torch.int64, device=device)
            seq_p = torch.tensor([6], dtype=torch.int32, device=device)
            pad_p = torch.zeros((6,), dtype=torch.bool, device=device)
            with (
                torch.no_grad(),
                set_forward_context(
                    None,
                    qw_vllm_cfg,
                    num_tokens=6,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    is_padding=pad_p,
                    positions=pos_p,
                    seq_lens=seq_p,
                ),
            ):
                y_probe = q_block(x_probe)
            vllm_dxmoe.check_dxmoe_workspaces_healthy(device)
            assert y_probe.shape == (6, 256) and bool(torch.isfinite(y_probe).all())
            assert len(captured_gate_inputs) == 1 and len(captured_gu_inputs) == 1
            # Qwen's shared_expert_gate MUST consume the original unrotated input x_probe,
            # while shared_expert.gate_up_proj consumes block128-rotated input!
            assert torch.equal(captured_gate_inputs[0], x_probe)
            expected_rot = q_block.shared_expert._dxmoe_shared_rotation.input(x_probe)
            assert torch.equal(captured_gu_inputs[0], expected_rot)
            assert not torch.equal(captured_gate_inputs[0], captured_gu_inputs[0])
        finally:
            h_gate.remove()
            h_gu.remove()
        del qw_vllm_model
    finally:
        if need_cleanup:
            cleanup_dist_env_and_memory()
    torch.cuda.empty_cache()

    # --- 3) Real full vLLM Engine loading & multi-token generation for BOTH W2/W3 checkpoints ---
    import gc

    prompt = "tok4 tok7 tok13 tok19 tok25 tok31 tok37 tok43"
    sampling = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True)
    for model_label, ckpt_dir in (("deepseek_v2_w23", ds_quant_dir), ("qwen2_moe_w23", qw_quant_dir)):
        gc.collect()
        torch.cuda.empty_cache()
        llm = LLM(
            model=str(ckpt_dir),
            runner="generate",
            dtype="bfloat16",
            quantization="dxmoe",
            max_model_len=64,
            max_num_seqs=1,
            kv_cache_memory_bytes=64 * 1024 * 1024,
            enforce_eager=True,
        )
        try:
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            assert len(outputs) == 1, f"{model_label}: expected 1 RequestOutput"
            gen_ids = list(outputs[0].outputs[0].token_ids)
            assert len(gen_ids) == 6, f"{model_label}: expected 6 generated tokens, got {gen_ids}"
            assert all(0 <= int(tid) < 128 for tid in gen_ids), f"{model_label}: invalid token IDs {gen_ids}"
            assert isinstance(outputs[0].outputs[0].text, str)
            # Repeat greedy generation to confirm determinism and clean workspace state across steps!
            outputs_repeat = llm.generate([prompt], sampling, use_tqdm=False)
            assert list(outputs_repeat[0].outputs[0].token_ids) == gen_ids
        finally:
            del llm
            cleanup_dist_env_and_memory()
            torch.cuda.empty_cache()
