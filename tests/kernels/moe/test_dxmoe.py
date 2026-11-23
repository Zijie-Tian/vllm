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

    # 2) Reject W16, W2, W3 weights.
    for bad_bw in (16, 2, 3):
        bad_meta = _sample_dxmoe_meta(w_bits=(4, bad_bw, 4, 8))
        with pytest.raises(ValueError, match="supports W4/W8 static weights only"):
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
    row_bytes = k // 2 if w_bits == 4 else k
    base_off = int(packed.w_ptrs_dev[expert].item()) - int(packed.codes_arena.data_ptr())
    raw = packed.codes_arena[base_off : base_off + n * row_bytes].reshape(n, row_bytes)
    if w_bits == 4:
        lo = (raw & 0x0F).to(torch.int16) - 8
        hi = (raw >> 4).to(torch.int16) - 8
        q = torch.stack((lo, hi), dim=-1).reshape(n, k).float()
    else:
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
