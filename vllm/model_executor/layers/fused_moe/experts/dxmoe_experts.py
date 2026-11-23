# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Modular FusedMoE expert execution backend for DxMoE."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.utils.dxmoe import (
    DTYPE_CODE,
    PIPELINED_CODE,
    SCHEDULE_MODE_CODE,
    DxMoELayerWeights,
    DxMoESharedWorkspace,
    NativeOmegaDesc,
    NativePolicyDesc,
    NativeReorderDesc,
    NativeUnpermuteDesc,
    get_dxmoe_lib_bundle,
    get_dxmoe_shared_workspace,
)


@dataclass(frozen=True)
class DxMoELayerRuntimeState:
    """Device-resident per-layer static weights, Hadamard signs, and policy priors."""

    layer_idx: int
    hidden_size: int
    inter_size: int
    num_experts: int
    top_k: int
    w13: DxMoELayerWeights
    w2: DxMoELayerWeights
    s_in: torch.Tensor | None  # int8[H] or None
    s_mid: torch.Tensor | None  # int8[I] or None
    beta_entry: torch.Tensor  # float64[E]
    beta_down: torch.Tensor  # float64[E]
    tau_entry: tuple[float, float]
    tau_down: tuple[float, float]
    schedule_mode: str
    apply_coupling_entry: bool
    apply_coupling_down: bool
    use_omega: bool
    sink_tokens: int
    local_window: int
    pipelined_code: int


class DxMoEMoEQuantConfig(FusedMoEQuantConfig):
    """FusedMoEQuantConfig wrapper carrying per-layer DxMoE runtime state."""

    dxmoe_state: DxMoELayerRuntimeState

    def __init__(self, dxmoe_state: DxMoELayerRuntimeState) -> None:
        super().__init__(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(),
            _w2=FusedMoEQuantDesc(),
        )
        object.__setattr__(self, "dxmoe_state", dxmoe_state)


def _extract_first_attn_metadata(fc: ForwardContext) -> Any | None:
    meta = fc.attn_metadata
    if isinstance(meta, dict) and meta:
        return next(iter(meta.values()))
    if isinstance(meta, list) and meta and isinstance(meta[0], dict) and meta[0]:
        return next(iter(meta[0].values()))
    return None


class DxMoEExperts(mk.FusedMoEExpertsModular):
    """5-step pure-GPU zero-copy MoE pipeline for DxMoE (B=1 scope)."""

    quant_config: DxMoEMoEQuantConfig

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ) -> None:
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self.dxmoe_state = cast(DxMoEMoEQuantConfig, quant_config).dxmoe_state
        self.schedule_code = SCHEDULE_MODE_CODE[self.dxmoe_state.schedule_mode]

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_cuda():
            return False
        cap = current_platform.get_device_capability()
        return cap is not None and (cap[0] * 10 + cap[1]) >= 80

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (
            moe_parallel_config.tp_size == 1
            and moe_parallel_config.dp_size == 1
            and moe_parallel_config.ep_size == 1
            and not moe_parallel_config.use_ep
            and not moe_parallel_config.enable_eplb
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Internal buffers live in DxMoESharedWorkspace; only the [M, K] final
        # output (aliased directly to FusedMoEKernel's output) is needed here.
        return (0,), (0,), (M, K)

    def _resolve_b1_omega_and_padding(
        self,
        *,
        num_tokens: int,
        ws: DxMoESharedWorkspace,
        lib: ctypes.CDLL,
        stream_ptr: ctypes.c_void_p,
    ) -> tuple[int, float, int, int]:
        """Return `(omega_ptr, scalar_omega, is_padding_ptr, num_valid_tokens)`."""
        state = self.dxmoe_state
        fc: ForwardContext | None = (
            get_forward_context() if is_forward_context_available() else None
        )
        is_capturing = torch.cuda.is_current_stream_capturing()
        in_cudagraph = is_capturing or (
            fc is not None and fc.cudagraph_runtime_mode != CUDAGraphMode.NONE
        )

        is_padding_ptr = 0
        if fc is not None and fc.is_padding is not None:
            pad_t = fc.is_padding
            if (
                pad_t.is_contiguous()
                and pad_t.dtype in (torch.bool, torch.uint8)
                and pad_t.numel() >= num_tokens
            ):
                is_padding_ptr = int(pad_t.data_ptr())

        num_valid_tokens = num_tokens
        attn_meta = _extract_first_attn_metadata(fc) if fc is not None else None
        if not in_cudagraph and fc is not None:
            num_reqs = getattr(attn_meta, "num_reqs", None)
            if num_reqs is None and fc.batch_descriptor is not None:
                num_reqs = fc.batch_descriptor.num_reqs
            if num_reqs is not None and int(num_reqs) > 1:
                raise NotImplementedError(
                    f"DxMoE vLLM backend currently supports single-request B=1 only "
                    f"(got num_reqs={num_reqs}); set max_num_seqs=1."
                )
            if attn_meta is not None:
                actual = getattr(attn_meta, "num_actual_tokens", None)
                if actual is not None:
                    num_valid_tokens = max(0, min(num_tokens, int(actual)))

        # Static schedule modes do not read omega at all.
        if self.schedule_code != 0:
            return 0, 1.0, is_padding_ptr, num_valid_tokens

        if not state.use_omega:
            return 0, 1.0, is_padding_ptr, num_valid_tokens

        # Single-token B=1 Decode: token sits at pos = L - 1 >= L - local_window -> omega = 2.0f.
        if num_tokens == 1 and state.local_window >= 1:
            return 0, 2.0, is_padding_ptr, num_valid_tokens

        # Check ForwardContext cache so all 26 MoE layers share one compute_omega launch.
        cache_key = ("_dxmoe_omega_ready", num_tokens, state.sink_tokens, state.local_window)
        if fc is not None and fc.additional_kwargs.get(cache_key) is True:
            return int(ws.omega_buf.data_ptr()), 1.0, is_padding_ptr, num_valid_tokens

        past_seq_len = 0
        uniform_seq_end = num_tokens
        omega_padding_ptr = is_padding_ptr
        if not in_cudagraph and attn_meta is not None:
            max_seq_len = getattr(attn_meta, "max_seq_len", None)
            actual = getattr(attn_meta, "num_actual_tokens", None)
            if max_seq_len is not None and actual is not None and int(actual) > 0:
                seq_end_val = int(max_seq_len)
                act_val = int(actual)
                past_seq_len = max(0, seq_end_val - act_val)
                uniform_seq_end = seq_end_val
                omega_padding_ptr = 0

        omega_desc = NativeOmegaDesc(
            position_ids=0,
            seq_ends=0,
            is_padding=omega_padding_ptr,
            out_omega=int(ws.omega_buf.data_ptr()),
            past_seq_len=past_seq_len,
            uniform_seq_end=uniform_seq_end,
            batch_size=1,
            seq_len=num_tokens,
            pos_batch_stride=0,
            sink_tokens=state.sink_tokens,
            local_window=state.local_window,
            mask_past_seq_end=1,
        )
        err = int(lib.smoe_dxmoe_compute_omega(ctypes.byref(omega_desc), stream_ptr))
        if err != 0:
            raise RuntimeError(f"smoe_dxmoe_compute_omega failed with CUDA status {err}")

        if fc is not None:
            fc.additional_kwargs[cache_key] = True
        return int(ws.omega_buf.data_ptr()), 1.0, is_padding_ptr, num_valid_tokens

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "DxMoEExperts does not support apply_router_weight_on_input=True"
            )
        if expert_map is not None:
            raise NotImplementedError("DxMoEExperts requires EP=1 (expert_map=None)")

        state = self.dxmoe_state
        x_2d = hidden_states.view(-1, state.hidden_size)
        num_tokens = int(x_2d.shape[0])
        if num_tokens == 0:
            return

        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()

        device = x_2d.device
        x_dtype_code = DTYPE_CODE[x_2d.dtype]
        rw_dtype_code = DTYPE_CODE[topk_weights.dtype]
        out_dtype_code = DTYPE_CODE[output.dtype]

        bundle = get_dxmoe_lib_bundle(device)
        lib, (serial_ctas, cons_ctas, prod_ctas) = bundle.select(num_tokens)
        ws = get_dxmoe_shared_workspace(
            num_tokens=num_tokens,
            max_num_tokens=self.moe_config.max_num_tokens,
            topk=state.top_k,
            num_experts=state.num_experts,
            hidden_size=state.hidden_size,
            inter_size=state.inter_size,
            in_dtype=x_2d.dtype,
            device=device,
            bundle=bundle,
        )
        stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)

        omega_ptr, scalar_omega, is_padding_ptr, num_valid_tokens = (
            self._resolve_b1_omega_and_padding(
                num_tokens=num_tokens,
                ws=ws,
                lib=lib,
                stream_ptr=stream_ptr,
            )
        )

        # Step 1: Entry Online Policy (Hadamard R_in + row amax + iso-error scoring)
        p_entry = NativePolicyDesc(
            x_in=int(x_2d.data_ptr()),
            signs=0 if state.s_in is None else int(state.s_in.data_ptr()),
            in_row_amax=0,
            omega=omega_ptr,
            token_indices=0,
            call_indices=0,
            routing_weights=int(topk_weights.data_ptr()),
            expert_ids=int(topk_ids.data_ptr()),
            expert_beta=int(state.beta_entry.data_ptr()),
            expert_w_bits=int(state.w13.w_bits_dev.data_ptr()),
            x_out=int(ws.x_tilde.data_ptr()),
            out_row_amax=int(ws.entry_amax.data_ptr()),
            out_expert_ids=0,
            out_activation_bits=int(ws.entry_bits.data_ptr()),
            out_scores=0,
            out_w8_hits=0,
            tau1=state.tau_entry[0],
            tau2=state.tau_entry[1],
            scalar_omega=scalar_omega,
            num_tokens=num_tokens,
            topk=state.top_k,
            num_experts=state.num_experts,
            hidden_dim=state.hidden_size,
            rows_are_calls=0,
            apply_swiglu=0,
            schedule_mode=self.schedule_code,
            apply_coupling=1 if state.apply_coupling_entry else 0,
            x_in_dtype=x_dtype_code,
            x_out_dtype=x_dtype_code,
            routing_dtype=rw_dtype_code,
        )
        err = int(lib.smoe_dxmoe_policy_run(ctypes.byref(p_entry), stream_ptr))
        if err != 0:
            raise RuntimeError(f"DxMoE Entry policy failed with CUDA status {err}")

        # Step 2: Entry Fused Reorder + Quantize + Grouped GEMM (gate_up_proj)
        r_entry = NativeReorderDesc(
            x=int(ws.x_tilde.data_ptr()),
            row_amax=int(ws.entry_amax.data_ptr()),
            expert_ids=int(topk_ids.data_ptr()),
            activation_bits=int(ws.entry_bits.data_ptr()),
            expert_w_bits=int(state.w13.w_bits_dev.data_ptr()),
            expert_w_ptrs=int(state.w13.w_ptrs_dev.data_ptr()),
            expert_sw_ptrs=int(state.w13.sw_ptrs_dev.data_ptr()),
            in_row_to_call=0,
            out_row_to_call=int(ws.r1_to_call.data_ptr()),
            out_call_to_row=0,
            pool_data=ws.pool_data_ptrs,
            pool_scale=ws.pool_scale_ptrs,
            pool_group_sum=ws.zero_ptrs_2,
            pool_offsets=ws.pool_offsets_ptrs,
            pool_call_ids=ws.zero_ptrs_3,
            pool_outputs=ws.entry_outputs_ptrs,
            workspace=int(ws.reorder_ws.data_ptr()),
            tasks=int(ws.tasks.data_ptr()),
            counts=int(ws.counts.data_ptr()),
            status=int(ws.status.data_ptr()),
            num_tokens=num_tokens,
            topk=state.top_k,
            num_experts=state.num_experts,
            k=state.hidden_size,
            n=2 * state.inter_size,
            rows_are_calls=0,
            x_dtype=x_dtype_code,
            a16_dtype=0,
            group_size=-1,
            compact_outputs=1,
        )
        err = int(
            lib.smoe_dxmoe_fused_run(
                ctypes.byref(r_entry),
                serial_ctas,
                cons_ctas,
                prod_ctas,
                state.pipelined_code,
                stream_ptr,
            )
        )
        if err != 0:
            raise RuntimeError(f"DxMoE Entry fused GEMM failed with CUDA status {err}")

        # Step 3: In-place Down Online Policy (SwiGLU + Hadamard R_mid + re-scoring)
        p_down = NativePolicyDesc(
            x_in=int(ws.gate_up_compact.data_ptr()),
            signs=0 if state.s_mid is None else int(state.s_mid.data_ptr()),
            in_row_amax=0,
            omega=omega_ptr,
            token_indices=0,
            call_indices=int(ws.r1_to_call.data_ptr()),
            routing_weights=int(topk_weights.data_ptr()),
            expert_ids=int(topk_ids.data_ptr()),
            expert_beta=int(state.beta_down.data_ptr()),
            expert_w_bits=int(state.w2.w_bits_dev.data_ptr()),
            x_out=int(ws.h_tilde.data_ptr()),
            out_row_amax=int(ws.down_amax.data_ptr()),
            out_expert_ids=int(ws.down_ids.data_ptr()),
            out_activation_bits=int(ws.down_bits.data_ptr()),
            out_scores=0,
            out_w8_hits=0,
            tau1=state.tau_down[0],
            tau2=state.tau_down[1],
            scalar_omega=scalar_omega,
            num_tokens=num_tokens,
            topk=state.top_k,
            num_experts=state.num_experts,
            hidden_dim=state.inter_size,
            rows_are_calls=1,
            apply_swiglu=1,
            schedule_mode=self.schedule_code,
            apply_coupling=1 if state.apply_coupling_down else 0,
            x_in_dtype=0,  # gate_up_compact is always float16
            x_out_dtype=x_dtype_code,
            routing_dtype=rw_dtype_code,
        )
        err = int(lib.smoe_dxmoe_policy_run(ctypes.byref(p_down), stream_ptr))
        if err != 0:
            raise RuntimeError(f"DxMoE Down policy failed with CUDA status {err}")

        # Step 4: Down Fused Reorder + Quantize + Grouped GEMM (down_proj)
        # Note: rows_are_calls == 1 uses serial 1x-read scatter path (pipelined=0 if forced).
        down_pipelined = 1 if state.pipelined_code == 2 else state.pipelined_code
        r_down = NativeReorderDesc(
            x=int(ws.h_tilde.data_ptr()),
            row_amax=int(ws.down_amax.data_ptr()),
            expert_ids=int(ws.down_ids.data_ptr()),
            activation_bits=int(ws.down_bits.data_ptr()),
            expert_w_bits=int(state.w2.w_bits_dev.data_ptr()),
            expert_w_ptrs=int(state.w2.w_ptrs_dev.data_ptr()),
            expert_sw_ptrs=int(state.w2.sw_ptrs_dev.data_ptr()),
            in_row_to_call=int(ws.r1_to_call.data_ptr()),
            out_row_to_call=0,
            out_call_to_row=int(ws.call_to_r2.data_ptr()),
            pool_data=ws.pool_data_ptrs,
            pool_scale=ws.pool_scale_ptrs,
            pool_group_sum=ws.zero_ptrs_2,
            pool_offsets=ws.pool_offsets_ptrs,
            pool_call_ids=ws.zero_ptrs_3,
            pool_outputs=ws.down_outputs_ptrs,
            workspace=int(ws.reorder_ws.data_ptr()),
            tasks=int(ws.tasks.data_ptr()),
            counts=int(ws.counts.data_ptr()),
            status=int(ws.status.data_ptr()),
            num_tokens=num_tokens,
            topk=state.top_k,
            num_experts=state.num_experts,
            k=state.inter_size,
            n=state.hidden_size,
            rows_are_calls=1,
            x_dtype=x_dtype_code,
            a16_dtype=0,
            group_size=-1,
            compact_outputs=1,
        )
        err = int(
            lib.smoe_dxmoe_fused_run(
                ctypes.byref(r_down),
                serial_ctas,
                cons_ctas,
                prod_ctas,
                down_pipelined,
                stream_ptr,
            )
        )
        if err != 0:
            raise RuntimeError(f"DxMoE Down fused GEMM failed with CUDA status {err}")

        # Step 5: Weighted Unpermute (FP16 compact -> FP32 accum -> FP16/BF16 output)
        u_desc = NativeUnpermuteDesc(
            permuted_in=int(ws.down_compact.data_ptr()),
            call_to_row=int(ws.call_to_r2.data_ptr()),
            routing_weights=int(topk_weights.data_ptr()),
            is_padding=is_padding_ptr,
            out=int(output.view(-1, state.hidden_size).data_ptr()),
            num_tokens=num_tokens,
            num_valid_tokens=num_valid_tokens,
            topk=state.top_k,
            hidden_dim=state.hidden_size,
            routing_dtype=rw_dtype_code,
            out_dtype=out_dtype_code,
        )
        err = int(lib.smoe_dxmoe_weighted_unpermute(ctypes.byref(u_desc), stream_ptr))
        if err != 0:
            raise RuntimeError(
                f"DxMoE weighted_unpermute failed with CUDA status {err}"
            )
