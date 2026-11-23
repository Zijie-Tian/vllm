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


def _tensor_byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    elem = tensor.element_size()
    base = int(tensor.data_ptr())
    if tensor.is_contiguous():
        return base, base + int(tensor.numel()) * elem
    min_off = 0
    max_off = 0
    for sz, st in zip(tensor.shape, tensor.stride()):
        if sz > 1:
            span = (sz - 1) * st
            if span < 0:
                min_off += span
            else:
                max_off += span
    return base + min_off * elem, base + (max_off + 1) * elem


def _tensors_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.numel() == 0 or b.numel() == 0 or a.device != b.device:
        return False
    a_lo, a_hi = _tensor_byte_span(a)
    b_lo, b_hi = _tensor_byte_span(b)
    return a_lo < b_hi and b_lo < a_hi


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
        return cap is not None and (int(cap[0]), int(cap[1])) in ((8, 0), (8, 6))

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
                pad_t.device == ws.device
                and pad_t.is_contiguous()
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

        pos_ptr = 0
        pos_dtype = 0
        seq_ends_ptr = 0
        pos_t = fc.positions if fc is not None else None
        if pos_t is not None and pos_t.ndim == 2 and pos_t.shape[0] >= 1:
            pos_t = pos_t[0]
        seq_t = fc.seq_lens if fc is not None else None

        has_dynamic_pos_and_seq = (
            pos_t is not None
            and seq_t is not None
            and pos_t.device == ws.device
            and seq_t.device == ws.device
            and pos_t.is_contiguous()
            and seq_t.is_contiguous()
            and pos_t.dtype in (torch.int32, torch.int64)
            and seq_t.dtype == torch.int32
            and pos_t.numel() >= num_tokens
            and seq_t.numel() >= 1
        )

        past_seq_len = 0
        uniform_seq_end = num_tokens
        omega_padding_ptr = is_padding_ptr

        if has_dynamic_pos_and_seq:
            assert pos_t is not None and seq_t is not None
            pos_dtype = 1 if pos_t.dtype == torch.int64 else 0
            pos_align = 8 if pos_dtype == 1 else 4
            pos_ptr = int(pos_t.data_ptr())
            seq_ends_ptr = int(seq_t.data_ptr())
            if pos_ptr % pos_align != 0 or seq_ends_ptr % 4 != 0:
                raise ValueError("ForwardContext positions/seq_lens pointers are misaligned")
        elif in_cudagraph:
            raise NotImplementedError(
                "DxMoE CUDA Graph capture/replay with schedule_mode='dxmoe', "
                "use_omega=True, and num_tokens > 1 requires persistent GPU "
                "positions and seq_lens buffers in ForwardContext."
            )
        elif attn_meta is not None:
            max_seq_len = getattr(attn_meta, "max_seq_len", None)
            actual = getattr(attn_meta, "num_actual_tokens", None)
            if max_seq_len is not None and actual is not None and int(actual) > 0:
                seq_end_val = int(max_seq_len)
                act_val = int(actual)
                past_seq_len = max(0, seq_end_val - act_val)
                uniform_seq_end = seq_end_val
                omega_padding_ptr = 0

        # Bind cache key to `ws.omega_buf.data_ptr()` and all dynamic input identities
        # so heterogeneous MoE layers never read an unpopulated `omega_buf`.
        omega_out_ptr = int(ws.omega_buf.data_ptr())
        cache_key = (
            "_dxmoe_omega_ready",
            omega_out_ptr,
            num_tokens,
            pos_ptr,
            pos_dtype,
            seq_ends_ptr,
            omega_padding_ptr,
            past_seq_len,
            uniform_seq_end,
            state.sink_tokens,
            state.local_window,
        )
        if fc is not None and fc.additional_kwargs.get(cache_key) is True:
            return omega_out_ptr, 1.0, is_padding_ptr, num_valid_tokens

        omega_desc = NativeOmegaDesc(
            position_ids=pos_ptr,
            seq_ends=seq_ends_ptr,
            is_padding=omega_padding_ptr,
            out_omega=omega_out_ptr,
            past_seq_len=past_seq_len,
            uniform_seq_end=uniform_seq_end,
            batch_size=1,
            seq_len=num_tokens,
            pos_batch_stride=0,
            sink_tokens=state.sink_tokens,
            local_window=state.local_window,
            mask_past_seq_end=1,
            pos_dtype=pos_dtype,
        )
        err = int(lib.smoe_dxmoe_compute_omega(ctypes.byref(omega_desc), stream_ptr))
        if err != 0:
            raise RuntimeError(f"smoe_dxmoe_compute_omega failed with CUDA status {err}")

        if fc is not None:
            fc.additional_kwargs[cache_key] = True
        return omega_out_ptr, 1.0, is_padding_ptr, num_valid_tokens

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
        if activation != MoEActivation.SILU:
            raise ValueError(f"DxMoEExperts requires SILU activation, got {activation!r}")

        state = self.dxmoe_state
        if global_num_experts != state.num_experts:
            raise ValueError(
                f"global_num_experts={global_num_experts} != state.num_experts={state.num_experts}"
            )
        w_device = state.w13.codes_arena.device
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.device != w_device
            or hidden_states.dtype not in (torch.float16, torch.bfloat16)
            or hidden_states.requires_grad
            or hidden_states.ndim < 2
            or hidden_states.shape[-1] != state.hidden_size
            or not hidden_states.is_contiguous()
        ):
            raise ValueError(
                f"hidden_states must be contiguous float16/bfloat16 [..., {state.hidden_size}] "
                f"inference tensor on {w_device}"
            )

        x_2d = hidden_states.view(-1, state.hidden_size)
        num_tokens = int(x_2d.shape[0])

        if (
            not isinstance(output, torch.Tensor)
            or output.device != w_device
            or output.dtype not in DTYPE_CODE
            or output.requires_grad
            or output.ndim < 2
            or output.shape[-1] != state.hidden_size
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"output must be contiguous float16/bfloat16/float32 [..., {state.hidden_size}] "
                f"inference tensor on {w_device}"
            )
        out_2d = output.view(-1, state.hidden_size)
        if out_2d.shape != (num_tokens, state.hidden_size):
            raise ValueError(
                f"output shape {tuple(out_2d.shape)} != expected {(num_tokens, state.hidden_size)}"
            )
        if _tensors_overlap(x_2d, out_2d):
            raise ValueError("DxMoEExperts requires non-overlapping hidden_states and output storage")

        if (
            not isinstance(topk_ids, torch.Tensor)
            or topk_ids.device != w_device
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_ids.requires_grad
            or topk_ids.ndim != 2
            or tuple(topk_ids.shape) != (num_tokens, state.top_k)
            or not topk_ids.is_contiguous()
        ):
            raise ValueError(
                f"topk_ids must be contiguous 2D int32/int64 {(num_tokens, state.top_k)} on {w_device}, "
                f"got shape={getattr(topk_ids, 'shape', None)} dtype={getattr(topk_ids, 'dtype', None)}"
            )
        if (
            not isinstance(topk_weights, torch.Tensor)
            or topk_weights.device != w_device
            or topk_weights.dtype not in DTYPE_CODE
            or topk_weights.requires_grad
            or topk_weights.ndim != 2
            or tuple(topk_weights.shape) != (num_tokens, state.top_k)
            or not topk_weights.is_contiguous()
        ):
            raise ValueError(
                f"topk_weights must be contiguous 2D float16/bfloat16/float32 "
                f"{(num_tokens, state.top_k)} on {w_device}, "
                f"got shape={getattr(topk_weights, 'shape', None)} dtype={getattr(topk_weights, 'dtype', None)}"
            )

        if num_tokens == 0:
            return

        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(dtype=torch.int32)

        rw_align = 4 if topk_weights.dtype == torch.float32 else 2
        out_align = 16 if (state.hidden_size & 7) == 0 else (4 if out_2d.dtype == torch.float32 else 2)
        if (
            int(x_2d.data_ptr()) % 16 != 0
            or int(out_2d.data_ptr()) % out_align != 0
            or int(topk_ids.data_ptr()) % 4 != 0
            or int(topk_weights.data_ptr()) % rw_align != 0
        ):
            raise ValueError("DxMoEExperts tensor base pointers violate native alignment requirements")

        device = x_2d.device
        x_dtype_code = DTYPE_CODE[x_2d.dtype]
        rw_dtype_code = DTYPE_CODE[topk_weights.dtype]
        out_dtype_code = DTYPE_CODE[out_2d.dtype]

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
            is_padding=is_padding_ptr,
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
            num_valid_tokens=num_valid_tokens,
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
            is_padding=is_padding_ptr,
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
            num_valid_tokens=num_valid_tokens,
            preserve_prior_status=0,
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
            is_padding=is_padding_ptr,
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
            num_valid_tokens=num_valid_tokens,
        )
        err = int(lib.smoe_dxmoe_policy_run(ctypes.byref(p_down), stream_ptr))
        if err != 0:
            raise RuntimeError(f"DxMoE Down policy failed with CUDA status {err}")

        # Step 4: Down Fused Reorder + Quantize + Grouped GEMM (down_proj)
        # Note: rows_are_calls == 1 uses serial 1x-read scatter path, and
        # preserve_prior_status=1 prevents overwriting an Entry status error with 0!
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
            is_padding=0,
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
            num_valid_tokens=num_valid_tokens,
            preserve_prior_status=1,
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
            out=int(out_2d.data_ptr()),
            status=int(ws.status.data_ptr()),
            sticky_status=int(ws.sticky_status.data_ptr()),
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
