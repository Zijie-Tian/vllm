# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    get_moe_permute_scratch,
    moe_permute,
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def scalingmoe_grouped_gemm_zero(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_offsets: torch.Tensor,
    permuted_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
) -> None:
    """Zero one grouped GEMM result, preserving its routing-aware interface.

    ``topk_ids`` and ``topk_weights`` retain their original [tokens, topk]
    layout, while ``router_logits`` is the original pre-TopK [tokens, experts]
    gate output. For a valid grouped row r, ``permuted_idx[r]`` identifies its
    flattened token/slot entry, and ``permuted_idx[r] // topk`` selects the
    corresponding router-logit row. ``expert_offsets`` delimits the expert
    groups. Inputs, weights, and routing metadata are read-only; only output is
    written. Router weighting belongs to moe_unpermute, not either GEMM
    placeholder.
    """
    output.zero_()


def scalingmoe_grouped_gemm_zero_fake(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_offsets: torch.Tensor,
    permuted_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="scalingmoe_grouped_gemm_zero",
    op_func=scalingmoe_grouped_gemm_zero,
    mutates_args=["output"],
    fake_impl=scalingmoe_grouped_gemm_zero_fake,
)


class ScalingMoEExperts(mk.FusedMoEExpertsModular):
    """Grouped MoE pipeline with only the two GEMM computations stubbed out."""

    requires_router_logits = True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and moe_permute_unpermute_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            moe_parallel_config.tp_size == 1
            and moe_parallel_config.pcp_size == 1
            and moe_parallel_config.dp_size == 1
            and moe_parallel_config.ep_size == 1
            and moe_parallel_config.sp_size == 1
            and not moe_parallel_config.use_ep
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True

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
        intermediate = self.adjust_N_for_activation(N, activation)
        # The framework may alias final output with workspace13. Keep GEMM2
        # output in workspace2, disjoint from both that output and its input.
        return (
            (M, topk * N),
            (M, topk * (intermediate + K)),
            (M, K),
        )

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
        router_logits: torch.Tensor | None = None,
    ) -> None:
        if apply_router_weight_on_input:
            raise ValueError("scalingmoe applies router weights on output only")
        if expert_map is not None or a1q_scale is not None or a2_scale is not None:
            raise ValueError("scalingmoe requires unquantized inputs without EP")
        if not self._supports_activation(activation):
            raise ValueError("scalingmoe currently supports only SiLU-and-mul")

        E, M, N, K, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if global_num_experts != E:
            raise ValueError("scalingmoe requires all experts on one GPU")
        if router_logits is None:
            raise ValueError("scalingmoe requires original pre-TopK router logits")
        if tuple(router_logits.shape) != (M, global_num_experts):
            raise ValueError(
                "scalingmoe router logits must have shape [tokens, experts]"
            )
        if not router_logits.is_floating_point():
            raise ValueError("scalingmoe router logits must be floating point")
        if router_logits.device != hidden_states.device:
            raise ValueError("scalingmoe router logits must be on the input device")
        if M == 0:
            return

        logger.info_once(
            "Executing ScalingMoE grouped MoE: permute, zero GEMM1, SiLU, "
            "zero GEMM2, and weighted unpermute."
        )

        max_expanded_rows = (
            self.moe_config.max_num_tokens
            * self.moe_config.dp_size
            * self.moe_config.experts_per_token
        )
        scratch = get_moe_permute_scratch(
            max_num_tokens=(max_expanded_rows + topk - 1) // topk,
            topk=topk,
            num_experts=global_num_experts,
            num_local_experts=E,
            device=hidden_states.device,
            hidden_size=K,
            hidden_dtype=hidden_states.dtype,
        )
        permuted, _, expert_offsets, inv_perm, permuted_idx = moe_permute(
            hidden_states=hidden_states,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=global_num_experts,
            n_local_expert=E,
            expert_map=None,
            scratch=scratch,
        )

        rows = M * topk
        intermediate = self.adjust_N_for_activation(N, activation)
        gate_up = workspace13.view(-1)[: rows * N].view(rows, N)
        flat2 = workspace2.view(-1)
        middle = flat2[: rows * intermediate].view(rows, intermediate)
        down = flat2[rows * intermediate : rows * (intermediate + K)].view(rows, K)

        torch.ops.vllm.scalingmoe_grouped_gemm_zero(
            gate_up,
            permuted,
            w1,
            expert_offsets,
            permuted_idx,
            topk_ids,
            topk_weights,
            router_logits,
        )
        self.activation(activation, middle, gate_up)
        torch.ops.vllm.scalingmoe_grouped_gemm_zero(
            down,
            middle,
            w2,
            expert_offsets,
            permuted_idx,
            topk_ids,
            topk_weights,
            router_logits,
        )
        moe_unpermute(
            out=output,
            permuted_hidden_states=down.view(M, topk, K),
            topk_weights=topk_weights,
            inv_permuted_idx=inv_perm,
            expert_first_token_offset=expert_offsets,
        )
