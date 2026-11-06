# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from types import MethodType
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.experts import (
    scalingmoe as scalingmoe_module,
)
from vllm.model_executor.layers.fused_moe.experts.scalingmoe import (
    ScalingMoEExperts,
    scalingmoe_grouped_gemm_zero,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    _get_priority_backends,
    backend_to_kernel_cls,
    map_unquantized_backend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    grouped_topk,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
)


def test_scalingmoe_is_explicit_and_not_an_auto_candidate() -> None:
    assert map_unquantized_backend("scalingmoe") is UnquantizedMoeBackend.SCALINGMOE
    kernel_classes = backend_to_kernel_cls(UnquantizedMoeBackend.SCALINGMOE)
    assert kernel_classes == [ScalingMoEExperts]

    moe_config = SimpleNamespace(
        activation=MoEActivation.SILU,
        moe_parallel_config=SimpleNamespace(dp_size=1),
    )
    with (
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(
            current_platform,
            "is_device_capability_family",
            return_value=False,
        ),
    ):
        assert UnquantizedMoeBackend.SCALINGMOE not in _get_priority_backends(
            moe_config
        )


def test_scalingmoe_grouped_gemm_stub_zeros_the_output() -> None:
    assert hasattr(torch.ops.vllm, "scalingmoe_grouped_gemm_zero")

    output = torch.full((3, 5), 7.0)
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(2, 5, 5)
    expert_offsets = torch.tensor([0, 2, 3], dtype=torch.int64)
    permuted_idx = torch.tensor([0, 2, 1], dtype=torch.int32)
    topk_ids = torch.tensor([[0], [1], [0]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.2], [0.3], [0.5]])
    router_logits = torch.tensor([[-2.0, 3.0], [4.0, -5.0], [1.5, -0.5]])
    original_logits = router_logits.clone()

    result = scalingmoe_grouped_gemm_zero(
        output,
        hidden_states,
        weight,
        expert_offsets,
        permuted_idx,
        topk_ids,
        topk_weights,
        router_logits,
    )

    assert result is None
    assert torch.count_nonzero(output) == 0
    assert torch.equal(router_logits, original_logits)


def test_scalingmoe_apply_runs_the_full_grouped_pipeline() -> None:
    experts = ScalingMoEExperts.__new__(ScalingMoEExperts)
    experts.moe_config = SimpleNamespace(
        max_num_tokens=8,
        dp_size=1,
        experts_per_token=2,
    )

    hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    w1 = torch.empty(2, 4, 3)
    w2 = torch.empty(2, 3, 2)
    topk_ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    router_logits = torch.tensor([[-3.0, 2.5], [4.0, -1.5]])
    workspace13 = torch.empty(2, 8)
    output = workspace13.view(-1)[:6].view(2, 3)
    workspace2 = torch.empty(2, 10)
    assert output.data_ptr() == workspace13.data_ptr()

    permuted_hidden = torch.full((4, 3), 11.0)
    expert_offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    inverse_permutation = torch.tensor([1, 2, 0, 3], dtype=torch.int32)
    permuted_idx = torch.tensor([2, 0, 1, 3], dtype=torch.int32)
    scratch = object()
    gemm_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def fake_grouped_gemm(
        result: torch.Tensor,
        activations: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        row_mapping: torch.Tensor,
        ids: torch.Tensor,
        alpha: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        assert offsets is expert_offsets
        assert row_mapping is permuted_idx
        assert ids is topk_ids
        assert alpha is topk_weights
        assert logits is router_logits
        assert logits.tolist() == [[-3.0, 2.5], [4.0, -1.5]]
        gemm_calls.append((result, activations, weight))
        if len(gemm_calls) == 1:
            assert activations is permuted_hidden
            assert weight is w1
            assert result.shape == (4, 4)
            result.fill_(13)
        else:
            assert weight is w2
            assert result.shape == (4, 3)
            assert torch.count_nonzero(activations - 17) == 0
            result.fill_(19)

    def fake_activation(
        activation: MoEActivation,
        result: torch.Tensor,
        gate_up: torch.Tensor,
        **kwargs,
    ) -> None:
        assert activation is MoEActivation.SILU
        assert gate_up.shape == (4, 4)
        assert torch.count_nonzero(gate_up - 13) == 0
        assert result.shape == (4, 2)
        result.fill_(17)

    def fake_unpermute(**kwargs) -> None:
        assert kwargs["out"] is output
        assert kwargs["topk_weights"] is topk_weights
        assert kwargs["inv_permuted_idx"] is inverse_permutation
        assert kwargs["expert_first_token_offset"] is expert_offsets
        down = kwargs["permuted_hidden_states"]
        assert down.shape == (2, 2, 3)
        assert torch.count_nonzero(down - 19) == 0
        output.fill_(23)

    with (
        patch.object(
            scalingmoe_module,
            "get_moe_permute_scratch",
            return_value=scratch,
        ) as get_scratch,
        patch.object(
            scalingmoe_module,
            "moe_permute",
            return_value=(
                permuted_hidden,
                None,
                expert_offsets,
                inverse_permutation,
                permuted_idx,
            ),
        ) as permute,
        patch.object(scalingmoe_module, "moe_unpermute", side_effect=fake_unpermute),
        patch.object(
            torch.ops.vllm,
            "scalingmoe_grouped_gemm_zero",
            side_effect=fake_grouped_gemm,
        ),
        patch.object(experts, "activation", side_effect=fake_activation),
    ):
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=2,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
            router_logits=router_logits,
        )

    get_scratch.assert_called_once_with(
        max_num_tokens=8,
        topk=2,
        num_experts=2,
        num_local_experts=2,
        device=hidden_states.device,
        hidden_size=3,
        hidden_dtype=hidden_states.dtype,
    )
    assert permute.call_args.kwargs["scratch"] is scratch
    assert len(gemm_calls) == 2
    assert torch.count_nonzero(output - 23) == 0


@pytest.mark.parametrize(
    ("router_logits", "message"),
    [
        (None, "requires original .* router logits"),
        (torch.empty(1, 3), "router logits must have shape"),
    ],
)
def test_scalingmoe_rejects_missing_or_wrong_router_logits(
    router_logits, message
) -> None:
    experts = ScalingMoEExperts.__new__(ScalingMoEExperts)
    experts.moe_config = SimpleNamespace(
        max_num_tokens=1,
        dp_size=1,
        experts_per_token=1,
    )
    with pytest.raises(ValueError, match=message):
        experts.apply(
            output=torch.empty(1, 2),
            hidden_states=torch.empty(1, 2),
            w1=torch.empty(2, 4, 2),
            w2=torch.empty(2, 2, 2),
            topk_weights=torch.ones(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
            activation=MoEActivation.SILU,
            global_num_experts=2,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=torch.empty(1, 4),
            workspace2=torch.empty(1, 4),
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
            router_logits=router_logits,
        )


@pytest.mark.parametrize("requires_router_logits", [False, True])
def test_runner_only_plumbs_router_logits_to_opted_in_experts(
    requires_router_logits,
) -> None:
    hidden_states = torch.randn(2, 3)
    initial_logits = torch.full((2, 2), 99.0)
    router_logits = torch.tensor([[-3.0, 2.0], [4.5, -1.0]])
    topk_weights = torch.tensor([[0.25], [0.75]])
    topk_ids = torch.tensor([[1], [0]], dtype=torch.int32)
    fused_output = torch.randn_like(hidden_states)

    kernel = SimpleNamespace(
        fused_experts=SimpleNamespace(
            requires_router_logits=requires_router_logits,
        ),
        prepare_finalize=SimpleNamespace(topk_indices_dtype=lambda: None),
        is_monolithic=False,
        apply=Mock(return_value=fused_output),
    )
    quant_method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
    quant_method.moe_kernel = kernel
    quant_method.forward = quant_method.forward_native

    routed_experts = SimpleNamespace(
        quant_method=quant_method,
        w13_weight=torch.empty(2, 4, 3),
        w2_weight=torch.empty(2, 3, 2),
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        global_num_experts=2,
        expert_map=None,
        _ensure_moe_quant_config_init=Mock(),
    )
    routed_experts.forward_modular = MethodType(
        RoutedExperts.forward_modular,
        routed_experts,
    )
    router = SimpleNamespace(
        select_experts=Mock(return_value=(topk_weights, topk_ids))
    )
    gate = Mock(return_value=(router_logits, None))

    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = routed_experts
    runner.router = router
    runner.gate = gate
    runner._fse_fuse_gate = False
    runner._shared_experts = None
    with (
        patch.object(runner, "_sequence_parallel_context", return_value=nullcontext()),
        patch.object(
            runner,
            "_maybe_dispatch",
            side_effect=lambda states, logits: (states, logits),
        ),
        patch.object(runner, "_maybe_combine", side_effect=lambda _, states: states),
    ):
        result = runner._forward_impl(
            hidden_states,
            initial_logits,
            shared_experts_input=None,
        )

    assert result is fused_output
    gate.assert_called_once_with(hidden_states)
    assert router.select_experts.call_args.kwargs["router_logits"] is router_logits
    kernel_kwargs = kernel.apply.call_args.kwargs
    if requires_router_logits:
        assert kernel_kwargs["router_logits"] is router_logits
    else:
        assert "router_logits" not in kernel_kwargs


def test_grouped_topk_preserves_original_router_scores() -> None:
    hidden_states = torch.zeros(2, 3)
    router_logits = torch.tensor([[-2.0, 3.0, 5.0, -1.0], [4.0, 2.0, -3.0, -4.0]])
    original_logits = router_logits.clone()

    weights, ids = grouped_topk(
        hidden_states,
        router_logits,
        topk=1,
        renormalize=False,
        num_expert_group=2,
        topk_group=1,
    )

    expected_ids = torch.tensor([[2], [0]], dtype=torch.int32)
    expected_weights = torch.softmax(original_logits, dim=-1).gather(
        1, expected_ids.long()
    )
    assert torch.equal(router_logits, original_logits)
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA device")
def test_scalingmoe_real_pipeline_returns_zero(dtype, use_graph) -> None:
    assert torch.cuda.device_count() == 1
    device = torch.device("cuda:0")
    tokens, topk, experts_count = 2, 2, 3
    hidden_size, gate_up_size = 8, 8
    intermediate_size = gate_up_size // 2

    experts = ScalingMoEExperts.__new__(ScalingMoEExperts)
    experts.activation_config = ApplyMoEActivationConfig()
    experts.moe_config = SimpleNamespace(
        max_num_tokens=tokens,
        dp_size=1,
        experts_per_token=topk,
    )
    hidden_states = torch.randn(tokens, hidden_size, dtype=dtype, device=device)
    w1 = torch.randn(
        experts_count,
        gate_up_size,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    w2 = torch.randn(
        experts_count,
        hidden_size,
        intermediate_size,
        dtype=dtype,
        device=device,
    )
    topk_ids = torch.tensor([[0, 2], [2, 0]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32, device=device
    )
    router_logits = torch.tensor(
        [[-2.0, 1.5, 3.0], [4.0, -5.0, 0.25]],
        dtype=torch.float32,
        device=device,
    )
    workspace13 = torch.empty(
        tokens,
        topk * gate_up_size,
        dtype=dtype,
        device=device,
    )
    output = workspace13.view(-1)[: tokens * hidden_size].view(tokens, hidden_size)
    workspace2 = torch.empty(
        tokens,
        topk * (intermediate_size + hidden_size),
        dtype=dtype,
        device=device,
    )
    assert output.data_ptr() == workspace13.data_ptr()

    def run() -> None:
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=experts_count,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
            router_logits=router_logits,
        )

    reset_workspace_manager()
    init_workspace_manager(device)
    try:
        run()
        torch.cuda.synchronize()
        assert torch.count_nonzero(output) == 0
        scratch = scalingmoe_module.get_moe_permute_scratch(
            max_num_tokens=tokens,
            topk=topk,
            num_experts=experts_count,
            num_local_experts=experts_count,
            device=device,
            hidden_size=hidden_size,
            hidden_dtype=dtype,
        )
        assert scratch.expert_first_token_offset.cpu().tolist() == [0, 2, 2, 4]

        if use_graph:
            lock_workspace()
            output.fill_(1)
            logits_ptr = router_logits.data_ptr()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            topk_ids.copy_(
                torch.tensor([[1, 1], [2, 1]], dtype=torch.int32, device=device)
            )
            topk_weights.copy_(
                torch.tensor(
                    [[0.9, 0.1], [0.25, 0.75]],
                    dtype=torch.float32,
                    device=device,
                )
            )
            router_logits.copy_(
                torch.tensor(
                    [[5.0, -4.0, 2.0], [-3.0, 1.25, 6.0]],
                    dtype=torch.float32,
                    device=device,
                )
            )
            assert router_logits.data_ptr() == logits_ptr
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(output) == 0
            assert scratch.expert_first_token_offset.cpu().tolist() == [0, 0, 3, 4]
    finally:
        reset_workspace_manager()

