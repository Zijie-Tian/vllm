# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ABI 11 & descriptor layout contracts for `vllm.utils.dxmoe`."""

from __future__ import annotations

import ctypes
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm.utils import dxmoe as vllm_dxmoe


def test_dxmoe_abi11_rejects_mismatched_dso_before_arch_or_prepare(
    tmp_path: Path,
) -> None:
    assert vllm_dxmoe.DXMOE_ABI_VERSION == 11
    dummy_so = tmp_path / "libsmoe_dxmoe_short.so"
    dummy_so.write_bytes(b"stub")

    for bad_abi in (0, 9, 10, 12):
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
                match=rf"DxMoE ABI mismatch.*expected 11, got {bad_abi}",
            ),
        ):
            vllm_dxmoe._bind_dxmoe_dso(dummy_so, expected_arch=80)
        assert calls == ["abi"]


def test_dxmoe_abi11_descriptor_sizes_and_key_offsets() -> None:
    assert ctypes.sizeof(vllm_dxmoe.NativeReorderDesc) == 280
    assert vllm_dxmoe.NativeReorderDesc.in_row_to_call.offset == 56
    assert vllm_dxmoe.NativeReorderDesc.out_row_to_call.offset == 64
    assert vllm_dxmoe.NativeReorderDesc.out_call_to_row.offset == 72
    assert vllm_dxmoe.NativeReorderDesc.pool_data.offset == 80
    assert vllm_dxmoe.NativeReorderDesc.workspace.offset == 208
    assert vllm_dxmoe.NativeReorderDesc.num_tokens.offset == 240
    assert vllm_dxmoe.NativeReorderDesc.compact_outputs.offset == 276

    assert ctypes.sizeof(vllm_dxmoe.NativeOmegaDesc) == 72
    assert vllm_dxmoe.NativeOmegaDesc.batch_size.offset == 48
    assert vllm_dxmoe.NativeOmegaDesc.mask_past_seq_end.offset == 68

    assert ctypes.sizeof(vllm_dxmoe.NativePolicyDesc) == 192
    assert vllm_dxmoe.NativePolicyDesc.call_indices.offset == 40
    assert vllm_dxmoe.NativePolicyDesc.out_expert_ids.offset == 96
    assert vllm_dxmoe.NativePolicyDesc.tau1.offset == 128
    assert vllm_dxmoe.NativePolicyDesc.scalar_omega.offset == 144
    assert vllm_dxmoe.NativePolicyDesc.routing_dtype.offset == 188

    assert ctypes.sizeof(vllm_dxmoe.NativeUnpermuteDesc) == 64
    assert vllm_dxmoe.NativeUnpermuteDesc.num_tokens.offset == 40
    assert vllm_dxmoe.NativeUnpermuteDesc.out_dtype.offset == 60
