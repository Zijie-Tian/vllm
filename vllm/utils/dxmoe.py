# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native C++/CUDA ctypes bridge, weight packer, and workspace manager for DxMoE."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import weakref

import torch

from vllm.utils.math_utils import round_up
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

DXMOE_ABI_VERSION = 12
DXMOE_SHORT_MAX_TOKENS = 1024
_TASK_COLUMNS = 11
_COUNTS_LEN = 8

DXMOE_STATUS_MESSAGES: dict[int, str] = {
    2: "expert_ids out of range or activation_bits not in {4, 8, 16}",
    3: "unsupported precision pair W8A4 (DxMoE coupling forbids W8 in the A4 pool)",
    4: "DxMoE tile count exceeds int32",
    5: "input activations must be finite",
    6: "FP16 A16 pool cast overflowed finite range",
}

DTYPE_CODE: dict[torch.dtype, int] = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}

SCHEDULE_MODE_CODE: dict[str, int] = {
    "dxmoe": 0,
    "static_a16": 1,
    "static_a8": 2,
    "static_a4": 3,
}

PIPELINED_CODE: dict[str | bool, int] = {
    False: 0,
    "false": 0,
    "0": 0,
    "serial": 0,
    "auto": 1,
    "1": 1,
    True: 2,
    "true": 2,
    "2": 2,
    "force": 2,
}


class NativeReorderDesc(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int64),
        ("row_amax", ctypes.c_int64),
        ("expert_ids", ctypes.c_int64),
        ("activation_bits", ctypes.c_int64),
        ("expert_w_bits", ctypes.c_int64),
        ("expert_w_ptrs", ctypes.c_int64),
        ("expert_sw_ptrs", ctypes.c_int64),
        ("in_row_to_call", ctypes.c_int64),
        ("out_row_to_call", ctypes.c_int64),
        ("out_call_to_row", ctypes.c_int64),
        ("pool_data", ctypes.c_int64 * 3),
        ("pool_scale", ctypes.c_int64 * 2),
        ("pool_group_sum", ctypes.c_int64 * 2),
        ("pool_offsets", ctypes.c_int64 * 3),
        ("pool_call_ids", ctypes.c_int64 * 3),
        ("pool_outputs", ctypes.c_int64 * 3),
        ("workspace", ctypes.c_int64),
        ("tasks", ctypes.c_int64),
        ("counts", ctypes.c_int64),
        ("status", ctypes.c_int64),
        ("is_padding", ctypes.c_int64),
        ("num_tokens", ctypes.c_int32),
        ("topk", ctypes.c_int32),
        ("num_experts", ctypes.c_int32),
        ("k", ctypes.c_int32),
        ("n", ctypes.c_int32),
        ("rows_are_calls", ctypes.c_int32),
        ("x_dtype", ctypes.c_int32),
        ("a16_dtype", ctypes.c_int32),
        ("group_size", ctypes.c_int32),
        ("compact_outputs", ctypes.c_int32),
        ("num_valid_tokens", ctypes.c_int32),
        ("preserve_prior_status", ctypes.c_int32),
    ]


class NativeOmegaDesc(ctypes.Structure):
    _fields_ = [
        ("position_ids", ctypes.c_int64),
        ("seq_ends", ctypes.c_int64),
        ("is_padding", ctypes.c_int64),
        ("out_omega", ctypes.c_int64),
        ("past_seq_len", ctypes.c_int64),
        ("uniform_seq_end", ctypes.c_int64),
        ("batch_size", ctypes.c_int32),
        ("seq_len", ctypes.c_int32),
        ("pos_batch_stride", ctypes.c_int32),
        ("sink_tokens", ctypes.c_int32),
        ("local_window", ctypes.c_int32),
        ("mask_past_seq_end", ctypes.c_int32),
        ("pos_dtype", ctypes.c_int32),
    ]


class NativePolicyDesc(ctypes.Structure):
    _fields_ = [
        ("x_in", ctypes.c_int64),
        ("signs", ctypes.c_int64),
        ("in_row_amax", ctypes.c_int64),
        ("omega", ctypes.c_int64),
        ("token_indices", ctypes.c_int64),
        ("call_indices", ctypes.c_int64),
        ("routing_weights", ctypes.c_int64),
        ("expert_ids", ctypes.c_int64),
        ("expert_beta", ctypes.c_int64),
        ("expert_w_bits", ctypes.c_int64),
        ("x_out", ctypes.c_int64),
        ("out_row_amax", ctypes.c_int64),
        ("out_expert_ids", ctypes.c_int64),
        ("out_activation_bits", ctypes.c_int64),
        ("out_scores", ctypes.c_int64),
        ("out_w8_hits", ctypes.c_int64),
        ("is_padding", ctypes.c_int64),
        ("tau1", ctypes.c_double),
        ("tau2", ctypes.c_double),
        ("scalar_omega", ctypes.c_float),
        ("num_tokens", ctypes.c_int32),
        ("topk", ctypes.c_int32),
        ("num_experts", ctypes.c_int32),
        ("hidden_dim", ctypes.c_int32),
        ("rows_are_calls", ctypes.c_int32),
        ("apply_swiglu", ctypes.c_int32),
        ("schedule_mode", ctypes.c_int32),
        ("apply_coupling", ctypes.c_int32),
        ("x_in_dtype", ctypes.c_int32),
        ("x_out_dtype", ctypes.c_int32),
        ("routing_dtype", ctypes.c_int32),
        ("num_valid_tokens", ctypes.c_int32),
    ]


class NativeUnpermuteDesc(ctypes.Structure):
    _fields_ = [
        ("permuted_in", ctypes.c_int64),
        ("call_to_row", ctypes.c_int64),
        ("routing_weights", ctypes.c_int64),
        ("is_padding", ctypes.c_int64),
        ("out", ctypes.c_int64),
        ("status", ctypes.c_int64),
        ("sticky_status", ctypes.c_int64),
        ("num_tokens", ctypes.c_int32),
        ("num_valid_tokens", ctypes.c_int32),
        ("topk", ctypes.c_int32),
        ("hidden_dim", ctypes.c_int32),
        ("routing_dtype", ctypes.c_int32),
        ("out_dtype", ctypes.c_int32),
    ]


def _bind_dxmoe_dso(path: Path, expected_arch: int) -> ctypes.CDLL:
    if not path.is_file():
        raise FileNotFoundError(
            f"DxMoE native library not found at {path}. "
            "Build libsmoe_dxmoe_{{short,long}}.so or set VLLM_DXMOE_SHORT_LIB / "
            "VLLM_DXMOE_LONG_LIB."
        )
    lib = ctypes.CDLL(str(path))
    lib.smoe_dxmoe_abi_version.argtypes = []
    lib.smoe_dxmoe_abi_version.restype = ctypes.c_int
    abi = int(lib.smoe_dxmoe_abi_version())
    if abi != DXMOE_ABI_VERSION:
        raise RuntimeError(
            f"DxMoE ABI mismatch in {path}: expected {DXMOE_ABI_VERSION}, got {abi}"
        )

    lib.smoe_dxmoe_arch.argtypes = []
    lib.smoe_dxmoe_arch.restype = ctypes.c_int
    arch = int(lib.smoe_dxmoe_arch())
    if arch != expected_arch:
        raise RuntimeError(
            f"DxMoE SM arch mismatch in {path}: library compiled for sm_{arch}, "
            f"current GPU is sm_{expected_arch}"
        )

    lib.smoe_dxmoe_prepare.argtypes = []
    lib.smoe_dxmoe_prepare.restype = ctypes.c_int
    lib.smoe_dxmoe_default_grid.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.smoe_dxmoe_default_grid.restype = ctypes.c_int
    lib.smoe_dxmoe_reorder_workspace_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.smoe_dxmoe_reorder_workspace_bytes.restype = ctypes.c_int64
    lib.smoe_dxmoe_fused_run.argtypes = [
        ctypes.POINTER(NativeReorderDesc),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.smoe_dxmoe_fused_run.restype = ctypes.c_int
    lib.smoe_dxmoe_compute_omega.argtypes = [
        ctypes.POINTER(NativeOmegaDesc),
        ctypes.c_void_p,
    ]
    lib.smoe_dxmoe_compute_omega.restype = ctypes.c_int
    lib.smoe_dxmoe_policy_run.argtypes = [
        ctypes.POINTER(NativePolicyDesc),
        ctypes.c_void_p,
    ]
    lib.smoe_dxmoe_policy_run.restype = ctypes.c_int
    lib.smoe_dxmoe_weighted_unpermute.argtypes = [
        ctypes.POINTER(NativeUnpermuteDesc),
        ctypes.c_void_p,
    ]
    lib.smoe_dxmoe_weighted_unpermute.restype = ctypes.c_int
    return lib


def _prepare_lib_on_current_device(lib: ctypes.CDLL) -> tuple[int, int, int]:
    err = int(lib.smoe_dxmoe_prepare())
    if err != 0:
        raise RuntimeError(f"smoe_dxmoe_prepare failed with CUDA status {err}")
    s_ctas = ctypes.c_int(0)
    c_ctas = ctypes.c_int(0)
    p_ctas = ctypes.c_int(0)
    err = int(
        lib.smoe_dxmoe_default_grid(
            ctypes.byref(s_ctas),
            ctypes.byref(c_ctas),
            ctypes.byref(p_ctas),
        )
    )
    if err != 0:
        raise RuntimeError(f"smoe_dxmoe_default_grid failed with CUDA status {err}")
    return int(s_ctas.value), int(c_ctas.value), int(p_ctas.value)


class DxMoELibBundle:
    """Dual short/long DSO bundle initialized for a specific CUDA device."""

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError(f"DxMoE requires a CUDA device, got {device}")
        dev_idx = 0 if device.index is None else int(device.index)
        cap = torch.cuda.get_device_capability(dev_idx)
        arch = cap[0] * 10 + cap[1]

        # Auto-discover from env vars or parent ScalingMoE build directory.
        repo_root = Path(__file__).resolve().parents[4]
        default_dir = repo_root / "build" / f"dxmoe_sm{arch}"
        short_raw = (
            os.environ.get("VLLM_DXMOE_SHORT_LIB")
            or os.environ.get("DXMOE_SHORT_LIB")
            or str(default_dir / "libsmoe_dxmoe_short.so")
        )
        long_raw = (
            os.environ.get("VLLM_DXMOE_LONG_LIB")
            or os.environ.get("DXMOE_LONG_LIB")
            or str(default_dir / "libsmoe_dxmoe_long.so")
        )

        self.short_path = Path(short_raw)
        self.long_path = Path(long_raw)
        self.short_lib = _bind_dxmoe_dso(self.short_path, arch)
        self.long_lib = (
            self.short_lib
            if self.long_path.resolve() == self.short_path.resolve()
            else _bind_dxmoe_dso(self.long_path, arch)
        )
        with torch.cuda.device(dev_idx):
            self.short_grid = _prepare_lib_on_current_device(self.short_lib)
            self.long_grid = (
                self.short_grid
                if self.long_lib is self.short_lib
                else _prepare_lib_on_current_device(self.long_lib)
            )

    def select(self, num_tokens: int) -> tuple[ctypes.CDLL, tuple[int, int, int]]:
        if num_tokens <= DXMOE_SHORT_MAX_TOKENS:
            return self.short_lib, self.short_grid
        return self.long_lib, self.long_grid

    def reorder_workspace_bytes(
        self, num_tokens: int, topk: int, num_experts: int
    ) -> int:
        b_short = int(
            self.short_lib.smoe_dxmoe_reorder_workspace_bytes(
                int(num_tokens), int(topk), int(num_experts)
            )
        )
        b_long = int(
            self.long_lib.smoe_dxmoe_reorder_workspace_bytes(
                int(num_tokens), int(topk), int(num_experts)
            )
        )
        return max(b_short, b_long, 256)


_BUNDLE_LOCK = threading.Lock()
_BUNDLES_BY_DEVICE: dict[int, DxMoELibBundle] = {}


def get_dxmoe_lib_bundle(device: torch.device) -> DxMoELibBundle:
    dev_idx = 0 if device.index is None else int(device.index)
    bundle = _BUNDLES_BY_DEVICE.get(dev_idx)
    if bundle is not None:
        return bundle
    with _BUNDLE_LOCK:
        bundle = _BUNDLES_BY_DEVICE.get(dev_idx)
        if bundle is None:
            bundle = DxMoELibBundle(device)
            _BUNDLES_BY_DEVICE[dev_idx] = bundle
        return bundle


@dataclass(frozen=True)
class DxMoELayerWeights:
    """Packed symmetric G=-1 W2/W3/W4/W8 expert payload for one MoE projection stage."""

    num_experts: int
    n: int
    k: int
    codes_arena: torch.Tensor  # uint8 [total_packed_bytes] (256B-aligned per expert)
    scales: torch.Tensor  # float16 [E, N]
    w_bits_dev: torch.Tensor  # int32 [E]
    w_ptrs_dev: torch.Tensor  # int64 [E]
    sw_ptrs_dev: torch.Tensor  # int64 [E]
    has_w8: bool


SUPPORTED_W_BITS: tuple[int, ...] = (2, 3, 4, 8)

GPTQ_SYM_QZEROS: dict[int, int] = {
    2: 0x55555555,
    4: 0x77777777,
    8: 0x7F7F7F7F,
}
# Signed int32 representation of (0xDB6DB6DB, 0xB6DB6DB6, 0x6DB6DB6D) for symmetric W3 (zero - 1 = 3).
GPTQ_SYM_QZEROS_W3: tuple[int, int, int] = (-0x24924925, -0x4924924A, 0x6DB6DB6D)

_W2_REFIT_DENOMS: tuple[float, ...] = (2.0, 1.5, 1.0)
_W3_REFIT_DENOMS: tuple[float, ...] = (4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0)
_W4_REFIT_DENOMS: tuple[float, ...] = (8.0, 7.5, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
_W8_REFIT_DENOMS: tuple[float, ...] = tuple(float(m) for m in range(128, 95, -1)) + (
    127.5,
    80.0,
    64.0,
    48.0,
    32.0,
    16.0,
    8.0,
    4.0,
    2.0,
    1.0,
)


def dxmoe_packed_row_bytes(k: int, w_bits: int) -> int:
    """Return exact packed byte width `(K * w_bits) // 8` for one output channel."""
    if w_bits not in SUPPORTED_W_BITS:
        raise ValueError(
            f"DxMoE native kernel requires w_bits in {SUPPORTED_W_BITS}, got {w_bits!r}"
        )
    if k <= 0 or (k * w_bits) % 8 != 0:
        raise ValueError(f"Invalid K={k} for W{w_bits} row byte packing")
    return (k * w_bits) // 8


def compute_dxmoe_arena_layout(
    num_experts: int,
    n: int,
    k: int,
    w_bits_list: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Validate `(E, N, K, w_bits)` and compute 256B-aligned per-expert byte offsets."""
    if len(w_bits_list) != num_experts:
        raise ValueError(
            f"w_bits_list length {len(w_bits_list)} != num_experts {num_experts}"
        )
    if num_experts <= 0 or n <= 0 or k <= 0 or n % 256 != 0 or k % 128 != 0:
        raise ValueError(
            f"DxMoE ABI {DXMOE_ABI_VERSION} requires E > 0, N % 256 == 0, and K % 128 == 0, "
            f"got E={num_experts}, N={n}, K={k}"
        )
    bits_clean = tuple(int(b) for b in w_bits_list)
    if any(b not in SUPPORTED_W_BITS for b in bits_clean):
        raise ValueError(
            f"DxMoE native kernel requires expert w_bits in {{2, 3, 4, 8}}, got {set(bits_clean)}"
        )
    expert_offsets: list[int] = []
    running_bytes = 0
    for b in bits_clean:
        expert_offsets.append(running_bytes)
        expert_bytes = n * ((k * b) // 8)
        running_bytes = round_up(running_bytes + expert_bytes, 256)
    return bits_clean, tuple(expert_offsets), max(running_bytes, 256)


def allocate_dxmoe_weight_buffers(
    num_experts: int,
    n: int,
    k: int,
    w_bits_list: Sequence[int],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Pre-allocate packed DxMoE weight buffers without materializing FP16 weights."""
    bits_clean, expert_offsets, arena_bytes = compute_dxmoe_arena_layout(
        num_experts, n, k, w_bits_list
    )
    codes_arena = torch.zeros((arena_bytes,), dtype=torch.uint8, device=device)
    scales = torch.ones((num_experts, n), dtype=torch.float16, device=device)
    w_bits_dev = torch.tensor(bits_clean, dtype=torch.int32, device=device)
    return codes_arena, scales, w_bits_dev, expert_offsets


def finalize_dxmoe_weight_channel(
    *,
    num_experts: int,
    n: int,
    k: int,
    w_bits_list: Sequence[int],
    codes_arena: torch.Tensor,
    scales: torch.Tensor,
    w_bits_dev: torch.Tensor,
    expert_offsets: Sequence[int],
) -> DxMoELayerWeights:
    """Build device pointer tables (`w_ptrs_dev`, `sw_ptrs_dev`) on finalized packed buffers."""
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("DxMoE weight scales must be finite and strictly positive")
    device = codes_arena.device
    arena_base = int(codes_arena.data_ptr())
    scales_base = int(scales.data_ptr())
    w_ptrs_host = [arena_base + int(off) for off in expert_offsets]
    sw_ptrs_host = [scales_base + e * n * 2 for e in range(num_experts)]
    w_ptrs_dev = torch.tensor(w_ptrs_host, dtype=torch.int64, device=device)
    sw_ptrs_dev = torch.tensor(sw_ptrs_host, dtype=torch.int64, device=device)
    bits_clean = [int(b) for b in w_bits_list]
    return DxMoELayerWeights(
        num_experts=num_experts,
        n=n,
        k=k,
        codes_arena=codes_arena,
        scales=scales,
        w_bits_dev=w_bits_dev,
        w_ptrs_dev=w_ptrs_dev,
        sw_ptrs_dev=sw_ptrs_dev,
        has_w8=any(b == 8 for b in bits_clean),
    )


def repack_gptq_qweight_shard(
    qweight: torch.Tensor,
    *,
    k_expected: int,
    n_expected: int,
    w_bits: int,
    target_device: torch.device,
) -> torch.Tensor:
    """Bit-losslessly transpose GPTQ `int32 [(K * w_bits) // 32, N]` into DxMoE `uint8 [N, K * w_bits // 8]`."""
    if w_bits not in SUPPORTED_W_BITS:
        raise ValueError(
            f"Unsupported DxMoE w_bits={w_bits}; expected one of {SUPPORTED_W_BITS}"
        )
    if k_expected <= 0 or n_expected <= 0 or (k_expected * w_bits) % 32 != 0:
        raise ValueError(
            f"Expected positive K, N with (K * w_bits) % 32 == 0 for W{w_bits}, "
            f"got K={k_expected}, N={n_expected}"
        )
    expected_shape = ((k_expected * w_bits) // 32, n_expected)
    if (
        not isinstance(qweight, torch.Tensor)
        or qweight.ndim != 2
        or qweight.dtype != torch.int32
        or tuple(qweight.shape) != expected_shape
    ):
        raise ValueError(
            f"Expected GPTQ qweight int32 {expected_shape} for W{w_bits}, "
            f"got {getattr(qweight, 'dtype', None)} {tuple(getattr(qweight, 'shape', ()))}"
        )
    qw_dev = qweight.to(device=target_device)
    if sys.byteorder == "little":
        return qw_dev.transpose(0, 1).contiguous().view(torch.uint8)
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=target_device).view(1, 1, 4)
    return (
        ((qw_dev.transpose(0, 1).unsqueeze(-1) >> shifts) & 0xFF)
        .to(dtype=torch.uint8)
        .reshape(n_expected, -1)
        .contiguous()
    )


def validate_gptq_scales_shard(
    scales: torch.Tensor,
    *,
    n_expected: int,
    target_device: torch.device,
) -> torch.Tensor:
    """Validate channel-wise (`group_size == -1`) GPTQ scales `[1, N]` and return `float16 [N]`."""
    if not isinstance(scales, torch.Tensor) or not scales.is_floating_point():
        raise ValueError(f"GPTQ scales must be a floating-point Tensor, got {type(scales)}")
    if scales.ndim == 2 and scales.shape[0] != 1:
        raise ValueError(
            f"DxMoE native operator requires channel-wise GPTQ scales [1, {n_expected}] "
            f"(group_size=-1), got {tuple(scales.shape)}"
        )
    if tuple(scales.shape) not in ((1, n_expected), (n_expected,)):
        raise ValueError(
            f"Expected GPTQ scales shape (1, {n_expected}), got {tuple(scales.shape)}"
        )
    s_f16 = scales.to(device=target_device, dtype=torch.float16).reshape(n_expected)
    if not bool(torch.isfinite(s_f16).all()) or bool((s_f16 <= 0).any()):
        raise ValueError("DxMoE GPTQ weight scales must be finite and strictly positive")
    return s_f16


def validate_gptq_qzeros_shard(
    qzeros: torch.Tensor,
    *,
    n_expected: int,
    w_bits: int,
) -> None:
    """Validate channel-wise (`group_size == -1`) symmetric `gptq-v1` `qzeros`."""
    if w_bits not in SUPPORTED_W_BITS:
        raise ValueError(
            f"Unsupported DxMoE w_bits={w_bits}; expected one of {SUPPORTED_W_BITS}"
        )
    if n_expected <= 0 or (n_expected * w_bits) % 32 != 0:
        raise ValueError(
            f"Expected positive N with (N * w_bits) % 32 == 0 for W{w_bits}, got N={n_expected}"
        )
    expected_shape = (1, (n_expected * w_bits) // 32)
    if not isinstance(qzeros, torch.Tensor) or qzeros.dtype != torch.int32:
        raise ValueError(f"GPTQ qzeros must be int32, got {getattr(qzeros, 'dtype', None)}")
    if qzeros.ndim == 2 and qzeros.shape[0] != 1:
        raise ValueError(
            f"DxMoE native operator requires channel-wise GPTQ qzeros {expected_shape} "
            f"(group_size=-1), got {tuple(qzeros.shape)}"
        )
    if tuple(qzeros.shape) != expected_shape:
        raise ValueError(
            f"Expected GPTQ qzeros shape {expected_shape} for W{w_bits}, got {tuple(qzeros.shape)}"
        )
    if w_bits == 3:
        expected_w3 = torch.tensor(
            GPTQ_SYM_QZEROS_W3, dtype=torch.int32, device=qzeros.device
        ).view(1, 1, 3)
        if not bool((qzeros.reshape(1, n_expected // 32, 3) == expected_w3).all()):
            raise ValueError(
                "DxMoE native operator requires symmetric gptq-v1 qzeros "
                "(expected (0xdb6db6db, 0xb6db6db6, 0x6db6db6d) per 32 channels for W3)"
            )
    else:
        expected_val = GPTQ_SYM_QZEROS[w_bits]
        if not bool((qzeros == expected_val).all()):
            raise ValueError(
                f"DxMoE native operator requires symmetric gptq-v1 qzeros "
                f"(expected 0x{expected_val:08x} for W{w_bits})"
            )


@torch.no_grad()
def pack_dxmoe_float_matrix(
    weight_2d: torch.Tensor,
    w_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refit and pack one 2D symmetric G=-1 float weight `[N, K]` to `(packed_u8, scales_f16)`."""
    if weight_2d.ndim != 2:
        raise ValueError(f"Expected 2D weight [N, K], got {tuple(weight_2d.shape)}")
    if w_bits not in SUPPORTED_W_BITS:
        raise ValueError(f"Expected w_bits in {SUPPORTED_W_BITS}, got {w_bits!r}")
    n, k = int(weight_2d.shape[0]), int(weight_2d.shape[1])
    dxmoe_packed_row_bytes(k, w_bits)
    device = weight_2d.device
    w_f32 = weight_2d.to(dtype=torch.float32)
    zero = 1 << (w_bits - 1)
    qmin_abs = float(zero)
    qmax_pos = float(zero - 1)
    amax = w_f32.abs().amax(dim=-1)
    if w_bits == 2:
        denoms = _W2_REFIT_DENOMS
    elif w_bits == 3:
        denoms = _W3_REFIT_DENOMS
    elif w_bits == 4:
        denoms = _W4_REFIT_DENOMS
    else:
        denoms = _W8_REFIT_DENOMS

    best_err = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
    best_s = torch.ones((n,), dtype=torch.float16, device=device)
    best_q = torch.zeros((n, k), dtype=torch.int16, device=device)

    for m_val in denoms:
        s_cand = torch.where(amax > 0.0, amax / m_val, torch.ones_like(amax))
        q_cand = torch.round(w_f32 / s_cand.unsqueeze(-1)).clamp(-qmin_abs, qmax_pos)
        denom = (q_cand * q_cand).sum(dim=-1)
        s_ls = torch.where(
            denom > 0.0,
            (w_f32 * q_cand).sum(dim=-1) / denom,
            torch.ones_like(amax),
        )
        s_f16 = s_ls.clamp(min=1e-7, max=65504.0).to(dtype=torch.float16)
        s_f32 = s_f16.to(dtype=torch.float32)
        q_fin = torch.round(w_f32 / s_f32.unsqueeze(-1)).clamp(-qmin_abs, qmax_pos)
        err = (q_fin * s_f32.unsqueeze(-1) - w_f32).square().sum(dim=-1)
        improved = err < best_err
        if bool(improved.any()):
            best_err = torch.where(improved, err, best_err)
            best_s = torch.where(improved, s_f16, best_s)
            best_q = torch.where(
                improved.unsqueeze(-1), q_fin.to(dtype=torch.int16), best_q
            )

    u8 = (best_q + zero).to(dtype=torch.uint8)
    if w_bits == 2:
        packed = (
            u8[:, 0::4]
            | (u8[:, 1::4] << 2)
            | (u8[:, 2::4] << 4)
            | (u8[:, 3::4] << 6)
        ).contiguous()
    elif w_bits == 3:
        g = u8.reshape(n, k // 8, 8).to(dtype=torch.int32)
        u24 = torch.zeros((n, k // 8), dtype=torch.int32, device=device)
        for idx in range(8):
            u24 = u24 | ((g[..., idx] & 0x7) << (3 * idx))
        b0 = (u24 & 0xFF).to(dtype=torch.uint8)
        b1 = ((u24 >> 8) & 0xFF).to(dtype=torch.uint8)
        b2 = ((u24 >> 16) & 0xFF).to(dtype=torch.uint8)
        packed = torch.stack((b0, b1, b2), dim=-1).reshape(n, (k * 3) // 8).contiguous()
    elif w_bits == 4:
        packed = (u8[:, 0::2] | (u8[:, 1::2] << 4)).contiguous()
    else:
        packed = u8.contiguous()
    return packed, best_s


def pack_dxmoe_weight_channel(
    weight_3d: torch.Tensor,
    w_bits_list: Sequence[int],
) -> DxMoELayerWeights:
    """Refit and pack reconstructed symmetric G=-1 float weights `[E, N, K]` to ABI 12 W4/W8.

    Note: Legacy float HF DxMoE exports store dequantized FP16/BF16 weights (`safetensors`)
    without raw integer codes or FP32 channel scales. This function executes a
    multi-candidate grid search plus least-squares FP16 scale refit per output channel
    to reconstruct `uint8` W4/W8 codes and `float16` channel scales (whereas standard
    HF GPTQ checkpoints bypass this refit and load losslessly via `repack_gptq_qweight_shard`).
    """
    if weight_3d.ndim != 3:
        raise ValueError(f"Expected 3D MoE weight [E, N, K], got {tuple(weight_3d.shape)}")
    num_experts, n, k = (
        int(weight_3d.shape[0]),
        int(weight_3d.shape[1]),
        int(weight_3d.shape[2]),
    )
    device = weight_3d.device
    codes_arena, scales, w_bits_dev, expert_offsets = allocate_dxmoe_weight_buffers(
        num_experts, n, k, w_bits_list, device=device
    )

    with torch.no_grad():
        for e, b in enumerate(w_bits_list):
            packed, best_s = pack_dxmoe_float_matrix(weight_3d[e], int(b))
            scales[e].copy_(best_s)
            start = expert_offsets[e]
            nbytes = packed.numel()
            codes_arena[start : start + nbytes].copy_(packed.view(-1))

    return finalize_dxmoe_weight_channel(
        num_experts=num_experts,
        n=n,
        k=k,
        w_bits_list=w_bits_list,
        codes_arena=codes_arena,
        scales=scales,
        w_bits_dev=w_bits_dev,
        expert_offsets=expert_offsets,
    )


class DxMoESharedWorkspace:
    """Persistent GPU scratch buffers shared sequentially across all MoE layers."""

    def __init__(
        self,
        *,
        max_tokens: int,
        topk: int,
        num_experts: int,
        hidden_size: int,
        inter_size: int,
        in_dtype: torch.dtype,
        device: torch.device,
        bundle: DxMoELibBundle,
    ) -> None:
        self.max_tokens = int(max_tokens)
        self.topk = int(topk)
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.inter_size = int(inter_size)
        self.in_dtype = in_dtype
        self.device = device

        c_max = max(1, self.max_tokens * self.topk)
        t_max = max(1, self.max_tokens)
        max_k = max(self.hidden_size, self.inter_size)

        self.omega_buf = torch.empty((t_max,), dtype=torch.float32, device=device)
        self.x_tilde = torch.empty(
            (t_max, self.hidden_size), dtype=in_dtype, device=device
        )
        self.entry_amax = torch.empty((t_max,), dtype=torch.float32, device=device)
        self.entry_bits = torch.empty((c_max,), dtype=torch.int32, device=device)

        # Note on BF16 model activations: SM80/SM86 DxMoE Tensor Core MMA tiles consume
        # `pool_a16` in FP16 and emit `gate_up_compact` / `down_compact` in FP16 (with FP32
        # MMA accumulation), before `weighted_unpermute` accumulates in FP32 and casts back
        # to the model's output dtype (FP16 or BF16). BF16 A16 rows therefore undergo an
        # FP16 pool conversion rather than raw BF16 passthrough.
        self.gate_up_compact = torch.empty(
            (c_max, 2 * self.inter_size), dtype=torch.float16, device=device
        )
        self.r1_to_call = torch.empty((c_max,), dtype=torch.int32, device=device)

        self.h_tilde = torch.empty(
            (c_max, self.inter_size), dtype=in_dtype, device=device
        )
        self.down_amax = torch.empty((c_max,), dtype=torch.float32, device=device)
        self.down_ids = torch.empty((c_max,), dtype=torch.int32, device=device)
        self.down_bits = torch.empty((c_max,), dtype=torch.int32, device=device)

        self.down_compact = torch.empty(
            (c_max, self.hidden_size), dtype=torch.float16, device=device
        )
        self.call_to_r2 = torch.empty((c_max,), dtype=torch.int32, device=device)

        self.pool_a4 = torch.empty((c_max, max_k // 2), dtype=torch.uint8, device=device)
        self.pool_a8 = torch.empty((c_max, max_k), dtype=torch.int8, device=device)
        self.pool_a16 = torch.empty((c_max, max_k), dtype=torch.float16, device=device)
        self.scale_a4 = torch.empty((c_max,), dtype=torch.float32, device=device)
        self.scale_a8 = torch.empty((c_max,), dtype=torch.float32, device=device)
        self.pool_offsets = tuple(
            torch.empty((self.num_experts + 1,), dtype=torch.int32, device=device)
            for _ in range(3)
        )

        self.tasks = torch.empty(
            (3 * self.num_experts, _TASK_COLUMNS), dtype=torch.int64, device=device
        )
        self.counts = torch.empty((_COUNTS_LEN,), dtype=torch.int32, device=device)
        self.status = torch.zeros((1,), dtype=torch.int32, device=device)
        self.sticky_status = torch.zeros((1,), dtype=torch.int32, device=device)
        ws_bytes = bundle.reorder_workspace_bytes(t_max, self.topk, self.num_experts)
        self.reorder_ws = torch.empty((ws_bytes,), dtype=torch.uint8, device=device)

        self.pool_data_ptrs = (
            ctypes.c_int64 * 3
        )(
            int(self.pool_a4.data_ptr()),
            int(self.pool_a8.data_ptr()),
            int(self.pool_a16.data_ptr()),
        )
        self.pool_scale_ptrs = (
            ctypes.c_int64 * 2
        )(
            int(self.scale_a4.data_ptr()),
            int(self.scale_a8.data_ptr()),
        )
        self.zero_ptrs_2 = (ctypes.c_int64 * 2)(0, 0)
        self.zero_ptrs_3 = (ctypes.c_int64 * 3)(0, 0, 0)
        self.pool_offsets_ptrs = (
            ctypes.c_int64 * 3
        )(*(int(t.data_ptr()) for t in self.pool_offsets))

        self.entry_outputs_ptrs = (
            ctypes.c_int64 * 3
        )(int(self.gate_up_compact.data_ptr()), 0, 0)
        self.down_outputs_ptrs = (
            ctypes.c_int64 * 3
        )(int(self.down_compact.data_ptr()), 0, 0)

        dev_idx = 0 if device.index is None else int(device.index)
        with _WORKSPACE_LOCK:
            _ACTIVE_WORKSPACES_BY_DEVICE.setdefault(dev_idx, []).append(
                weakref.ref(self)
            )

    def assert_healthy_and_reset(self) -> None:
        """Synchronize sticky_status outside CUDA Graph capture and raise on latched error."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DxMoESharedWorkspace.assert_healthy_and_reset() must run outside CUDA Graph capture"
            )
        code = int(self.sticky_status.item())
        if code != 0:
            self.sticky_status.zero_()
            self.status.zero_()
            detail = DXMOE_STATUS_MESSAGES.get(code, f"native status code {code}")
            raise RuntimeError(
                f"DxMoE native operator latched error status={code}: {detail}"
            )


_WORKSPACE_LOCK = threading.Lock()
_ACTIVE_WORKSPACES_BY_DEVICE: dict[int, list[weakref.ReferenceType[DxMoESharedWorkspace]]] = {}
_FALLBACK_WORKSPACES: dict[tuple[object, ...], DxMoESharedWorkspace] = {}


def check_dxmoe_workspaces_healthy(device: torch.device | int | None = None) -> None:
    """Zero-overhead post-step health check; no-op when DxMoE is inactive or capturing."""
    if not _ACTIVE_WORKSPACES_BY_DEVICE or torch.cuda.is_current_stream_capturing():
        return
    if device is None:
        dev_indices = list(_ACTIVE_WORKSPACES_BY_DEVICE.keys())
    elif isinstance(device, int):
        dev_indices = [device]
    elif device.type == "cuda":
        dev_indices = [0 if device.index is None else int(device.index)]
    else:
        return

    live_workspaces: list[DxMoESharedWorkspace] = []
    with _WORKSPACE_LOCK:
        for idx in dev_indices:
            refs = _ACTIVE_WORKSPACES_BY_DEVICE.get(idx)
            if not refs:
                continue
            kept: list[weakref.ReferenceType[DxMoESharedWorkspace]] = []
            for ref in refs:
                ws = ref()
                if ws is not None:
                    kept.append(ref)
                    live_workspaces.append(ws)
            if kept:
                _ACTIVE_WORKSPACES_BY_DEVICE[idx] = kept
            else:
                _ACTIVE_WORKSPACES_BY_DEVICE.pop(idx, None)

    for ws in live_workspaces:
        ws.assert_healthy_and_reset()


def get_dxmoe_shared_workspace(
    *,
    num_tokens: int,
    max_num_tokens: int,
    topk: int,
    num_experts: int,
    hidden_size: int,
    inter_size: int,
    in_dtype: torch.dtype,
    device: torch.device,
    bundle: DxMoELibBundle,
) -> DxMoESharedWorkspace:
    cap_tokens = max(int(num_tokens), int(max_num_tokens), 1)
    key = (
        DxMoESharedWorkspace,
        cap_tokens,
        int(topk),
        int(num_experts),
        int(hidden_size),
        int(inter_size),
        in_dtype,
        device,
    )

    def _factory() -> DxMoESharedWorkspace:
        return DxMoESharedWorkspace(
            max_tokens=cap_tokens,
            topk=topk,
            num_experts=num_experts,
            hidden_size=hidden_size,
            inter_size=inter_size,
            in_dtype=in_dtype,
            device=device,
            bundle=bundle,
        )

    if is_workspace_manager_initialized():
        return current_workspace_manager().get_persistent_resource(key, _factory)

    ws = _FALLBACK_WORKSPACES.get(key)
    if ws is None:
        ws = _factory()
        _FALLBACK_WORKSPACES[key] = ws
    return ws


class DxMoEWNA16LinearWorkspace:
    """Persistent `E=1, topk=1, A16` scratch for native W2A16/W3A16 non-routed Linears."""

    def __init__(
        self,
        *,
        max_tokens: int,
        k_pad: int,
        n_pad: int,
        device: torch.device,
        bundle: DxMoELibBundle,
    ) -> None:
        self.max_tokens = max(1, int(max_tokens))
        self.k_pad = int(k_pad)
        self.n_pad = int(n_pad)
        self.device = device

        t_max = self.max_tokens
        self.expert_ids = torch.zeros((t_max,), dtype=torch.int32, device=device)
        self.activation_bits = torch.full(
            (t_max,), 16, dtype=torch.int32, device=device
        )
        # Unused A4/A8 pool base addresses (never written when activation_bits == 16).
        self.pool_a4 = torch.empty((256,), dtype=torch.uint8, device=device)
        self.pool_a8 = torch.empty((256,), dtype=torch.int8, device=device)
        self.pool_a16 = torch.empty(
            (t_max, self.k_pad), dtype=torch.float16, device=device
        )
        self.scale_a4 = torch.empty((16,), dtype=torch.float32, device=device)
        self.scale_a8 = torch.empty((16,), dtype=torch.float32, device=device)
        self.pool_offsets = tuple(
            torch.empty((2,), dtype=torch.int32, device=device) for _ in range(3)
        )
        self.out_f16 = torch.empty(
            (t_max, self.n_pad), dtype=torch.float16, device=device
        )
        self.tasks = torch.empty(
            (4, _TASK_COLUMNS), dtype=torch.int64, device=device
        )
        self.counts = torch.empty((_COUNTS_LEN,), dtype=torch.int32, device=device)
        self.status = torch.zeros((1,), dtype=torch.int32, device=device)
        ws_bytes = bundle.reorder_workspace_bytes(t_max, 1, 1)
        self.reorder_ws = torch.empty((ws_bytes,), dtype=torch.uint8, device=device)

        self.pool_data_ptrs = (ctypes.c_int64 * 3)(
            int(self.pool_a4.data_ptr()),
            int(self.pool_a8.data_ptr()),
            int(self.pool_a16.data_ptr()),
        )
        self.pool_scale_ptrs = (ctypes.c_int64 * 2)(
            int(self.scale_a4.data_ptr()),
            int(self.scale_a8.data_ptr()),
        )
        self.zero_ptrs_2 = (ctypes.c_int64 * 2)(0, 0)
        self.zero_ptrs_3 = (ctypes.c_int64 * 3)(0, 0, 0)
        self.pool_offsets_ptrs = (ctypes.c_int64 * 3)(
            *(int(t.data_ptr()) for t in self.pool_offsets)
        )
        self.out_ptrs = (ctypes.c_int64 * 3)(int(self.out_f16.data_ptr()), 0, 0)

        dev_idx = 0 if device.index is None else int(device.index)
        with _WORKSPACE_LOCK:
            _ACTIVE_WORKSPACES_BY_DEVICE.setdefault(dev_idx, []).append(
                weakref.ref(self)  # type: ignore[arg-type]
            )

    def assert_healthy_and_reset(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DxMoEWNA16LinearWorkspace.assert_healthy_and_reset() must run outside CUDA Graph capture"
            )
        code = int(self.status.item())
        if code != 0:
            self.status.zero_()
            detail = DXMOE_STATUS_MESSAGES.get(code, f"native status code {code}")
            raise RuntimeError(
                f"DxMoE native WNA16 linear latched error status={code}: {detail}"
            )


def get_dxmoe_wna16_linear_workspace(
    *,
    num_tokens: int,
    max_num_tokens: int,
    k_pad: int,
    n_pad: int,
    device: torch.device,
    bundle: DxMoELibBundle,
) -> DxMoEWNA16LinearWorkspace:
    cap_tokens = max(int(num_tokens), int(max_num_tokens), 1)
    key = (
        DxMoEWNA16LinearWorkspace,
        cap_tokens,
        int(k_pad),
        int(n_pad),
        device,
    )

    def _factory() -> DxMoEWNA16LinearWorkspace:
        return DxMoEWNA16LinearWorkspace(
            max_tokens=cap_tokens,
            k_pad=k_pad,
            n_pad=n_pad,
            device=device,
            bundle=bundle,
        )

    if is_workspace_manager_initialized():
        return current_workspace_manager().get_persistent_resource(key, _factory)

    ws = _FALLBACK_WORKSPACES.get(key)  # type: ignore[assignment]
    if ws is None:
        ws = _factory()  # type: ignore[assignment]
        _FALLBACK_WORKSPACES[key] = ws
    return ws  # type: ignore[return-value]


def run_dxmoe_wna16_linear(
    x_2d: torch.Tensor,
    *,
    w_bits_dev: torch.Tensor,
    w_ptrs_dev: torch.Tensor,
    sw_ptrs_dev: torch.Tensor,
    k_orig: int,
    n_orig: int,
    k_pad: int,
    n_pad: int,
    max_num_tokens: int,
) -> torch.Tensor:
    """Execute native CUDA W2A16/W3A16 GEMM on packed symmetric G=-1 weights."""
    if x_2d.device.type != "cuda":
        raise RuntimeError(
            f"DxMoE native WNA16 linear requires a CUDA tensor, got device={x_2d.device}"
        )
    if x_2d.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"DxMoE native WNA16 linear requires float16 or bfloat16 activations, got {x_2d.dtype}"
        )
    m = int(x_2d.shape[0])
    device = x_2d.device
    if m == 0:
        return torch.empty((0, n_orig), dtype=x_2d.dtype, device=device)

    if k_orig < k_pad:
        x_in = torch.zeros((m, k_pad), dtype=x_2d.dtype, device=device)
        x_in[:, :k_orig].copy_(x_2d)
    elif not x_2d.is_contiguous() or (int(x_2d.data_ptr()) % 16 != 0):
        x_in = x_2d.contiguous()
        if int(x_in.data_ptr()) % 16 != 0:
            x_in = x_in.clone()
    else:
        x_in = x_2d

    bundle = get_dxmoe_lib_bundle(device)
    lib, (serial_ctas, cons_ctas, prod_ctas) = bundle.select(m)
    ws = get_dxmoe_wna16_linear_workspace(
        num_tokens=m,
        max_num_tokens=max_num_tokens,
        k_pad=k_pad,
        n_pad=n_pad,
        device=device,
        bundle=bundle,
    )
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)

    desc = NativeReorderDesc(
        x=int(x_in.data_ptr()),
        row_amax=0,
        expert_ids=int(ws.expert_ids.data_ptr()),
        activation_bits=int(ws.activation_bits.data_ptr()),
        expert_w_bits=int(w_bits_dev.data_ptr()),
        expert_w_ptrs=int(w_ptrs_dev.data_ptr()),
        expert_sw_ptrs=int(sw_ptrs_dev.data_ptr()),
        in_row_to_call=0,
        out_row_to_call=0,
        out_call_to_row=0,
        pool_data=ws.pool_data_ptrs,
        pool_scale=ws.pool_scale_ptrs,
        pool_group_sum=ws.zero_ptrs_2,
        pool_offsets=ws.pool_offsets_ptrs,
        pool_call_ids=ws.zero_ptrs_3,
        pool_outputs=ws.out_ptrs,
        workspace=int(ws.reorder_ws.data_ptr()),
        tasks=int(ws.tasks.data_ptr()),
        counts=int(ws.counts.data_ptr()),
        status=int(ws.status.data_ptr()),
        is_padding=0,
        num_tokens=m,
        topk=1,
        num_experts=1,
        k=k_pad,
        n=n_pad,
        rows_are_calls=0,
        x_dtype=DTYPE_CODE[x_in.dtype],
        a16_dtype=0,
        group_size=-1,
        compact_outputs=1,
        num_valid_tokens=m,
        preserve_prior_status=1,
    )
    err = int(
        lib.smoe_dxmoe_fused_run(
            ctypes.byref(desc),
            serial_ctas,
            cons_ctas,
            prod_ctas,
            0,
            stream_ptr,
        )
    )
    if err != 0:
        raise RuntimeError(f"DxMoE native WNA16 linear failed with CUDA status {err}")

    out_view = ws.out_f16[:m, :n_orig]
    if x_2d.dtype == torch.float16:
        return out_view.clone()
    return out_view.to(dtype=x_2d.dtype)
