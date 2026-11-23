# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native C++/CUDA ctypes bridge, weight packer, and workspace manager for DxMoE."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
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
    """Packed symmetric G=-1 W4/W8 expert payload for one MoE projection stage."""

    num_experts: int
    n: int
    k: int
    codes_arena: torch.Tensor  # uint8 [total_packed_bytes] (256B-aligned per expert)
    scales: torch.Tensor  # float16 [E, N]
    w_bits_dev: torch.Tensor  # int32 [E]
    w_ptrs_dev: torch.Tensor  # int64 [E]
    sw_ptrs_dev: torch.Tensor  # int64 [E]
    has_w8: bool


def pack_dxmoe_weight_channel(
    weight_3d: torch.Tensor,
    w_bits_list: Sequence[int],
) -> DxMoELayerWeights:
    """Refit and pack reconstructed symmetric G=-1 float weights `[E, N, K]` to ABI 12 W4/W8.

    Note: Standard HF DxMoE exports store dequantized FP16/BF16 weights (`safetensors`)
    without raw integer codes or FP32 channel scales. This function executes a
    multi-candidate grid search plus least-squares FP16 scale refit per output channel
    to reconstruct `uint8` W4/W8 codes and `float16` channel scales (not a bit-lossless
    recovery of the offline quantizer's internal integer state).
    """
    if weight_3d.ndim != 3:
        raise ValueError(f"Expected 3D MoE weight [E, N, K], got {tuple(weight_3d.shape)}")
    num_experts, n, k = (
        int(weight_3d.shape[0]),
        int(weight_3d.shape[1]),
        int(weight_3d.shape[2]),
    )
    if len(w_bits_list) != num_experts:
        raise ValueError(
            f"w_bits_list length {len(w_bits_list)} != num_experts {num_experts}"
        )
    if n <= 0 or k <= 0 or n % 256 != 0 or k % 128 != 0:
        raise ValueError(
            f"DxMoE ABI {DXMOE_ABI_VERSION} requires N % 256 == 0 and K % 128 == 0, got N={n}, K={k}"
        )
    bits_clean = [int(b) for b in w_bits_list]
    if any(b not in (4, 8) for b in bits_clean):
        raise ValueError(
            f"DxMoE native kernel requires expert w_bits in {{4, 8}}, got {set(bits_clean)}"
        )

    device = weight_3d.device
    expert_offsets: list[int] = []
    running_bytes = 0
    for b in bits_clean:
        expert_offsets.append(running_bytes)
        expert_bytes = n * (k // 2 if b == 4 else k)
        running_bytes = round_up(running_bytes + expert_bytes, 256)

    codes_arena = torch.empty((max(running_bytes, 256),), dtype=torch.uint8, device=device)
    scales = torch.empty((num_experts, n), dtype=torch.float16, device=device)

    with torch.no_grad():
        w4_denoms = (8.0, 7.5, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
        w8_denoms = tuple(float(m) for m in range(128, 95, -1)) + (
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
        for e, b in enumerate(bits_clean):
            w_f32 = weight_3d[e].to(dtype=torch.float32)
            qmin_abs, qmax_pos, zero = (8.0, 7.0, 8) if b == 4 else (128.0, 127.0, 128)
            amax = w_f32.abs().amax(dim=-1)
            denoms = w4_denoms if b == 4 else w8_denoms

            best_err = torch.full((n,), float("inf"), dtype=torch.float32, device=device)
            best_s = torch.ones((n,), dtype=torch.float16, device=device)
            best_q = torch.zeros((n, k), dtype=torch.int16, device=device)

            for m_val in denoms:
                s_cand = torch.where(amax > 0.0, amax / m_val, torch.ones_like(amax))
                q_cand = torch.round(w_f32 / s_cand.unsqueeze(-1)).clamp(
                    -qmin_abs, qmax_pos
                )
                denom = (q_cand * q_cand).sum(dim=-1)
                s_ls = torch.where(
                    denom > 0.0,
                    (w_f32 * q_cand).sum(dim=-1) / denom,
                    torch.ones_like(amax),
                )
                s_f16 = s_ls.clamp(min=1e-7, max=65504.0).to(dtype=torch.float16)
                s_f32 = s_f16.to(dtype=torch.float32)
                q_fin = torch.round(w_f32 / s_f32.unsqueeze(-1)).clamp(
                    -qmin_abs, qmax_pos
                )
                err = (q_fin * s_f32.unsqueeze(-1) - w_f32).square().sum(dim=-1)
                improved = err < best_err
                if bool(improved.any()):
                    best_err = torch.where(improved, err, best_err)
                    best_s = torch.where(improved, s_f16, best_s)
                    best_q = torch.where(
                        improved.unsqueeze(-1), q_fin.to(dtype=torch.int16), best_q
                    )

            scales[e].copy_(best_s)
            u8 = (best_q + zero).to(dtype=torch.uint8)
            packed = (
                (u8[:, 0::2] | (u8[:, 1::2] << 4)).contiguous()
                if b == 4
                else u8.contiguous()
            )

            start = expert_offsets[e]
            nbytes = packed.numel()
            codes_arena[start : start + nbytes].copy_(packed.view(-1))

    arena_base = int(codes_arena.data_ptr())
    scales_base = int(scales.data_ptr())
    w_ptrs_host = [arena_base + off for off in expert_offsets]
    sw_ptrs_host = [scales_base + e * n * 2 for e in range(num_experts)]

    w_bits_dev = torch.tensor(bits_clean, dtype=torch.int32, device=device)
    w_ptrs_dev = torch.tensor(w_ptrs_host, dtype=torch.int64, device=device)
    sw_ptrs_dev = torch.tensor(sw_ptrs_host, dtype=torch.int64, device=device)

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
