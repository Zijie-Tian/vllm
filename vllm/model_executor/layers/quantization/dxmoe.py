# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DxMoE offline static W2/W3/W4/W8 + online dynamic A4/A8/A16 quantization config & MoE method."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from safetensors import safe_open
import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.dxmoe_experts import (
    DxMoEExperts,
    DxMoELayerRuntimeState,
    DxMoEMoEQuantConfig,
)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
)
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    replace_parameter,
    set_weight_attrs,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.utils.dxmoe import (
    GPTQ_SYM_QZEROS,
    GPTQ_SYM_QZEROS_W3,
    PIPELINED_CODE,
    SCHEDULE_MODE_CODE,
    SUPPORTED_W_BITS,
    allocate_dxmoe_weight_buffers,
    dxmoe_packed_row_bytes,
    finalize_dxmoe_weight_channel,
    get_dxmoe_lib_bundle,
    get_dxmoe_wna16_linear_workspace,
    pack_dxmoe_float_matrix,
    repack_gptq_qweight_shard,
    run_dxmoe_wna16_linear,
    validate_gptq_qzeros_shard,
    validate_gptq_scales_shard,
)
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

logger = init_logger(__name__)

_DXMOE_META_FORMAT = "scaling-moe.dxmoe.meta"
_DXMOE_THRESHOLDS_FORMAT = "scaling-moe.dxmoe.thresholds"
_DXMOE_THRESHOLDS_SUBDIR = "thresholds"
_DXMOE_WEIGHT_FP_SCHEMA = "scaling-moe.dxmoe.weight-fp.v2"
_DXMOE_META_FILENAME = "dxmoe_meta.json"
_DXMOE_SIGNS_FILENAME = "dxmoe_signs.safetensors"
_LAYER_IDX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_SHA256_PREFIX_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_V2_KEYS = frozenset(
    {
        "model",
        "model_dir",
        "out_dir",
        "path",
        "file_path",
        "elapsed_s",
        "elapsed_seconds",
        "timestamp",
        "created_at",
        "updated_at",
        "hostname",
        "host",
        "runs",
        "counts",
        "entry_counts",
        "down_counts",
        "calib_realized",
        "threshold_calibration",
    }
)
_SUPPORTED_SM_CAPS = ((8, 0), (8, 6))
_SUPPORTED_ROUTING_METHODS = (
    RoutingMethodType.Default,
    RoutingMethodType.RenormalizeNaive,
    RoutingMethodType.Renormalize,
)


@dataclass(frozen=True)
class _ParsedDxMoELayerSpec:
    layer_idx: int
    hidden_size: int
    inter_size: int
    num_experts: int
    top_k: int
    w_bits_entry: tuple[int, ...]
    w_bits_down: tuple[int, ...]
    beta_entry: tuple[float, ...]
    beta_down: tuple[float, ...]
    tau_entry: tuple[float, float]
    tau_down: tuple[float, float]
    s_in_cpu: torch.Tensor | None
    s_mid_cpu: torch.Tensor | None
    schedule_mode: str
    apply_coupling_entry: bool
    apply_coupling_down: bool
    use_omega: bool
    sink_tokens: int
    local_window: int
    pipelined_code: int


def _parse_bool_env(name: str, default: bool | None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    norm = raw.strip().lower()
    if norm in ("1", "true", "yes", "on"):
        return True
    if norm in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


def _load_signs_from_safetensors(
    signs_path: Path, expected_layers: set[int]
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if not signs_path.is_file():
        raise FileNotFoundError(f"DxMoE Hadamard signs file not found: {signs_path}")
    s_in_map: dict[int, torch.Tensor] = {}
    s_mid_map: dict[int, torch.Tensor] = {}
    with safe_open(str(signs_path), framework="pt", device="cpu") as reader:
        for key in reader.keys():
            parts = key.split(".")
            if (
                len(parts) != 3
                or parts[0] != "layer"
                or not parts[1].isdigit()
                or parts[2] not in ("s_in", "s_mid", "s_mid_shared")
            ):
                raise ValueError(f"Unexpected tensor key in {signs_path}: {key!r}")
            if parts[2] == "s_mid_shared":
                continue
            l_idx = int(parts[1])
            vec = reader.get_tensor(key).detach().to(device="cpu", dtype=torch.int8).contiguous()
            if vec.ndim != 1 or not bool(((vec == -1) | (vec == 1)).all()):
                raise ValueError(f"Invalid +/-1 sign vector {key!r} in {signs_path}")
            if parts[2] == "s_in":
                s_in_map[l_idx] = vec
            else:
                s_mid_map[l_idx] = vec
    missing = expected_layers - (set(s_in_map) & set(s_mid_map))
    if missing:
        raise ValueError(
            f"Missing s_in/s_mid Hadamard signs for layers {sorted(missing)} in {signs_path}"
        )
    return {idx: (s_in_map[idx], s_mid_map[idx]) for idx in sorted(expected_layers)}


def _load_shared_mid_signs(signs_path: Path) -> dict[int, torch.Tensor]:
    found: dict[int, torch.Tensor] = {}
    if not signs_path.is_file():
        return found
    with safe_open(str(signs_path), framework="pt", device="cpu") as reader:
        for key in reader.keys():
            parts = key.split(".")
            if (
                len(parts) == 3
                and parts[0] == "layer"
                and parts[1].isdigit()
                and parts[2] == "s_mid_shared"
            ):
                vec = (
                    reader.get_tensor(key)
                    .detach()
                    .to(device="cpu", dtype=torch.int8)
                    .contiguous()
                )
                if vec.ndim != 1 or not bool(((vec == -1) | (vec == 1)).all()):
                    raise ValueError(f"Invalid +/-1 sign vector {key!r} in {signs_path}")
                found[int(parts[1])] = vec
    return found


def _fmt_nonneg_f64(value: Any, label: str) -> str:
    if (
        type(value) is bool
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{label} must be finite and non-negative, got {value!r}")
    norm = 0.0 if value == 0 else float(value)
    return f"{norm:.17g}"


def _canonical_shared_ffn(shared_ffn: Any) -> list[list[Any]] | None:
    if shared_ffn is None:
        return None
    if not isinstance(shared_ffn, dict):
        raise ValueError(f"shared_ffn must be None or a dict, got {type(shared_ffn).__name__}")
    out: list[list[Any]] = []
    for cname in sorted(shared_ffn):
        cval = shared_ffn[cname]
        if not isinstance(cval, dict):
            raise ValueError(f"shared_ffn[{cname!r}] must be a dict")
        projs: list[list[Any]] = []
        for pname in sorted(cval):
            pinfo = cval[pname]
            if not isinstance(pinfo, dict) or type(pinfo.get("w_bits")) is not int:
                raise ValueError(f"shared_ffn[{cname!r}][{pname!r}] missing int w_bits")
            projs.append([str(pname), int(pinfo["w_bits"])])
        out.append([str(cname), projs])
    return out


def _sign_vec_hex(vec: torch.Tensor, expected_len: int | None, label: str) -> str:
    cpu = vec.detach().to(device="cpu", dtype=torch.int8).contiguous()
    if cpu.ndim != 1 or cpu.numel() == 0 or not bool(((cpu == -1) | (cpu == 1)).all()):
        raise ValueError(f"Invalid +/-1 sign vector {label}")
    if expected_len is not None and cpu.numel() != expected_len:
        raise ValueError(f"{label} length {cpu.numel()} != expected {expected_len}")
    return bytes(cpu.view(torch.uint8).tolist()).hex()


def _compute_dxmoe_weight_fingerprint(
    meta: dict[str, Any],
    signs_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
    shared_mid_signs: dict[int, torch.Tensor],
) -> str:
    """Compute canonical `sha256:<64-hex>` static weight fingerprint identical to scaling_moe."""
    had = meta.get("hadamard", {})
    quant = meta.get("quantization", {})
    had_mode = str(had.get("mode", "none"))
    had_block_size = int(had["block_size"]) if had_mode == "block128" else None
    group_size = int(quant["group_size"])
    sym = bool(quant["sym"])

    canonical_layers: list[dict[str, Any]] = []
    for l_idx, l_meta in sorted(
        ((int(k), v) for k, v in meta["layers"].items()), key=lambda kv: kv[0]
    ):
        h_size = int(l_meta["hidden_size"])
        i_size = int(l_meta["moe_intermediate_size"])
        n_exp = int(l_meta["num_experts"])
        top_k = int(l_meta["top_k"])
        shared_rot = bool(l_meta.get("shared_ffn_rotated", False))
        experts_in = l_meta["experts"]
        exp_rows: list[list[Any]] = []
        for e_idx in range(n_exp):
            ed = experts_in[str(e_idx)]
            exp_rows.append(
                [
                    int(ed["gate"]["w_bits"]),
                    _fmt_nonneg_f64(ed["gate"]["beta"], f"Layer {l_idx} expert {e_idx} beta"),
                    int(ed["up"]["w_bits"]),
                    _fmt_nonneg_f64(ed["up"]["beta"], f"Layer {l_idx} expert {e_idx} beta"),
                    int(ed["entry"]["w_bits"]),
                    _fmt_nonneg_f64(ed["entry"]["beta"], f"Layer {l_idx} expert {e_idx} beta"),
                    int(ed["down"]["w_bits"]),
                    _fmt_nonneg_f64(ed["down"]["beta"], f"Layer {l_idx} expert {e_idx} beta"),
                ]
            )
        canonical_layers.append(
            {
                "layer": l_idx,
                "hidden_size": h_size,
                "moe_intermediate_size": i_size,
                "num_experts": n_exp,
                "top_k": top_k,
                "shared_ffn_rotated": shared_rot,
                "shared_ffn": _canonical_shared_ffn(l_meta.get("shared_ffn")),
                "experts": exp_rows,
            }
        )

    header = {
        "schema": _DXMOE_WEIGHT_FP_SCHEMA,
        "hadamard_mode": had_mode,
        "hadamard_block_size": had_block_size,
        "group_size": group_size,
        "sym": sym,
        "layers": canonical_layers,
    }
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    if had_mode == "block128":
        for l_entry in canonical_layers:
            l_idx = l_entry["layer"]
            s_in, s_mid = signs_by_layer[l_idx]
            s_in_hex = _sign_vec_hex(s_in, l_entry["hidden_size"], f"layer.{l_idx}.s_in")
            s_mid_hex = _sign_vec_hex(
                s_mid, l_entry["moe_intermediate_size"], f"layer.{l_idx}.s_mid"
            )
            if l_entry["shared_ffn_rotated"] and l_idx in shared_mid_signs:
                s_sh_hex = _sign_vec_hex(
                    shared_mid_signs[l_idx], None, f"layer.{l_idx}.s_mid_shared"
                )
            else:
                s_sh_hex = ""
            hasher.update(
                f"|L{l_idx}:in={s_in_hex}:mid={s_mid_hex}:sh={s_sh_hex}".encode("ascii")
            )
    return f"sha256:{hasher.hexdigest()}"


def _format_pct_token(ratio: float) -> str:
    pct = round(float(ratio) * 100.0, 6)
    if abs(pct - round(pct)) < 1e-9:
        return str(int(round(pct)))
    return f"{pct:.6f}".rstrip("0").rstrip(".")


def _format_ratio_tag(ratios: list[Any] | tuple[Any, ...]) -> str:
    return (
        f"r{_format_pct_token(float(ratios[0]))}-"
        f"{_format_pct_token(float(ratios[1]))}-"
        f"{_format_pct_token(float(ratios[2]))}"
    )


def _check_no_forbidden_v2_keys(node: Any, path: str = "$") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _FORBIDDEN_V2_KEYS:
                raise ValueError(
                    f"Forbidden non-deterministic or legacy field {k!r} at {path}.{k} "
                    "in v2 thresholds JSON"
                )
            _check_no_forbidden_v2_keys(v, f"{path}.{k}")
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _check_no_forbidden_v2_keys(item, f"{path}[{idx}]")


def _validate_tau_pair(raw_pair: Any, l_idx: int, stage_name: str) -> tuple[float, float]:
    if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
        raise ValueError(
            f"Layer {l_idx} missing 2-element {stage_name}, got {raw_pair!r}"
        )
    t0 = float(raw_pair[0])
    t1 = float(raw_pair[1])
    if not (math.isfinite(t0) and math.isfinite(t1) and 0.0 <= t0 <= t1):
        raise ValueError(
            f"Layer {l_idx} invalid {stage_name}={(t0, t1)}; "
            "expected finite 0 <= tau1 <= tau2"
        )
    return (t0, t1)


def _extract_tau_from_external_doc(
    ext_doc: Any,
    *,
    threshold_key: str,
    expected_layer_ids: set[int],
    expected_weight_fp: str,
) -> dict[int, tuple[tuple[float, float], tuple[float, float]]]:
    if not isinstance(ext_doc, dict):
        raise ValueError("External DxMoE thresholds JSON must be a JSON object")
    if "format" in ext_doc or "version" in ext_doc:
        if ext_doc.get("format") != _DXMOE_THRESHOLDS_FORMAT or ext_doc.get(
            "version"
        ) not in (1, 2):
            raise ValueError(
                f"Expected thresholds JSON format={_DXMOE_THRESHOLDS_FORMAT!r} "
                f"version in (1, 2), got format={ext_doc.get('format')!r} "
                f"version={ext_doc.get('version')!r}"
            )

    if ext_doc.get("version") == 2:
        _check_no_forbidden_v2_keys(ext_doc)
        mspec = ext_doc.get("model_spec")
        if not isinstance(mspec, dict):
            raise ValueError("v2 thresholds JSON missing 'model_spec' object")
        file_fp = mspec.get("weight_fingerprint")
        if type(file_fp) is not str or not _SHA256_PREFIX_RE.match(file_fp):
            raise ValueError(f"Invalid v2 model_spec.weight_fingerprint={file_fp!r}")
        if file_fp != expected_weight_fp:
            raise ValueError(
                f"DxMoE threshold weight_fingerprint mismatch: threshold file has {file_fp}, "
                f"but loaded model has {expected_weight_fp}"
            )
        layers_v2 = ext_doc.get("layers")
        if not isinstance(layers_v2, dict):
            raise ValueError("v2 thresholds JSON 'layers' must be a JSON object")
        if {int(k) for k in layers_v2 if str(k).isdigit()} != expected_layer_ids:
            raise ValueError(
                f"v2 thresholds layers {sorted(layers_v2)} do not match "
                f"model MoE layers {sorted(expected_layer_ids)}"
            )
        out_v2: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
        for l_idx in sorted(expected_layer_ids):
            lv = layers_v2[str(l_idx)]
            if not isinstance(lv, dict) or set(lv) != {"tau_entry", "tau_down"}:
                raise ValueError(
                    f"v2 thresholds layer {l_idx} must contain exact keys "
                    f"['tau_entry', 'tau_down'], got {lv!r}"
                )
            out_v2[l_idx] = (
                _validate_tau_pair(lv["tau_entry"], l_idx, "tau_entry"),
                _validate_tau_pair(lv["tau_down"], l_idx, "tau_down"),
            )
        return out_v2

    # Legacy v1 external thresholds JSON
    ext_layers = ext_doc.get("layers", ext_doc)
    if not isinstance(ext_layers, dict):
        raise ValueError("External DxMoE thresholds 'layers' must be a JSON object")
    out_v1: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for l_idx in sorted(expected_layer_ids):
        l_str = str(l_idx)
        if l_str not in ext_layers:
            raise ValueError(
                f"External thresholds JSON missing layer {l_idx}; "
                f"available layers: {sorted(ext_layers)}"
            )
        layer_ext = ext_layers[l_str]
        if not isinstance(layer_ext, dict):
            raise ValueError(f"External thresholds entry for layer {l_idx} must be a dict")
        if "tau_entry" in layer_ext or "tau_down" in layer_ext or "tau" in layer_ext:
            tdoc = layer_ext
        else:
            t_map = layer_ext.get("thresholds", layer_ext)
            if not isinstance(t_map, dict) or threshold_key not in t_map:
                raise ValueError(
                    f"schedule_mode='dxmoe' requires calibrated thresholds under "
                    f"threshold_key={threshold_key!r} for layer {l_idx}; "
                    f"available keys: {sorted(t_map) if isinstance(t_map, dict) else []}"
                )
            tdoc = t_map[threshold_key]
        raw_te = tdoc.get("tau_entry", tdoc.get("tau"))
        raw_td = tdoc.get("tau_down", tdoc.get("tau"))
        out_v1[l_idx] = (
            _validate_tau_pair(raw_te, l_idx, "tau_entry"),
            _validate_tau_pair(raw_td, l_idx, "tau_down"),
        )
    return out_v1


def _matches_threshold_file_key(path: Path, doc: dict[str, Any], key: str) -> bool:
    if key == path.name or key == path.stem:
        return True
    if doc.get("version") == 2:
        prof = doc.get("profile", {})
        if not isinstance(prof, dict):
            return False
        if str(prof.get("name", "")) == key:
            return True
        ratios = prof.get("ratios")
        if isinstance(ratios, list) and len(ratios) == 3:
            if _format_ratio_tag(ratios) == key:
                return True
            t_bits = round(
                4.0 * float(ratios[0]) + 8.0 * float(ratios[1]) + 16.0 * float(ratios[2]),
                6,
            )
            if f"b{t_bits:.2f}" == key:
                return True
        return False
    ext_layers = doc.get("layers", doc)
    if isinstance(ext_layers, dict) and ext_layers:
        first_val = next(iter(ext_layers.values()))
        if isinstance(first_val, dict):
            t_map = first_val.get("thresholds", first_val)
            if isinstance(t_map, dict) and key in t_map:
                return True
    return False


def _normalized_hadamard_128(device: torch.device) -> torch.Tensor:
    """Build the normalized Sylvester H128 once, outside model forward."""
    base = torch.ones((1, 1), dtype=torch.float64)
    size = 1
    while size < 128:
        base = torch.cat(
            (torch.cat((base, base), dim=1), torch.cat((base, -base), dim=1)),
            dim=0,
        )
        size *= 2
    return (base / math.sqrt(128.0)).to(device=device, dtype=torch.float32)


class DxMoESharedFFNRotation(torch.nn.Module):
    """Device-resident block128 rotation used by one shared FFN."""

    def __init__(self, s_in: torch.Tensor, s_mid: torch.Tensor) -> None:
        super().__init__()
        device = torch.get_default_device()
        self.register_buffer(
            "s_in", s_in.to(device=device, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "s_mid", s_mid.to(device=device, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "hadamard", _normalized_hadamard_128(device), persistent=False
        )

    def _apply_rotation(self, x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        if dim % 128 != 0 or signs.numel() != dim:
            raise ValueError(
                f"shared Hadamard dim {dim} does not match signs {tuple(signs.shape)}"
            )
        if signs.device != x.device or self.hadamard.device != x.device:
            raise RuntimeError(
                "DxMoE shared Hadamard buffers must be on the activation device"
            )
        signed = x.to(dtype=torch.float32) * signs
        blocks = signed.reshape(*x.shape[:-1], dim // 128, 128)
        return torch.matmul(blocks, self.hadamard).reshape(x.shape).to(dtype=x.dtype)

    def input(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_rotation(x, self.s_in)

    def intermediate(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_rotation(x, self.s_mid)


def prepare_shared_ffn_rotation(
    quant_config: Any,
    prefix: str,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> DxMoESharedFFNRotation | None:
    """Create shared rotation buffers only for explicitly marked new artifacts."""
    if not isinstance(quant_config, DxMoEConfig):
        return None
    if "shared_expert" not in prefix:
        return None
    match = re.search(r"layers\.(\d+)", prefix)
    if match is None:
        raise ValueError(f"Cannot parse layer index from shared FFN prefix {prefix!r}")
    layer_idx = int(match.group(1))
    quant_config._ensure_resolved()
    if not quant_config.shared_ffn_rotated_layers.get(layer_idx, False):
        return None
    if quant_config.hadamard_mode != "block128":
        raise ValueError(
            f"Layer {layer_idx} marks shared_ffn_rotated without block128 Hadamard"
        )
    pair = quant_config.signs_by_layer.get(layer_idx)
    if pair is None:
        raise ValueError(f"Layer {layer_idx} rotated shared FFN is missing s_in/s_mid")
    s_in, s_mid = pair
    if s_in.numel() != hidden_size:
        raise ValueError(
            f"Layer {layer_idx} shared hidden size {hidden_size} != s_in length {s_in.numel()}"
        )
    if intermediate_size == s_mid.numel():
        selected_mid = s_mid
    else:
        selected_mid = quant_config.shared_mid_signs.get(layer_idx)
        if selected_mid is None:
            raise ValueError(
                f"Layer {layer_idx} shared intermediate size {intermediate_size} differs "
                f"from routed size {s_mid.numel()} but s_mid_shared is missing"
            )
        if selected_mid.numel() != intermediate_size:
            raise ValueError(
                f"Layer {layer_idx} shared intermediate size {intermediate_size} != "
                f"s_mid_shared length {selected_mid.numel()}"
            )
    return DxMoESharedFFNRotation(s_in, selected_mid)


class DxMoEConfig(QuantizationConfig):
    """QuantizationConfig loading `dxmoe_meta.json` + `dxmoe_signs.safetensors`."""

    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, raw_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.raw_config: dict[str, Any] = dict(raw_config or {})
        self._resolved_specs: dict[int, _ParsedDxMoELayerSpec] | None = None
        self._checkpoint_meta_resolved: bool = False
        self._safetensors_param_meta: dict[str, dict[str, Any]] = {}
        self._hf_gptq_raw_cfg: dict[str, Any] | None = None
        self._explicit_model_name: str | None = None
        self._explicit_revision: str | None = None
        self.hadamard_mode = "none"
        self.signs_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.shared_mid_signs: dict[int, torch.Tensor] = {}
        self.shared_ffn_rotated_layers: dict[int, bool] = {}

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "dxmoe"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return [_DXMOE_META_FILENAME]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DxMoEConfig":
        return cls(raw_config=config)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: Any, user_quant: str | None, hf_config: Any = None
    ) -> QuantizationMethods | None:
        if user_quant == "dxmoe":
            return "dxmoe"
        if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") == "dxmoe":
            return "dxmoe"
        return None

    def _resolve_model_dir(self) -> Path | None:
        explicit = (
            self._explicit_model_name
            or self.raw_config.get("_model_dir")
            or self.raw_config.get("model_dir")
            or os.environ.get("VLLM_DXMOE_MODEL_DIR")
        )
        if explicit:
            p = Path(str(explicit))
            if p.is_dir():
                return p
        vllm_cfg = get_current_vllm_config_or_none()
        if vllm_cfg is not None and vllm_cfg.model_config is not None:
            candidate = Path(str(vllm_cfg.model_config.model))
            if candidate.is_dir():
                return candidate
        return None

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: Any = None,
        revision: str | None = None,
    ) -> None:
        self._explicit_model_name = model_name
        self._explicit_revision = revision
        if hf_config is not None:
            qcfg = getattr(hf_config, "quantization_config", None)
            if isinstance(qcfg, dict) and str(qcfg.get("quant_method", "")).lower() == "gptq":
                self._hf_gptq_raw_cfg = dict(qcfg)
        self._ensure_checkpoint_metadata()

    def _ensure_checkpoint_metadata(self) -> dict[str, dict[str, Any]]:
        if self._checkpoint_meta_resolved:
            return self._safetensors_param_meta
        self._checkpoint_meta_resolved = True

        if (
            self._hf_gptq_raw_cfg is None
            and str(self.raw_config.get("quant_method", "")).lower() == "gptq"
        ):
            self._hf_gptq_raw_cfg = dict(self.raw_config)

        model_dir = self._resolve_model_dir()
        if model_dir is None:
            return self._safetensors_param_meta

        cfg_json = model_dir / "config.json"
        if self._hf_gptq_raw_cfg is None and cfg_json.is_file():
            cfg_doc = json.loads(cfg_json.read_text(encoding="utf-8"))
            qcfg = cfg_doc.get("quantization_config")
            if isinstance(qcfg, dict) and str(qcfg.get("quant_method", "")).lower() == "gptq":
                self._hf_gptq_raw_cfg = dict(qcfg)

        has_safetensors = any(
            p.is_file()
            and p.suffix == ".safetensors"
            and p.name != _DXMOE_SIGNS_FILENAME
            for p in model_dir.iterdir()
        )
        if has_safetensors:
            self._safetensors_param_meta = dict(
                get_safetensors_params_metadata(
                    str(model_dir), revision=self._explicit_revision
                )
            )
        return self._safetensors_param_meta

    def _resolve_linear_shard_prefixes(
        self,
        prefix: str,
        param_meta: dict[str, dict[str, Any]],
    ) -> list[str]:
        if f"{prefix}.qweight" in param_meta or f"{prefix}.weight" in param_meta:
            return [prefix]
        proj_name = prefix.rsplit(".", 1)[-1]
        fused_map: dict[str, list[str]] = {
            "gate_up_proj": ["gate_proj", "up_proj"],
            **self.packed_modules_mapping,
        }
        if proj_name in fused_map:
            stem = prefix[: -len(proj_name)]
            return [f"{stem}{shard_name}" for shard_name in fused_map[proj_name]]
        return [prefix]

    def _build_wna16_linear_method(
        self,
        prefix: str,
        shard_prefixes: list[str],
        param_meta: dict[str, dict[str, Any]],
    ) -> LinearMethodBase:
        """Build a WNA16 (`input_dtype=None`) GPTQ linear method for quantized non-routed Linears."""
        raw_qcfg = self._hf_gptq_raw_cfg or {}
        dynamic_cfg = raw_qcfg.get("dynamic")
        dynamic_dict = dict(dynamic_cfg) if isinstance(dynamic_cfg, dict) else {}

        inferred_pairs: set[tuple[int, int]] = set()
        for sp in shard_prefixes:
            qw_info = param_meta.get(f"{sp}.qweight")
            sc_info = param_meta.get(f"{sp}.scales")
            qz_info = param_meta.get(f"{sp}.qzeros")
            if qw_info is None or sc_info is None or qz_info is None:
                raise ValueError(
                    f"Incomplete GPTQ safetensors metadata for {sp!r} (expected .qweight, .scales, .qzeros)"
                )
            qw_shape = qw_info["shape"]
            sc_shape = sc_info["shape"]
            qz_shape = qz_info["shape"]
            if (
                len(qw_shape) != 2
                or len(sc_shape) != 2
                or len(qz_shape) != 2
                or int(qw_shape[0]) <= 0
                or int(qw_shape[1]) <= 0
                or int(qz_shape[1]) <= 0
                or int(sc_shape[1]) != int(qw_shape[1])
                or (int(qz_shape[1]) * 32) % int(qw_shape[1]) != 0
            ):
                raise ValueError(
                    f"Invalid GPTQ tensor shapes for {sp!r}: qweight={qw_shape}, "
                    f"scales={sc_shape}, qzeros={qz_shape}"
                )
            n_out = int(qw_shape[1])
            qz_cols = int(qz_shape[1])
            qw_rows = int(qw_shape[0])
            bits = (qz_cols * 32) // n_out
            if bits not in SUPPORTED_W_BITS or (qw_rows * 32) % bits != 0:
                raise ValueError(
                    f"Unsupported GPTQ bitwidth/shape (bits={bits}, qweight={qw_shape}, qzeros={qz_shape}) on {sp!r}"
                )
            k_full = (qw_rows * 32) // bits
            if bits == 3 and (n_out % 32 != 0 or k_full % 32 != 0):
                raise ValueError(
                    f"3-bit GPTQ requires N={n_out} and K={k_full} divisible by 32 on {sp!r}"
                )
            num_groups = int(sc_shape[0])
            if num_groups <= 0 or k_full % num_groups != 0:
                raise ValueError(
                    f"Invalid GPTQ scales num_groups={num_groups} for K={k_full} on {sp!r}"
                )

            cfg_gs = raw_qcfg.get("group_size")
            if dynamic_dict:
                dummy_holder = type("_Holder", (), {"dynamic": dynamic_dict})()
                ov_gs = get_dynamic_override(dummy_holder, sp, "group_size", cfg_gs)
                if isinstance(ov_gs, int):
                    cfg_gs = ov_gs

            if num_groups == 1 and (cfg_gs in (None, -1) or k_full != int(cfg_gs)):
                group_size = -1
            else:
                group_size = k_full // num_groups
            inferred_pairs.add((bits, group_size))

        if len(inferred_pairs) != 1:
            raise ValueError(
                f"Fused linear module {prefix!r} has mismatched GPTQ (bits, group_size) across shards: "
                f"{sorted(inferred_pairs)}"
            )
        bits, group_size = next(iter(inferred_pairs))
        is_sym = bool(raw_qcfg.get("sym", True))
        if not is_sym:
            raise ValueError(
                f"DxMoE WNA16 non-routed GPTQ linear ({prefix!r}) requires sym=True, got sym=False"
            )
        if bits in (2, 3):
            if group_size != -1:
                raise ValueError(
                    f"DxMoE native W{bits}A16 non-routed GPTQ linear ({prefix!r}) requires "
                    f"channel-wise group_size=-1, got group_size={group_size}"
                )
            logger.info_once(
                "DxMoE dispatching quantized non-routed linear (e.g. %s) to native W%dA16 "
                "DxMoEWNA16LinearMethod (group_size=-1, activations strictly FP16/BF16).",
                prefix,
                bits,
            )
            return DxMoEWNA16LinearMethod(w_bits=bits, prefix=prefix)

        full_cfg = {
            "bits": bits,
            "group_size": group_size,
            "desc_act": False,
            "sym": True,
            "lm_head": False,
        }
        auto_gptq_cfg = AutoGPTQConfig(
            weight_bits=bits,
            group_size=group_size,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config=full_cfg,
        )
        linear_method = AutoGPTQLinearMethod(auto_gptq_cfg)
        # Strict DxMoE algorithm invariant: ONLY RoutedExperts undergo dynamic activation
        # scheduling (A4/A8/A16). Non-routed linears (especially `shared_experts`) MUST
        # keep activations in FP16/BF16 (`input_dtype=None`, i.e. WNA16).
        linear_method.input_dtype = None
        logger.info_once(
            "DxMoE dispatching quantized non-routed linear (e.g. %s) to W%dA16 "
            "AutoGPTQLinearMethod (group_size=%d, activations strictly FP16/BF16).",
            prefix,
            bits,
            group_size,
        )
        return linear_method

    def _validate_deployment_config(self) -> None:
        vllm_cfg = get_current_vllm_config_or_none()
        if vllm_cfg is not None:
            p_cfg = getattr(vllm_cfg, "parallel_config", None)
            if p_cfg is not None:
                tp = int(getattr(p_cfg, "tensor_parallel_size", 1))
                pp = int(getattr(p_cfg, "pipeline_parallel_size", 1))
                dp = int(getattr(p_cfg, "data_parallel_size", 1))
                pcp = int(getattr(p_cfg, "prefill_context_parallel_size", 1))
                use_ep = bool(getattr(p_cfg, "enable_expert_parallel", False))
                enable_eplb = bool(getattr(p_cfg, "enable_eplb", False))
                if (
                    tp != 1
                    or pp != 1
                    or dp != 1
                    or pcp != 1
                    or use_ep
                    or enable_eplb
                ):
                    raise ValueError(
                        "DxMoE requires single-GPU execution "
                        "(tp=pp=dp=ep=pcp=1, enable_expert_parallel=False, enable_eplb=False), "
                        f"got tp={tp}, pp={pp}, dp={dp}, pcp={pcp}, "
                        f"use_ep={use_ep}, enable_eplb={enable_eplb}."
                    )
            s_cfg = getattr(vllm_cfg, "scheduler_config", None)
            if s_cfg is not None and int(getattr(s_cfg, "max_num_seqs", 1)) != 1:
                raise ValueError(
                    "DxMoE vLLM backend supports single-request B=1 execution only "
                    f"(expected scheduler_config.max_num_seqs == 1, got "
                    f"{s_cfg.max_num_seqs}); pass max_num_seqs=1."
                )
            k_cfg = getattr(vllm_cfg, "kernel_config", None)
            if k_cfg is not None:
                moe_backend = getattr(k_cfg, "moe_backend", "auto")
                if moe_backend not in ("auto", "dxmoe"):
                    raise ValueError(
                        f"DxMoE quantization requires moe_backend in ('auto', 'dxmoe'), "
                        f"got {moe_backend!r}."
                    )

        if torch.cuda.is_available():
            cap = current_platform.get_device_capability()
            if cap is None or (int(cap[0]), int(cap[1])) not in _SUPPORTED_SM_CAPS:
                raise ValueError(
                    "DxMoE native CUDA operator supports SM80 and SM86 only, "
                    f"got device capability {cap!r}."
                )

    def _ensure_resolved(self) -> dict[int, _ParsedDxMoELayerSpec]:
        self._validate_deployment_config()
        if self._resolved_specs is not None:
            return self._resolved_specs

        model_dir = self._resolve_model_dir()
        meta: dict[str, Any]
        if "layers" in self.raw_config and "quantization" in self.raw_config:
            meta = self.raw_config
        else:
            if model_dir is None or not (model_dir / _DXMOE_META_FILENAME).is_file():
                raise FileNotFoundError(
                    f"DxMoE requires {_DXMOE_META_FILENAME} in model directory "
                    f"(resolved model_dir={model_dir!r})."
                )
            meta = json.loads(
                (model_dir / _DXMOE_META_FILENAME).read_text(encoding="utf-8")
            )

        if meta.get("format") != _DXMOE_META_FORMAT or meta.get("version") != 1:
            raise ValueError(
                f"Expected DxMoE metadata format={_DXMOE_META_FORMAT!r} version=1, "
                f"got format={meta.get('format')!r} version={meta.get('version')!r}"
            )

        q_meta = meta.get("quantization", {})
        if q_meta.get("group_size") != -1 or q_meta.get("sym") is not True:
            raise ValueError(
                f"vLLM native DxMoE backend requires symmetric channel-wise weights "
                f"(group_size=-1, sym=True); got group_size={q_meta.get('group_size')!r}, "
                f"sym={q_meta.get('sym')!r}."
            )

        schedule_mode = str(
            os.environ.get("VLLM_DXMOE_SCHEDULE_MODE")
            or self.raw_config.get("schedule_mode")
            or "dxmoe"
        )
        if schedule_mode not in SCHEDULE_MODE_CODE:
            raise ValueError(
                f"Unsupported DxMoE schedule_mode={schedule_mode!r}; "
                f"expected one of {tuple(SCHEDULE_MODE_CODE)}"
            )

        threshold_key = str(
            os.environ.get("VLLM_DXMOE_THRESHOLD_KEY")
            or self.raw_config.get("threshold_key")
            or "5.0"
        ).strip()
        thresholds_json_raw = os.environ.get(
            "VLLM_DXMOE_THRESHOLDS_JSON"
        ) or self.raw_config.get("thresholds_json")

        raw_pipelined = (
            os.environ.get("VLLM_DXMOE_PIPELINED")
            if os.environ.get("VLLM_DXMOE_PIPELINED") is not None
            else self.raw_config.get("pipelined", "auto")
        )
        pipelined_lookup = (
            raw_pipelined.strip().lower()
            if isinstance(raw_pipelined, str)
            else raw_pipelined
        )
        if pipelined_lookup not in PIPELINED_CODE:
            raise ValueError(f"Unsupported DxMoE pipelined setting: {raw_pipelined!r}")
        pipelined_code = PIPELINED_CODE[pipelined_lookup]

        cfg_use_omega = self.raw_config.get("use_omega", True)
        use_omega = bool(_parse_bool_env("VLLM_DXMOE_USE_OMEGA", bool(cfg_use_omega)))
        sink_tokens = int(
            os.environ.get("VLLM_DXMOE_SINK_TOKENS")
            if os.environ.get("VLLM_DXMOE_SINK_TOKENS") is not None
            else self.raw_config.get("sink_tokens", 4)
        )
        local_window = int(
            os.environ.get("VLLM_DXMOE_LOCAL_WINDOW")
            if os.environ.get("VLLM_DXMOE_LOCAL_WINDOW") is not None
            else self.raw_config.get("local_window", 32)
        )
        if sink_tokens < 0 or local_window < 0:
            raise ValueError(
                f"sink_tokens and local_window must be non-negative, got {(sink_tokens, local_window)}"
            )
        raw_coupling = _parse_bool_env(
            "VLLM_DXMOE_APPLY_COUPLING", self.raw_config.get("apply_coupling")
        )

        had_meta = meta.get("hadamard", {})
        had_mode = had_meta.get("mode", "none")
        if had_mode not in ("block128", "none"):
            raise ValueError(f"Unsupported hadamard.mode={had_mode!r}")
        if had_mode == "block128" and had_meta.get("block_size") != 128:
            raise ValueError(
                f"Native DxMoE policy kernel requires hadamard.block_size==128, "
                f"got {had_meta.get('block_size')!r}"
            )

        layers_dict = meta.get("layers", {})
        expected_layer_ids = {int(k) for k in layers_dict}
        signs_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if had_mode == "block128":
            signs_fname = str(had_meta.get("signs_file") or _DXMOE_SIGNS_FILENAME)
            explicit_signs = os.environ.get(
                "VLLM_DXMOE_SIGNS_PATH"
            ) or self.raw_config.get("signs_path")
            if explicit_signs:
                signs_path = Path(str(explicit_signs))
            elif model_dir is not None:
                signs_path = model_dir / signs_fname
            else:
                raise FileNotFoundError(
                    "Cannot locate dxmoe_signs.safetensors without model_dir or "
                    "VLLM_DXMOE_SIGNS_PATH"
                )
            signs_by_layer = _load_signs_from_safetensors(signs_path, expected_layer_ids)
            self.shared_mid_signs = _load_shared_mid_signs(signs_path)
        else:
            self.shared_mid_signs = {}
        self.hadamard_mode = had_mode
        self.signs_by_layer = signs_by_layer

        # First pass: validate per-layer geometry, static w_bits, and finite non-negative beta
        validated_layers: dict[int, tuple[int, int, int, int, tuple[int, ...], tuple[int, ...], tuple[float, ...], tuple[float, ...]]] = {}
        for l_str, l_meta in layers_dict.items():
            l_idx = int(l_str)
            shared_ffn_rotated = l_meta.get("shared_ffn_rotated", False)
            if type(shared_ffn_rotated) is not bool:
                raise ValueError(
                    f"Layer {l_idx} shared_ffn_rotated must be a boolean, "
                    f"got {shared_ffn_rotated!r}"
                )
            if shared_ffn_rotated and had_mode != "block128":
                raise ValueError(
                    f"Layer {l_idx} shared_ffn_rotated requires hadamard.mode=block128"
                )
            self.shared_ffn_rotated_layers[l_idx] = shared_ffn_rotated
            hidden_size = int(l_meta["hidden_size"])
            inter_size = int(l_meta["moe_intermediate_size"])
            num_experts = int(l_meta["num_experts"])
            top_k = int(l_meta["top_k"])
            experts_meta = l_meta["experts"]

            w_bits_entry: list[int] = []
            w_bits_down: list[int] = []
            beta_entry: list[float] = []
            beta_down: list[float] = []
            for e_idx in range(num_experts):
                ed = experts_meta[str(e_idx)]
                bw_g = int(ed["gate"]["w_bits"])
                bw_u = int(ed["up"]["w_bits"])
                bw_e = int(ed["entry"]["w_bits"])
                bw_d = int(ed["down"]["w_bits"])
                if bw_g != bw_u or bw_g != bw_e:
                    raise ValueError(
                        f"Layer {l_idx} expert {e_idx} violates gate==up==entry "
                        f"coupling: ({bw_g}, {bw_u}, {bw_e})"
                    )
                if bw_e not in SUPPORTED_W_BITS or bw_d not in SUPPORTED_W_BITS:
                    raise ValueError(
                        f"Native DxMoE kernel supports W2/W3/W4/W8 static weights only; "
                        f"layer {l_idx} expert {e_idx} has entry=W{bw_e}, down=W{bw_d}"
                    )
                w_bits_entry.append(bw_e)
                w_bits_down.append(bw_d)
                bg = float(ed["gate"]["beta"])
                bu = float(ed["up"]["beta"])
                be = float(ed["entry"]["beta"])
                bd = float(ed["down"]["beta"])
                if not (
                    math.isfinite(bg)
                    and bg >= 0.0
                    and math.isfinite(bu)
                    and bu >= 0.0
                    and math.isfinite(be)
                    and be >= 0.0
                    and math.isfinite(bd)
                    and bd >= 0.0
                ):
                    raise ValueError(
                        f"Layer {l_idx} expert {e_idx} beta must be finite and "
                        f"non-negative, got entry={be!r}, down={bd!r}"
                    )
                beta_entry.append(be)
                beta_down.append(bd)

            s_in_cpu, s_mid_cpu = signs_by_layer.get(l_idx, (None, None))
            if s_in_cpu is not None and s_in_cpu.numel() != hidden_size:
                raise ValueError(
                    f"Layer {l_idx} s_in length {s_in_cpu.numel()} != {hidden_size}"
                )
            if s_mid_cpu is not None and s_mid_cpu.numel() != inter_size:
                raise ValueError(
                    f"Layer {l_idx} s_mid length {s_mid_cpu.numel()} != {inter_size}"
                )
            validated_layers[l_idx] = (
                hidden_size,
                inter_size,
                num_experts,
                top_k,
                tuple(w_bits_entry),
                tuple(w_bits_down),
                tuple(beta_entry),
                tuple(beta_down),
            )

        weight_fp = _compute_dxmoe_weight_fingerprint(
            meta, signs_by_layer, self.shared_mid_signs
        )

        # Three-tier threshold resolution:
        #   1) Explicit VLLM_DXMOE_THRESHOLDS_JSON / raw_config['thresholds_json']
        #   2) <model_dir>/thresholds/*.json auto-discovery (fail-closed on >= 2 matches)
        #   3) Fallback to dxmoe_meta.json embedded layers[l]['thresholds'][threshold_key]
        resolved_ext_tau: dict[int, tuple[tuple[float, float], tuple[float, float]]] | None = None
        dir_candidates: list[str] = []
        if thresholds_json_raw:
            ext_doc = json.loads(Path(str(thresholds_json_raw)).read_text(encoding="utf-8"))
            resolved_ext_tau = _extract_tau_from_external_doc(
                ext_doc,
                threshold_key=threshold_key,
                expected_layer_ids=expected_layer_ids,
                expected_weight_fp=weight_fp,
            )
        elif model_dir is not None and (model_dir / _DXMOE_THRESHOLDS_SUBDIR).is_dir():
            t_dir = model_dir / _DXMOE_THRESHOLDS_SUBDIR
            matched_files: list[tuple[Path, dict[str, Any], dict[int, tuple[tuple[float, float], tuple[float, float]]]]] = []
            for cand in sorted(p for p in t_dir.glob("*.json") if p.is_file()):
                dir_candidates.append(cand.name)
                cand_doc = json.loads(cand.read_text(encoding="utf-8"))
                if not isinstance(cand_doc, dict):
                    raise ValueError(f"Thresholds JSON {cand} must be a JSON object")
                # Always validate v2 integrity/fingerprint inside <model_dir>/thresholds/
                if cand_doc.get("version") == 2:
                    cand_tau = _extract_tau_from_external_doc(
                        cand_doc,
                        threshold_key=threshold_key,
                        expected_layer_ids=expected_layer_ids,
                        expected_weight_fp=weight_fp,
                    )
                else:
                    cand_tau = {}
                if _matches_threshold_file_key(cand, cand_doc, threshold_key):
                    if not cand_tau:
                        cand_tau = _extract_tau_from_external_doc(
                            cand_doc,
                            threshold_key=threshold_key,
                            expected_layer_ids=expected_layer_ids,
                            expected_weight_fp=weight_fp,
                        )
                    matched_files.append((cand, cand_doc, cand_tau))

            if len(matched_files) > 1:
                cand_names = [m[0].name for m in matched_files]
                raise ValueError(
                    f"Ambiguous DxMoE threshold_key={threshold_key!r} matched "
                    f"{len(matched_files)} files in {t_dir}: {cand_names}; "
                    f"set VLLM_DXMOE_THRESHOLDS_JSON or specify the exact filename stem."
                )
            if len(matched_files) == 1:
                resolved_ext_tau = matched_files[0][2]

        specs: dict[int, _ParsedDxMoELayerSpec] = {}
        for l_str, l_meta in layers_dict.items():
            l_idx = int(l_str)
            (
                hidden_size,
                inter_size,
                num_experts,
                top_k,
                w_bits_entry_t,
                w_bits_down_t,
                beta_entry_t,
                beta_down_t,
            ) = validated_layers[l_idx]

            has_w8_entry = any(b == 8 for b in w_bits_entry_t)
            has_w8_down = any(b == 8 for b in w_bits_down_t)
            if raw_coupling is not None:
                coup_entry = raw_coupling
                coup_down = raw_coupling
            else:
                coup_entry = schedule_mode in ("dxmoe", "static_a4")
                coup_down = schedule_mode in ("dxmoe", "static_a4")

            if resolved_ext_tau is not None:
                tau_entry, tau_down = resolved_ext_tau[l_idx]
            else:
                t_map = l_meta.get("thresholds", {})
                if not isinstance(t_map, dict):
                    raise ValueError(f"Layer {l_idx} thresholds mapping must be a dict")
                if threshold_key in t_map:
                    tdoc = t_map[threshold_key]
                    raw_te = tdoc.get("tau_entry", tdoc.get("tau"))
                    raw_td = tdoc.get("tau_down", tdoc.get("tau"))
                    if (
                        raw_te is None
                        or raw_td is None
                        or len(raw_te) != 2
                        or len(raw_td) != 2
                    ):
                        raise ValueError(
                            f"Layer {l_idx} threshold_key={threshold_key!r} missing "
                            "2-element tau_entry/tau_down"
                        )
                    tau_entry = _validate_tau_pair(raw_te, l_idx, "tau_entry")
                    tau_down = _validate_tau_pair(raw_td, l_idx, "tau_down")
                elif schedule_mode == "dxmoe":
                    raise ValueError(
                        f"schedule_mode='dxmoe' requires calibrated thresholds under "
                        f"threshold_key={threshold_key!r} for layer {l_idx}; "
                        f"thresholds/ candidates={dir_candidates}, "
                        f"available keys: {sorted(t_map)}"
                    )
                else:
                    tau_entry = (0.0, 0.0)
                    tau_down = (0.0, 0.0)

            if raw_coupling is False:
                if schedule_mode == "static_a4" and (has_w8_entry or has_w8_down):
                    raise ValueError(
                        f"Layer {l_idx} contains W8 experts, which cannot execute "
                        "schedule_mode='static_a4' with apply_coupling=False "
                        "(W8A4 is forbidden by the native DxMoE kernel)."
                    )
                if schedule_mode == "dxmoe" and (
                    (has_w8_entry and tau_entry[0] > 0.0)
                    or (has_w8_down and tau_down[0] > 0.0)
                ):
                    raise ValueError(
                        f"Layer {l_idx} contains W8 experts with positive A4 threshold "
                        f"(tau_entry={tau_entry}, tau_down={tau_down}); "
                        "apply_coupling=False is forbidden under schedule_mode='dxmoe' "
                        "because it can select W8A4."
                    )

            s_in_cpu, s_mid_cpu = signs_by_layer.get(l_idx, (None, None))

            specs[l_idx] = _ParsedDxMoELayerSpec(
                layer_idx=l_idx,
                hidden_size=hidden_size,
                inter_size=inter_size,
                num_experts=num_experts,
                top_k=top_k,
                w_bits_entry=w_bits_entry_t,
                w_bits_down=w_bits_down_t,
                beta_entry=beta_entry_t,
                beta_down=beta_down_t,
                tau_entry=tau_entry,
                tau_down=tau_down,
                s_in_cpu=s_in_cpu,
                s_mid_cpu=s_mid_cpu,
                schedule_mode=schedule_mode,
                apply_coupling_entry=coup_entry,
                apply_coupling_down=coup_down,
                use_omega=use_omega,
                sink_tokens=sink_tokens,
                local_window=local_window,
                pipelined_code=pipelined_code,
            )

        self._resolved_specs = specs
        logger.info_once(
            "Initialized DxMoE quantization config for %d MoE layers "
            "(schedule_mode=%s, threshold_key=%s, hadamard=%s)",
            len(specs),
            schedule_mode,
            threshold_key,
            had_mode,
        )
        return specs

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            param_meta = self._ensure_checkpoint_metadata()
            if not param_meta:
                return UnquantizedLinearMethod()
            shard_prefixes = self._resolve_linear_shard_prefixes(prefix, param_meta)
            has_qweight = [f"{sp}.qweight" in param_meta for sp in shard_prefixes]
            if any(has_qweight):
                if not all(has_qweight):
                    raise ValueError(
                        f"Detected mixed GPTQ/float shards for fused LinearBase {prefix!r}: "
                        f"{shard_prefixes}"
                    )
                return self._build_wna16_linear_method(prefix, shard_prefixes, param_meta)
            return UnquantizedLinearMethod()

        if isinstance(layer, RoutedExperts):
            specs = self._ensure_resolved()
            param_meta = self._ensure_checkpoint_metadata()
            m = _LAYER_IDX_RE.search(prefix)
            if m is None:
                raise ValueError(
                    f"DxMoE cannot parse layer index from RoutedExperts prefix {prefix!r}"
                )
            l_idx = int(m.group(1))
            spec = specs.get(l_idx)
            if spec is None:
                raise ValueError(
                    f"DxMoE metadata ({_DXMOE_META_FILENAME}) is missing RoutedExperts "
                    f"layer {l_idx} (prefix={prefix!r}); available layers: {sorted(specs)}"
                )
            ckpt_experts_prefix = prefix.removesuffix(".routed_experts")
            if param_meta:
                prefix_dot = f"{ckpt_experts_prefix}."
                has_routed_qw = any(
                    k.startswith(prefix_dot) and k.endswith(".qweight")
                    for k in param_meta
                )
                has_routed_fp = any(
                    k.startswith(prefix_dot)
                    and (
                        k.endswith(".weight")
                        or k == f"{ckpt_experts_prefix}.gate_up_proj"
                        or k == f"{ckpt_experts_prefix}.down_proj"
                    )
                    for k in param_meta
                )
                if has_routed_qw and has_routed_fp:
                    raise ValueError(
                        f"RoutedExperts {ckpt_experts_prefix!r} mixes GPTQ .qweight and float .weight tensors"
                    )
                checkpoint_is_gptq = has_routed_qw
            else:
                q_doc = self.raw_config.get("quantization", {})
                checkpoint_is_gptq = (
                    self.raw_config.get("checkpoint_format") == "gptq"
                    or (isinstance(q_doc, dict) and q_doc.get("checkpoint_format") == "gptq")
                )
            return DxMoEMoEMethod(
                spec, layer.moe_config, checkpoint_is_gptq=checkpoint_is_gptq
            )
        return None


class DxMoEWNA16LinearMethod(LinearMethodBase):
    """Native packed W2A16/W3A16 CUDA LinearMethod for quantized non-routed Linears (`shared_experts`)."""

    def __init__(self, *, w_bits: int, prefix: str = "") -> None:
        if w_bits not in (2, 3):
            raise ValueError(
                f"DxMoEWNA16LinearMethod supports w_bits in (2, 3), got {w_bits!r}"
            )
        self.w_bits = int(w_bits)
        self.prefix = prefix
        self.input_dtype = None
        self._processed = False
        self._max_num_tokens = 2048
        self._loaded_qweight_shards: set[int] = set()
        self._loaded_scales_shards: set[int] = set()
        self._loaded_qzeros_shards: set[int] = set()

    def _resolve_shard_span(
        self,
        part_sizes: tuple[int, ...],
        loaded_shard_id: int | tuple[int, ...] | None,
    ) -> tuple[int, int, tuple[int, ...]]:
        if loaded_shard_id is None:
            return 0, sum(part_sizes), tuple(range(len(part_sizes)))
        if isinstance(loaded_shard_id, int):
            if not 0 <= loaded_shard_id < len(part_sizes):
                raise ValueError(
                    f"Invalid shard_id={loaded_shard_id} for {self.prefix!r} (partitions={len(part_sizes)})"
                )
            return (
                sum(part_sizes[:loaded_shard_id]),
                part_sizes[loaded_shard_id],
                (loaded_shard_id,),
            )
        if isinstance(loaded_shard_id, tuple) and loaded_shard_id:
            lo, hi = loaded_shard_id[0], loaded_shard_id[-1]
            if (
                lo < 0
                or hi >= len(part_sizes)
                or list(loaded_shard_id) != list(range(lo, hi + 1))
            ):
                raise ValueError(
                    f"Invalid tuple shard_id={loaded_shard_id!r} for {self.prefix!r}"
                )
            return (
                sum(part_sizes[:lo]),
                sum(part_sizes[lo : hi + 1]),
                tuple(range(lo, hi + 1)),
            )
        raise ValueError(f"Unsupported loaded_shard_id={loaded_shard_id!r} for {self.prefix!r}")

    def _make_param_loader(self, layer: torch.nn.Module, kind: str):
        w_bits = self.w_bits

        @torch.no_grad()
        def _loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int | tuple[int, ...] | None = None,
        ) -> None:
            k_in = int(layer._dxmoe_wna16_k)
            part_sizes: tuple[int, ...] = layer._dxmoe_wna16_part_sizes
            n_start, n_len, shard_indices = self._resolve_shard_span(
                part_sizes, loaded_shard_id
            )
            if kind == "qweight":
                exp_shape = ((k_in * w_bits) // 32, n_len)
                if loaded_weight.dtype != torch.int32 or tuple(loaded_weight.shape) != exp_shape:
                    raise ValueError(
                        f"{self.prefix!r} qweight shard expected int32 {exp_shape}, "
                        f"got {loaded_weight.dtype} {tuple(loaded_weight.shape)}"
                    )
                param.data[:, n_start : n_start + n_len].copy_(loaded_weight)
                self._loaded_qweight_shards.update(shard_indices)
            elif kind == "scales":
                if not loaded_weight.is_floating_point() or tuple(loaded_weight.shape) not in (
                    (1, n_len),
                    (n_len,),
                ):
                    raise ValueError(
                        f"{self.prefix!r} scales shard expected float (1, {n_len}), "
                        f"got {loaded_weight.dtype} {tuple(loaded_weight.shape)}"
                    )
                param.data[:, n_start : n_start + n_len].copy_(
                    loaded_weight.reshape(1, n_len).to(
                        device=param.device, dtype=torch.float16
                    )
                )
                self._loaded_scales_shards.update(shard_indices)
            elif kind == "qzeros":
                qz_start = (n_start * w_bits) // 32
                qz_len = (n_len * w_bits) // 32
                exp_shape = (1, qz_len)
                if loaded_weight.dtype != torch.int32 or tuple(loaded_weight.shape) != exp_shape:
                    raise ValueError(
                        f"{self.prefix!r} qzeros shard expected int32 {exp_shape}, "
                        f"got {loaded_weight.dtype} {tuple(loaded_weight.shape)}"
                    )
                param.data[:, qz_start : qz_start + qz_len].copy_(loaded_weight)
                self._loaded_qzeros_shards.update(shard_indices)
            else:
                assert kind == "g_idx"
                if loaded_weight.dtype != torch.int32 or tuple(loaded_weight.shape) != (k_in,):
                    raise ValueError(
                        f"{self.prefix!r} g_idx expected int32 ({k_in},), "
                        f"got {loaded_weight.dtype} {tuple(loaded_weight.shape)}"
                    )
                if not bool((loaded_weight == 0).all()):
                    raise ValueError(
                        f"{self.prefix!r} channel-wise W{w_bits}A16 requires all-zero g_idx"
                    )
                param.data.copy_(loaded_weight)

        return _loader

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(int(s) for s in output_partition_sizes)
        if input_size != input_size_per_partition or output_size != output_size_per_partition:
            raise ValueError(
                f"DxMoEWNA16LinearMethod ({self.prefix!r}) requires single-GPU TP=1, "
                f"got input={input_size_per_partition}/{input_size}, "
                f"output={output_size_per_partition}/{output_size}"
            )
        elem_group = 32 if self.w_bits == 3 else (32 // self.w_bits)
        if input_size <= 0 or input_size % elem_group != 0:
            raise ValueError(
                f"DxMoEWNA16LinearMethod ({self.prefix!r}) W{self.w_bits} requires K={input_size} "
                f"divisible by {elem_group}"
            )
        part_tuple = tuple(int(s) for s in output_partition_sizes)
        if not part_tuple or any(s <= 0 or s % elem_group != 0 for s in part_tuple):
            raise ValueError(
                f"DxMoEWNA16LinearMethod ({self.prefix!r}) W{self.w_bits} requires output "
                f"partitions {part_tuple} divisible by {elem_group}"
            )

        vllm_cfg = get_current_vllm_config_or_none()
        if vllm_cfg is not None and vllm_cfg.scheduler_config is not None:
            self._max_num_tokens = max(
                int(getattr(vllm_cfg.scheduler_config, "max_num_batched_tokens", 2048)),
                1,
            )

        layer._dxmoe_wna16_k = int(input_size)
        layer._dxmoe_wna16_n = int(output_size)
        layer._dxmoe_wna16_part_sizes = part_tuple

        qw_init = torch.zeros(
            ((input_size * self.w_bits) // 32, output_size), dtype=torch.int32
        )
        sc_init = torch.ones((1, output_size), dtype=torch.float16)
        qz_cols = (output_size * self.w_bits) // 32
        if self.w_bits == 3:
            qz_init = (
                torch.tensor(GPTQ_SYM_QZEROS_W3, dtype=torch.int32)
                .repeat(output_size // 32)
                .reshape(1, qz_cols)
                .contiguous()
            )
        else:
            qz_init = torch.full(
                (1, qz_cols), GPTQ_SYM_QZEROS[self.w_bits], dtype=torch.int32
            )
        gidx_init = torch.zeros((input_size,), dtype=torch.int32)

        for name, init_t, kind in (
            ("qweight", qw_init, "qweight"),
            ("scales", sc_init, "scales"),
            ("qzeros", qz_init, "qzeros"),
            ("g_idx", gidx_init, "g_idx"),
        ):
            p = torch.nn.Parameter(init_t, requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)
            p.weight_loader = self._make_param_loader(layer, kind)  # type: ignore[attr-defined]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._processed or is_weights_pre_processed():
            return
        self._processed = True

        vllm_cfg = get_current_vllm_config_or_none()
        is_dummy_load = (
            vllm_cfg is not None
            and vllm_cfg.load_config is not None
            and vllm_cfg.load_config.load_format == "dummy"
        )
        part_sizes: tuple[int, ...] = layer._dxmoe_wna16_part_sizes
        expected_shards = set(range(len(part_sizes)))
        if not is_dummy_load:
            for label, loaded_set in (
                ("qweight", self._loaded_qweight_shards),
                ("scales", self._loaded_scales_shards),
                ("qzeros", self._loaded_qzeros_shards),
            ):
                missing = sorted(expected_shards - loaded_set)
                if missing:
                    raise ValueError(
                        f"DxMoEWNA16LinearMethod ({self.prefix!r}) missing GPTQ {label} shards: {missing}"
                    )

        k_orig = int(layer._dxmoe_wna16_k)
        n_orig = int(layer._dxmoe_wna16_n)
        device = layer.qweight.device

        validate_gptq_qzeros_shard(
            layer.qzeros.data, n_expected=n_orig, w_bits=self.w_bits
        )
        s_f16 = validate_gptq_scales_shard(
            layer.scales.data, n_expected=n_orig, target_device=device
        )
        codes = repack_gptq_qweight_shard(
            layer.qweight.data,
            k_expected=k_orig,
            n_expected=n_orig,
            w_bits=self.w_bits,
            target_device=device,
        )

        k_pad = round_up(k_orig, 128)
        n_pad = round_up(n_orig, 256)
        row_bytes_orig = dxmoe_packed_row_bytes(k_orig, self.w_bits)
        row_bytes_pad = dxmoe_packed_row_bytes(k_pad, self.w_bits)
        if k_pad == k_orig and n_pad == n_orig:
            codes_pad = codes.contiguous()
            scales_pad = s_f16.contiguous()
        else:
            if self.w_bits == 2:
                codes_pad = torch.full(
                    (n_pad, row_bytes_pad), 0xAA, dtype=torch.uint8, device=device
                )
            else:
                zero_pat = torch.tensor(
                    [0x24, 0x92, 0x49], dtype=torch.uint8, device=device
                )
                codes_pad = (
                    zero_pat.repeat(row_bytes_pad // 3)
                    .unsqueeze(0)
                    .expand(n_pad, row_bytes_pad)
                    .clone()
                )
            codes_pad[:n_orig, :row_bytes_orig].copy_(codes)
            scales_pad = torch.ones((n_pad,), dtype=torch.float16, device=device)
            scales_pad[:n_orig].copy_(s_f16)

        for name in ("qweight", "scales", "qzeros", "g_idx"):
            if hasattr(layer, name):
                delattr(layer, name)

        layer.register_parameter(
            "_dxmoe_wna16_codes",
            torch.nn.Parameter(codes_pad, requires_grad=False),
        )
        layer.register_parameter(
            "_dxmoe_wna16_scales",
            torch.nn.Parameter(scales_pad, requires_grad=False),
        )
        layer._dxmoe_wna16_k_pad = int(k_pad)
        layer._dxmoe_wna16_n_pad = int(n_pad)
        layer.register_buffer(
            "_dxmoe_wna16_w_bits_dev",
            torch.tensor([self.w_bits], dtype=torch.int32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "_dxmoe_wna16_w_ptrs_dev",
            torch.tensor([int(codes_pad.data_ptr())], dtype=torch.int64, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "_dxmoe_wna16_sw_ptrs_dev",
            torch.tensor([int(scales_pad.data_ptr())], dtype=torch.int64, device=device),
            persistent=False,
        )

        if device.type == "cuda":
            bundle = get_dxmoe_lib_bundle(device)
            get_dxmoe_wna16_linear_workspace(
                num_tokens=1,
                max_num_tokens=self._max_num_tokens,
                k_pad=k_pad,
                n_pad=n_pad,
                device=device,
                bundle=bundle,
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k_orig = int(layer._dxmoe_wna16_k)
        n_orig = int(layer._dxmoe_wna16_n)
        k_pad = int(layer._dxmoe_wna16_k_pad)
        n_pad = int(layer._dxmoe_wna16_n_pad)
        if x.shape[-1] != k_orig:
            raise ValueError(
                f"DxMoEWNA16LinearMethod ({self.prefix!r}) expected input dim {k_orig}, "
                f"got {tuple(x.shape)}"
            )
        codes_ptr = int(layer._dxmoe_wna16_codes.data_ptr())
        scales_ptr = int(layer._dxmoe_wna16_scales.data_ptr())
        # Refresh device pointer table if module parameters were relocated after loading.
        if getattr(layer, "_dxmoe_wna16_cached_ptrs", None) != (codes_ptr, scales_ptr):
            dev = layer._dxmoe_wna16_codes.device
            layer._dxmoe_wna16_w_ptrs_dev = torch.tensor(
                [codes_ptr], dtype=torch.int64, device=dev
            )
            layer._dxmoe_wna16_sw_ptrs_dev = torch.tensor(
                [scales_ptr], dtype=torch.int64, device=dev
            )
            layer._dxmoe_wna16_cached_ptrs = (codes_ptr, scales_ptr)

        x_2d = x.reshape(-1, k_orig)
        out_2d = run_dxmoe_wna16_linear(
            x_2d,
            w_bits_dev=layer._dxmoe_wna16_w_bits_dev,
            w_ptrs_dev=layer._dxmoe_wna16_w_ptrs_dev,
            sw_ptrs_dev=layer._dxmoe_wna16_sw_ptrs_dev,
            k_orig=k_orig,
            n_orig=n_orig,
            k_pad=k_pad,
            n_pad=n_pad,
            max_num_tokens=self._max_num_tokens,
        )
        out = out_2d.reshape(*x.shape[:-1], n_orig)
        if bias is not None:
            out = out + bias
        return out


class DxMoEMoEMethod(FusedMoEMethodBase):
    """Zero-peak streaming FusedMoE method for GPTQ-packed or float DxMoE W2/W3/W4/W8 weights."""

    def __init__(
        self,
        layer_spec: _ParsedDxMoELayerSpec,
        moe: FusedMoEConfig,
        *,
        checkpoint_is_gptq: bool = False,
    ) -> None:
        super().__init__(moe)
        self.layer_spec = layer_spec
        self.checkpoint_is_gptq = bool(checkpoint_is_gptq)
        self.experts_cls = DxMoEExperts
        self._dxmoe_processed = False
        self._w13_offsets: tuple[int, ...] = ()
        self._w2_offsets: tuple[int, ...] = ()
        self._loaded_qweight: set[tuple[int, str]] = set()
        self._loaded_scales: set[tuple[int, str]] = set()
        self._loaded_qzeros: set[tuple[int, str]] = set()
        self._loaded_float_shards: set[tuple[int, str]] = set()

    @property
    def supports_eplb(self) -> bool:
        return False

    def _validate_layer_contract(self, layer: RoutedExperts) -> None:
        spec = self.layer_spec
        if self.moe.moe_backend not in ("auto", "dxmoe"):
            raise ValueError(
                f"DxMoEMoEMethod requires moe_backend in ('auto', 'dxmoe'), "
                f"got {self.moe.moe_backend!r}"
            )
        p_cfg = self.moe.moe_parallel_config
        if (
            self.moe.tp_size != 1
            or self.moe.dp_size != 1
            or self.moe.ep_size != 1
            or self.moe.use_ep
            or p_cfg.enable_eplb
        ):
            raise ValueError(
                "DxMoEMoEMethod requires single-GPU TP=1, DP=1, EP=1 execution "
                "with enable_eplb=False."
            )
        if self.moe.has_bias:
            raise ValueError("DxMoEMoEMethod does not support biased MoE layers.")
        if (
            not self.moe.is_act_and_mul
            or self.moe.activation != MoEActivation.SILU
            or self.moe.swiglu_limit is not None
        ):
            raise ValueError(
                "DxMoEMoEMethod requires unclamped gated SiLU-and-mul (SwiGLU) MoE layers, "
                f"got activation={self.moe.activation!r}, "
                f"swiglu_limit={self.moe.swiglu_limit!r}."
            )
        if (
            self.moe.experts_per_token != spec.top_k
            or getattr(layer, "top_k", spec.top_k) != spec.top_k
        ):
            raise ValueError(
                f"Layer {spec.layer_idx} top_k mismatch between vLLM RoutedExperts "
                f"(experts_per_token={self.moe.experts_per_token}, "
                f"layer.top_k={getattr(layer, 'top_k', None)}) and "
                f"dxmoe_meta.json (top_k={spec.top_k})."
            )
        scoring_func = getattr(layer, "scoring_func", "softmax")
        use_grouped_topk = bool(getattr(layer, "use_grouped_topk", False))
        num_expert_group = getattr(layer, "num_expert_group", None)
        topk_group = getattr(layer, "topk_group", None)
        # DeepseekV2MoE always sets `use_grouped_topk=True`, which `router_factory`
        # degenerates to standard ungrouped `FusedTopKRouter` when
        # `num_expert_group <= 1 and topk_group <= 1` (e.g. DeepSeek-V2-Lite).
        nondegenerate_grouped_topk = (
            (num_expert_group is not None and int(num_expert_group) > 1)
            or (topk_group is not None and int(topk_group) > 1)
            or (
                use_grouped_topk
                and (num_expert_group is None or topk_group is None)
            )
        )
        e_bias = getattr(layer, "e_score_correction_bias", None)
        custom_fn = getattr(layer, "custom_routing_function", None)
        apply_on_in = bool(getattr(layer, "apply_router_weight_on_input", False))
        if (
            scoring_func != "softmax"
            or nondegenerate_grouped_topk
            or e_bias is not None
            or custom_fn is not None
            or apply_on_in
            or self.moe.has_hash_routing
            or self.moe.routing_method not in _SUPPORTED_ROUTING_METHODS
        ):
            raise ValueError(
                f"Layer {spec.layer_idx} uses unsupported MoE routing for DxMoE: "
                "only standard unbiased, ungrouped global Softmax TopK is supported "
                f"(got scoring_func={scoring_func!r}, use_grouped_topk={use_grouped_topk}, "
                f"num_expert_group={num_expert_group!r}, topk_group={topk_group!r}, "
                f"has_bias={e_bias is not None}, has_custom_router={custom_fn is not None}, "
                f"apply_router_weight_on_input={apply_on_in}, "
                f"routing_method={self.moe.routing_method!r})."
            )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        self._validate_layer_contract(layer)

        spec = self.layer_spec
        if (
            num_experts != spec.num_experts
            or hidden_size != spec.hidden_size
            or intermediate_size_per_partition != spec.inter_size
        ):
            raise ValueError(
                f"Layer {spec.layer_idx} shape mismatch between vLLM RoutedExperts "
                f"(E={num_experts}, H={hidden_size}, I={intermediate_size_per_partition}) "
                f"and dxmoe_meta.json (E={spec.num_experts}, H={spec.hidden_size}, "
                f"I={spec.inter_size})"
            )

        w13_up_dim = 2 * intermediate_size_per_partition
        unpadded_hidden = self.moe.hidden_dim_unpadded
        unpadded_inter = self.moe.intermediate_size_per_partition_unpadded
        assert unpadded_hidden is not None and unpadded_inter is not None

        probe_dev = torch.empty(0).device
        w13_codes, w13_scales, w13_w_bits, self._w13_offsets = (
            allocate_dxmoe_weight_buffers(
                num_experts,
                w13_up_dim,
                hidden_size,
                spec.w_bits_entry,
                device=probe_dev,
            )
        )
        w2_codes, w2_scales, w2_w_bits, self._w2_offsets = (
            allocate_dxmoe_weight_buffers(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                spec.w_bits_down,
                device=probe_dev,
            )
        )
        layer.register_buffer("_dxmoe_w13_codes", w13_codes, persistent=False)
        layer.register_buffer("_dxmoe_w13_scales", w13_scales, persistent=False)
        layer.register_buffer("_dxmoe_w13_w_bits", w13_w_bits, persistent=False)
        layer.register_buffer("_dxmoe_w2_codes", w2_codes, persistent=False)
        layer.register_buffer("_dxmoe_w2_scales", w2_scales, persistent=False)
        layer.register_buffer("_dxmoe_w2_w_bits", w2_w_bits, persistent=False)

        if self.checkpoint_is_gptq:
            self._register_gptq_proxy_params(
                layer,
                num_experts=num_experts,
                params_dtype=params_dtype,
                extra_weight_attrs=extra_weight_attrs,
            )
        else:
            self._register_float_proxy_params(
                layer,
                num_experts=num_experts,
                w13_up_dim=w13_up_dim,
                hidden_size=hidden_size,
                unpadded_inter=unpadded_inter,
                unpadded_hidden=unpadded_hidden,
                params_dtype=params_dtype,
                extra_weight_attrs=extra_weight_attrs,
            )

    def _register_gptq_proxy_params(
        self,
        layer: RoutedExperts,
        *,
        num_experts: int,
        params_dtype: torch.dtype,
        extra_weight_attrs: dict[str, Any],
    ) -> None:
        proxy_specs = (
            ("w13_qweight", torch.int32, "qweight"),
            ("w2_qweight", torch.int32, "qweight"),
            ("w13_scales", params_dtype, "scales"),
            ("w2_scales", params_dtype, "scales"),
            ("w13_qzeros", torch.int32, "qzeros"),
            ("w2_qzeros", torch.int32, "qzeros"),
        )
        for param_name, dtype, kind in proxy_specs:
            p = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=dtype),
                requires_grad=False,
            )
            layer.register_parameter(param_name, p)
            set_weight_attrs(p, extra_weight_attrs)
            loader_fn = self._make_gptq_shard_loader(layer, kind)
            loader_fn.supports_moe_loading = True  # type: ignore[attr-defined]
            p.weight_loader = loader_fn  # type: ignore[attr-defined]

    def _register_float_proxy_params(
        self,
        layer: RoutedExperts,
        *,
        num_experts: int,
        w13_up_dim: int,
        hidden_size: int,
        unpadded_inter: int,
        unpadded_hidden: int,
        params_dtype: torch.dtype,
        extra_weight_attrs: dict[str, Any],
    ) -> None:
        w13_weight = torch.nn.Parameter(
            torch.empty((num_experts, w13_up_dim, 0), dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w13_weight.weight_loader_numel = num_experts * (2 * unpadded_inter) * unpadded_hidden
        w13_loader = self._make_float_shard_loader(layer)
        w13_loader.supports_moe_loading = True  # type: ignore[attr-defined]
        w13_weight.weight_loader = w13_loader  # type: ignore[attr-defined]

        w2_weight = torch.nn.Parameter(
            torch.empty((num_experts, hidden_size, 0), dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        w2_weight.weight_loader_numel = num_experts * unpadded_hidden * unpadded_inter
        w2_loader = self._make_float_shard_loader(layer)
        w2_loader.supports_moe_loading = True  # type: ignore[attr-defined]
        w2_weight.weight_loader = w2_loader  # type: ignore[attr-defined]

    def _make_gptq_shard_loader(self, layer: RoutedExperts, kind: str):
        spec = self.layer_spec

        @torch.no_grad()
        def _loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            local_e = layer._map_global_expert_id_to_local_expert_id(int(expert_id))
            if local_e == -1:
                return False if return_success else None
            if shard_id not in ("w1", "w2", "w3"):
                raise ValueError(f"Unexpected MoE shard_id={shard_id!r} for {weight_name!r}")
            if not 0 <= local_e < spec.num_experts:
                raise ValueError(
                    f"Layer {spec.layer_idx} expert_id={local_e} out of [0, {spec.num_experts})"
                )

            is_entry = shard_id in ("w1", "w3")
            w_bits = spec.w_bits_entry[local_e] if is_entry else spec.w_bits_down[local_e]
            k_exp = spec.hidden_size if is_entry else spec.inter_size
            n_exp = spec.inter_size if is_entry else spec.hidden_size
            codes_buf = layer._dxmoe_w13_codes if is_entry else layer._dxmoe_w2_codes
            scales_buf = layer._dxmoe_w13_scales if is_entry else layer._dxmoe_w2_scales
            target_dev = codes_buf.device

            if kind == "qweight":
                repacked = repack_gptq_qweight_shard(
                    loaded_weight,
                    k_expected=k_exp,
                    n_expected=n_exp,
                    w_bits=w_bits,
                    target_device=target_dev,
                )
                row_bytes = dxmoe_packed_row_bytes(k_exp, w_bits)
                shard_nbytes = n_exp * row_bytes
                base_off = (
                    self._w13_offsets[local_e] if is_entry else self._w2_offsets[local_e]
                )
                if shard_id == "w3":
                    base_off += shard_nbytes
                codes_buf[base_off : base_off + shard_nbytes].copy_(repacked.view(-1))
                self._loaded_qweight.add((local_e, shard_id))
            elif kind == "scales":
                s_f16 = validate_gptq_scales_shard(
                    loaded_weight,
                    n_expected=n_exp,
                    target_device=scales_buf.device,
                )
                if shard_id == "w1":
                    scales_buf[local_e, :n_exp].copy_(s_f16)
                elif shard_id == "w3":
                    scales_buf[local_e, n_exp : 2 * n_exp].copy_(s_f16)
                else:
                    scales_buf[local_e, :n_exp].copy_(s_f16)
                self._loaded_scales.add((local_e, shard_id))
            else:
                assert kind == "qzeros"
                validate_gptq_qzeros_shard(
                    loaded_weight,
                    n_expected=n_exp,
                    w_bits=w_bits,
                )
                self._loaded_qzeros.add((local_e, shard_id))

            return True if return_success else None

        return _loader

    def _make_float_shard_loader(self, layer: RoutedExperts):
        spec = self.layer_spec

        @torch.no_grad()
        def _loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            local_e = layer._map_global_expert_id_to_local_expert_id(int(expert_id))
            if local_e == -1:
                return False if return_success else None
            if shard_id not in ("w1", "w2", "w3"):
                raise ValueError(f"Unexpected MoE shard_id={shard_id!r} for {weight_name!r}")
            if not 0 <= local_e < spec.num_experts:
                raise ValueError(
                    f"Layer {spec.layer_idx} expert_id={local_e} out of [0, {spec.num_experts})"
                )

            is_entry = shard_id in ("w1", "w3")
            w_bits = spec.w_bits_entry[local_e] if is_entry else spec.w_bits_down[local_e]
            k_exp = spec.hidden_size if is_entry else spec.inter_size
            n_exp = spec.inter_size if is_entry else spec.hidden_size
            if loaded_weight.ndim != 2 or tuple(loaded_weight.shape) != (n_exp, k_exp):
                raise ValueError(
                    f"Layer {spec.layer_idx} expert {local_e} ({shard_id}) expected float shape "
                    f"({n_exp}, {k_exp}), got {tuple(loaded_weight.shape)}"
                )

            codes_buf = layer._dxmoe_w13_codes if is_entry else layer._dxmoe_w2_codes
            scales_buf = layer._dxmoe_w13_scales if is_entry else layer._dxmoe_w2_scales
            packed, s_f16 = pack_dxmoe_float_matrix(
                loaded_weight.to(device=codes_buf.device), w_bits
            )
            row_bytes = dxmoe_packed_row_bytes(k_exp, w_bits)
            shard_nbytes = n_exp * row_bytes
            base_off = (
                self._w13_offsets[local_e] if is_entry else self._w2_offsets[local_e]
            )
            if shard_id == "w3":
                base_off += shard_nbytes
            codes_buf[base_off : base_off + shard_nbytes].copy_(packed.view(-1))
            if shard_id == "w1":
                scales_buf[local_e, :n_exp].copy_(s_f16)
            elif shard_id == "w3":
                scales_buf[local_e, n_exp : 2 * n_exp].copy_(s_f16)
            else:
                scales_buf[local_e, :n_exp].copy_(s_f16)
            self._loaded_float_shards.add((local_e, shard_id))
            return True if return_success else None

        return _loader

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return self.moe_quant_config

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        super().process_weights_after_loading(layer)
        if self._dxmoe_processed or is_weights_pre_processed():
            return
        self._dxmoe_processed = True

        spec = self.layer_spec
        vllm_cfg = get_current_vllm_config_or_none()
        is_dummy_load = (
            vllm_cfg is not None
            and vllm_cfg.load_config is not None
            and vllm_cfg.load_config.load_format == "dummy"
        )
        expected_shards = {
            (e_idx, s_id)
            for e_idx in range(spec.num_experts)
            for s_id in ("w1", "w2", "w3")
        }
        if not is_dummy_load:
            if self.checkpoint_is_gptq:
                for label, loaded_set in (
                    ("qweight", self._loaded_qweight),
                    ("scales", self._loaded_scales),
                    ("qzeros", self._loaded_qzeros),
                ):
                    missing = sorted(expected_shards - loaded_set)
                    if missing:
                        raise ValueError(
                            f"Layer {spec.layer_idx} missing GPTQ {label} expert shards "
                            f"(sample: {missing[:4]})"
                        )
            else:
                missing = sorted(expected_shards - self._loaded_float_shards)
                if missing:
                    raise ValueError(
                        f"Layer {spec.layer_idx} missing float expert weight shards "
                        f"(sample: {missing[:4]})"
                    )

        ref_param = (
            getattr(layer, "w13_qweight", None)
            if self.checkpoint_is_gptq
            else getattr(layer, "w13_weight", None)
        )
        device = (
            ref_param.device
            if isinstance(ref_param, torch.Tensor)
            else layer._dxmoe_w13_codes.device
        )
        for buf_name in (
            "_dxmoe_w13_codes",
            "_dxmoe_w13_scales",
            "_dxmoe_w13_w_bits",
            "_dxmoe_w2_codes",
            "_dxmoe_w2_scales",
            "_dxmoe_w2_w_bits",
        ):
            buf = getattr(layer, buf_name)
            if buf.device != device:
                layer.register_buffer(buf_name, buf.to(device=device).contiguous(), persistent=False)

        w13_packed = finalize_dxmoe_weight_channel(
            num_experts=spec.num_experts,
            n=2 * spec.inter_size,
            k=spec.hidden_size,
            w_bits_list=spec.w_bits_entry,
            codes_arena=layer._dxmoe_w13_codes,
            scales=layer._dxmoe_w13_scales,
            w_bits_dev=layer._dxmoe_w13_w_bits,
            expert_offsets=self._w13_offsets,
        )
        w2_packed = finalize_dxmoe_weight_channel(
            num_experts=spec.num_experts,
            n=spec.hidden_size,
            k=spec.inter_size,
            w_bits_list=spec.w_bits_down,
            codes_arena=layer._dxmoe_w2_codes,
            scales=layer._dxmoe_w2_scales,
            w_bits_dev=layer._dxmoe_w2_w_bits,
            expert_offsets=self._w2_offsets,
        )
        layer.register_buffer("_dxmoe_w13_w_ptrs", w13_packed.w_ptrs_dev, persistent=False)
        layer.register_buffer("_dxmoe_w13_sw_ptrs", w13_packed.sw_ptrs_dev, persistent=False)
        layer.register_buffer("_dxmoe_w2_w_ptrs", w2_packed.w_ptrs_dev, persistent=False)
        layer.register_buffer("_dxmoe_w2_sw_ptrs", w2_packed.sw_ptrs_dev, persistent=False)

        s_in_dev = (
            spec.s_in_cpu.to(device=device, dtype=torch.int8).contiguous()
            if spec.s_in_cpu is not None
            else None
        )
        s_mid_dev = (
            spec.s_mid_cpu.to(device=device, dtype=torch.int8).contiguous()
            if spec.s_mid_cpu is not None
            else None
        )
        if s_in_dev is not None:
            layer.register_buffer("_dxmoe_s_in", s_in_dev, persistent=False)
        if s_mid_dev is not None:
            layer.register_buffer("_dxmoe_s_mid", s_mid_dev, persistent=False)

        beta_entry_dev = torch.tensor(
            spec.beta_entry, dtype=torch.float64, device=device
        )
        beta_down_dev = torch.tensor(
            spec.beta_down, dtype=torch.float64, device=device
        )
        layer.register_buffer("_dxmoe_beta_entry", beta_entry_dev, persistent=False)
        layer.register_buffer("_dxmoe_beta_down", beta_down_dev, persistent=False)

        if self.checkpoint_is_gptq:
            for pname in (
                "w13_qweight",
                "w2_qweight",
                "w13_scales",
                "w2_scales",
                "w13_qzeros",
                "w2_qzeros",
            ):
                if pname in layer._parameters:
                    del layer._parameters[pname]

        # Keep 3D [E, N, 0] placeholders for FusedMoEKernel.moe_problem_size
        # (`local_num_experts = w1.shape[0]`) with zero data bytes.
        empty_w13 = torch.empty(
            (spec.num_experts, 2 * spec.inter_size, 0),
            dtype=torch.uint8,
            device=device,
        )
        empty_w2 = torch.empty(
            (spec.num_experts, spec.hidden_size, 0),
            dtype=torch.uint8,
            device=device,
        )
        if "w13_weight" in layer._parameters:
            replace_parameter(layer, "w13_weight", empty_w13)
        else:
            layer.register_parameter(
                "w13_weight", torch.nn.Parameter(empty_w13, requires_grad=False)
            )
        if "w2_weight" in layer._parameters:
            replace_parameter(layer, "w2_weight", empty_w2)
        else:
            layer.register_parameter(
                "w2_weight", torch.nn.Parameter(empty_w2, requires_grad=False)
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        runtime_state = DxMoELayerRuntimeState(
            layer_idx=spec.layer_idx,
            hidden_size=spec.hidden_size,
            inter_size=spec.inter_size,
            num_experts=spec.num_experts,
            top_k=spec.top_k,
            w13=w13_packed,
            w2=w2_packed,
            s_in=s_in_dev,
            s_mid=s_mid_dev,
            beta_entry=beta_entry_dev,
            beta_down=beta_down_dev,
            tau_entry=spec.tau_entry,
            tau_down=spec.tau_down,
            schedule_mode=spec.schedule_mode,
            apply_coupling_entry=spec.apply_coupling_entry,
            apply_coupling_down=spec.apply_coupling_down,
            use_omega=spec.use_omega,
            sink_tokens=spec.sink_tokens,
            local_window=spec.local_window,
            pipelined_code=spec.pipelined_code,
        )
        self.moe_quant_config = DxMoEMoEQuantConfig(runtime_state)

        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
            use_monolithic=False,
        )
        assert prepare_finalize is not None
        experts = DxMoEExperts(
            moe_config=self.moe,
            quant_config=self.moe_quant_config,
        )
        self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        self._validate_layer_contract(layer)
        assert self.moe_kernel is not None
        router_kwargs = (
            {} if router_logits is None else {"router_logits": router_logits}
        )
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            **router_kwargs,
        )
