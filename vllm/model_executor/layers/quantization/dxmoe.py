# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DxMoE offline static W4/W8 + online dynamic A4/A8/A16 quantization config & MoE method."""

from __future__ import annotations

from dataclasses import dataclass
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
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import (
    is_weights_pre_processed,
    replace_parameter,
    set_weight_attrs,
)
from vllm.platforms import current_platform
from vllm.utils.dxmoe import (
    PIPELINED_CODE,
    SCHEDULE_MODE_CODE,
    pack_dxmoe_weight_channel,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

logger = init_logger(__name__)

_DXMOE_META_FORMAT = "scaling-moe.dxmoe.meta"
_DXMOE_THRESHOLDS_FORMAT = "scaling-moe.dxmoe.thresholds"
_DXMOE_META_FILENAME = "dxmoe_meta.json"
_DXMOE_SIGNS_FILENAME = "dxmoe_signs.safetensors"
_LAYER_IDX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
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
                or parts[2] not in ("s_in", "s_mid")
            ):
                raise ValueError(f"Unexpected tensor key in {signs_path}: {key!r}")
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


class DxMoEConfig(QuantizationConfig):
    """QuantizationConfig loading `dxmoe_meta.json` + `dxmoe_signs.safetensors`."""

    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, raw_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.raw_config: dict[str, Any] = dict(raw_config or {})
        self._resolved_specs: dict[int, _ParsedDxMoELayerSpec] | None = None

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
            self.raw_config.get("_model_dir")
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
        )
        thresholds_json_raw = os.environ.get(
            "VLLM_DXMOE_THRESHOLDS_JSON"
        ) or self.raw_config.get("thresholds_json")
        external_thresholds: dict[str, Any] | None = None
        if thresholds_json_raw:
            ext_doc = json.loads(Path(str(thresholds_json_raw)).read_text(encoding="utf-8"))
            if not isinstance(ext_doc, dict):
                raise ValueError("External DxMoE thresholds JSON must be a JSON object")
            if "format" in ext_doc or "version" in ext_doc:
                if (
                    ext_doc.get("format") != _DXMOE_THRESHOLDS_FORMAT
                    or ext_doc.get("version") != 1
                ):
                    raise ValueError(
                        f"Expected thresholds JSON format={_DXMOE_THRESHOLDS_FORMAT!r} "
                        f"version=1, got format={ext_doc.get('format')!r} "
                        f"version={ext_doc.get('version')!r}"
                    )
            ext_layers = ext_doc.get("layers", ext_doc)
            if not isinstance(ext_layers, dict):
                raise ValueError("External DxMoE thresholds 'layers' must be a JSON object")
            external_thresholds = ext_layers

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

        specs: dict[int, _ParsedDxMoELayerSpec] = {}
        for l_str, l_meta in layers_dict.items():
            l_idx = int(l_str)
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
                if bw_e not in (4, 8) or bw_d not in (4, 8):
                    raise ValueError(
                        f"Native DxMoE kernel supports W4/W8 static weights only; "
                        f"layer {l_idx} expert {e_idx} has entry=W{bw_e}, down=W{bw_d}"
                    )
                w_bits_entry.append(bw_e)
                w_bits_down.append(bw_d)
                be = float(ed["entry"]["beta"])
                bd = float(ed["down"]["beta"])
                if not (
                    math.isfinite(be)
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

            has_w8_entry = any(b == 8 for b in w_bits_entry)
            has_w8_down = any(b == 8 for b in w_bits_down)
            if raw_coupling is not None:
                coup_entry = raw_coupling
                coup_down = raw_coupling
            else:
                coup_entry = schedule_mode in ("dxmoe", "static_a4")
                coup_down = schedule_mode in ("dxmoe", "static_a4")

            if external_thresholds is not None:
                if l_str not in external_thresholds:
                    raise ValueError(
                        f"External thresholds JSON missing layer {l_idx}; "
                        f"available layers: {sorted(external_thresholds)}"
                    )
                layer_ext = external_thresholds[l_str]
                if not isinstance(layer_ext, dict):
                    raise ValueError(
                        f"External thresholds entry for layer {l_idx} must be a dict"
                    )
                t_map = layer_ext.get("thresholds", layer_ext)
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
                tau_entry = (float(raw_te[0]), float(raw_te[1]))
                tau_down = (float(raw_td[0]), float(raw_td[1]))
                for stage_name, (t0, t1) in (
                    ("tau_entry", tau_entry),
                    ("tau_down", tau_down),
                ):
                    if not (
                        math.isfinite(t0)
                        and math.isfinite(t1)
                        and 0.0 <= t0 <= t1
                    ):
                        raise ValueError(
                            f"Layer {l_idx} invalid {stage_name}={(t0, t1)}; "
                            "expected finite 0 <= tau1 <= tau2"
                        )
            elif schedule_mode == "dxmoe":
                raise ValueError(
                    f"schedule_mode='dxmoe' requires calibrated thresholds under "
                    f"threshold_key={threshold_key!r} for layer {l_idx}; "
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
            if s_in_cpu is not None and s_in_cpu.numel() != hidden_size:
                raise ValueError(
                    f"Layer {l_idx} s_in length {s_in_cpu.numel()} != {hidden_size}"
                )
            if s_mid_cpu is not None and s_mid_cpu.numel() != inter_size:
                raise ValueError(
                    f"Layer {l_idx} s_mid length {s_mid_cpu.numel()} != {inter_size}"
                )

            specs[l_idx] = _ParsedDxMoELayerSpec(
                layer_idx=l_idx,
                hidden_size=hidden_size,
                inter_size=inter_size,
                num_experts=num_experts,
                top_k=top_k,
                w_bits_entry=tuple(w_bits_entry),
                w_bits_down=tuple(w_bits_down),
                beta_entry=tuple(beta_entry),
                beta_down=tuple(beta_down),
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
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            specs = self._ensure_resolved()
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
            return DxMoEMoEMethod(spec, layer.moe_config)
        return None


class DxMoEMoEMethod(FusedMoEMethodBase):
    """FusedMoE method loading reconstructed float MoE weights and refitting to DxMoE W4/W8."""

    def __init__(self, layer_spec: _ParsedDxMoELayerSpec, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        self.layer_spec = layer_spec
        self.experts_cls = DxMoEExperts
        self._dxmoe_processed = False

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

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w13_weight.weight_loader_numel = num_experts * (2 * unpadded_inter) * unpadded_hidden

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        w2_weight.weight_loader_numel = num_experts * unpadded_hidden * unpadded_inter

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
        device = layer.w13_weight.device
        w13_packed = pack_dxmoe_weight_channel(layer.w13_weight.data, spec.w_bits_entry)
        w2_packed = pack_dxmoe_weight_channel(layer.w2_weight.data, spec.w_bits_down)

        # Register packed tensors on `layer` so module lifetime owns all GPU allocations.
        layer.register_buffer("_dxmoe_w13_codes", w13_packed.codes_arena, persistent=False)
        layer.register_buffer("_dxmoe_w13_scales", w13_packed.scales, persistent=False)
        layer.register_buffer("_dxmoe_w13_w_bits", w13_packed.w_bits_dev, persistent=False)
        layer.register_buffer("_dxmoe_w13_w_ptrs", w13_packed.w_ptrs_dev, persistent=False)
        layer.register_buffer("_dxmoe_w13_sw_ptrs", w13_packed.sw_ptrs_dev, persistent=False)

        layer.register_buffer("_dxmoe_w2_codes", w2_packed.codes_arena, persistent=False)
        layer.register_buffer("_dxmoe_w2_scales", w2_packed.scales, persistent=False)
        layer.register_buffer("_dxmoe_w2_w_bits", w2_packed.w_bits_dev, persistent=False)
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

        # Free dense float MoE weights while keeping 3D [E, N, 0] placeholders for
        # FusedMoEKernel.moe_problem_size (`local_num_experts = w1.shape[0]`).
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
        replace_parameter(layer, "w13_weight", empty_w13)
        replace_parameter(layer, "w2_weight", empty_w2)
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
