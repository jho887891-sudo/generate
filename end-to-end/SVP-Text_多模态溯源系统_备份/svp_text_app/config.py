from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .bootstrap import DEFAULT_END_TO_END_ROOT, ensure_region_route_on_path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_config_path() -> Path:
    explicit = os.environ.get("SVP_TEXT_CONFIG_JSON", "")
    if explicit:
        return Path(explicit)
    route_root = ensure_region_route_on_path()
    route_cfg = route_root / "e2e_compare_graph_config_hardcoded.json"
    if route_cfg.exists():
        return route_cfg
    bundled = SYSTEM_ROOT / "configs" / "svp_text_config.example.json"
    return bundled


def _default_model_paths() -> Dict[str, str]:
    try:
        ensure_region_route_on_path()
        from multimodal.config_region_mm import RegionAwareMMConfig

        mm_cfg = RegionAwareMMConfig()
        return {
            "base_path": str(mm_cfg.llm_base_path),
            "stage2_adapter_path": str(mm_cfg.llm_stage2_adapter_path),
        }
    except Exception:
        return {
            "base_path": str(DEFAULT_END_TO_END_ROOT / "models" / "Qwen3-4B-stage1-book-merged-4bit"),
            "stage2_adapter_path": str(
                DEFAULT_END_TO_END_ROOT
                / "runs"
                / "qwen3_4b_stage2_from_merged_Third_sft"
                / "best_adapter"
            ),
        }


def _normalize_adapter_path(path_text: str) -> str:
    """Accept either the adapter directory itself or its parent run directory."""
    if not path_text:
        return ""
    path = Path(path_text)
    nested_best = path / "best_adapter"
    if nested_best.exists():
        return str(nested_best)
    return str(path)


@dataclass(frozen=True)
class SVPTextSettings:
    app_name: str = "SVP-Text"
    config_json: Path = _default_config_path()
    runtime_dir: Path = Path(
        os.environ.get(
            "SVP_TEXT_RUNTIME_DIR",
            str(DEFAULT_END_TO_END_ROOT / "compare_outputs" / "svp_text_runtime"),
        )
    )
    max_new_tokens: int = int(os.environ.get("SVP_TEXT_MAX_NEW_TOKENS", "1280"))
    max_seq_length: int = int(os.environ.get("SVP_TEXT_MAX_SEQ_LENGTH", "4096"))
    generation_max_time_sec: float = float(os.environ.get("SVP_TEXT_GENERATION_MAX_TIME_SEC", "180"))
    max_upload_images: int = int(os.environ.get("SVP_TEXT_MAX_UPLOAD_IMAGES", "12"))
    max_upload_mb: int = int(os.environ.get("SVP_TEXT_MAX_UPLOAD_MB", "25"))

    def load_base_config(self) -> Dict[str, Any]:
        cfg = _read_json(self.config_json)
        cfg.setdefault("save_intermediate_visual_files", True)
        cfg.setdefault("use_graph_rag", False)
        cfg.setdefault("graph_top_k", 10)
        cfg.setdefault("prompt_graph_max_chars", 700)
        cfg.setdefault("prompt_visual_max_chars", 3000)
        cfg.setdefault("prompt_params_max_chars", 1800)
        cfg.setdefault("max_new_tokens", self.max_new_tokens)
        cfg.setdefault("max_seq_length", self.max_seq_length)
        cfg.setdefault("generation_max_time_sec", self.generation_max_time_sec)
        return cfg

    def resolve_llm_paths(self, cfg_json: Dict[str, Any]) -> Dict[str, str]:
        defaults = _default_model_paths()
        base_path = os.environ.get("DIECAST_LLM_BASE_PATH", defaults["base_path"])

        adapter_path = os.environ.get(
            "SVP_TEXT_LLM_ADAPTER_PATH",
            os.environ.get("DIECAST_STAGE2_ADAPTER_PATH", defaults["stage2_adapter_path"]),
        )
        if not adapter_path:
            adapter_path = defaults["stage2_adapter_path"]

        return {
            "base_path": base_path,
            "adapter_path": _normalize_adapter_path(adapter_path),
        }


settings = SVPTextSettings()
