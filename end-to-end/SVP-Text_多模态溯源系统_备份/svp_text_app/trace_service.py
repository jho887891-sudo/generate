from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .bootstrap import ensure_region_route_on_path
from .config import settings
from .llm_service import LLMPaths, get_llm
from .prompting import SVP_TEXT_SYSTEM_PROMPT, build_prompt_sections, build_visual_text, classify_question


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HISTORY_DB = "svp_text_history.sqlite3"
RUN_ID_PATTERN = re.compile(r"^[0-9A-Za-z_\-\u4e00-\u9fff]+$")
HIGH_IMPACT_REGIONS = {"牙孔区域", "焊接区域", "重点区域"}
REGION_ORDER = ["一般区域", "牙孔区域", "焊接区域", "重点区域"]
REQUIRED_PARAM_FIELDS = ["慢速速度", "高速速度", "料饼厚度", "铸造压力", "建压时间", "射料结束位置"]


def _e2e_runtime():
    ensure_region_route_on_path()
    from e2e_compare_three_modes_graph_4b_v3 import (
        VisualStageRunner,
        build_info_text_for_prompt,
        filter_visual_bundle,
        normalize_output,
        retrieve_graph_context,
    )

    return {
        "VisualStageRunner": VisualStageRunner,
        "build_info_text_for_prompt": build_info_text_for_prompt,
        "filter_visual_bundle": filter_visual_bundle,
        "normalize_output": normalize_output,
        "retrieve_graph_context": retrieve_graph_context,
    }


def _fallback_normalize_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r"【结论】", text)
    return text[match.start():].strip() if match else text


def _save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: str | Path, default: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def sanitize_run_tag(text: str, max_len: int = 40) -> str:
    raw = str(text or "sample").strip()
    raw = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", raw)
    raw = raw.strip("._-") or "sample"
    return raw[:max_len]


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".jpg"


def _make_run_dir(sample_id: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = sanitize_run_tag(sample_id or "sample")
    run_id = f"{stamp}_{tag}_{uuid.uuid4().hex[:8]}"
    path = settings.runtime_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_questions(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("至少需要输入一个溯源问题。")
    if raw.startswith("["):
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError("问题 JSON 必须是字符串数组。")
        questions = [str(x).strip() for x in value if str(x).strip()]
    else:
        questions = [
            re.sub(r"^\s*(?:\d+[.、)]|-)\s*", "", line).strip()
            for line in raw.splitlines()
            if line.strip()
        ]
    if not questions:
        raise ValueError("至少需要输入一个溯源问题。")
    return questions


def save_uploaded_images(files: Iterable[Any], image_dir: Path) -> List[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    file_list = list(files or [])
    if not file_list:
        raise ValueError("至少需要上传一张图片。")
    if len(file_list) > settings.max_upload_images:
        raise ValueError(f"最多支持上传 {settings.max_upload_images} 张图片。")
    max_bytes = int(settings.max_upload_mb) * 1024 * 1024
    for index, file in enumerate(file_list, start=1):
        filename = getattr(file, "filename", "") or f"view_{index:02d}.jpg"
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"图片格式不支持：{filename}")
        try:
            stream = file.file
            current = stream.tell()
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(current)
        except Exception:
            size = 0
        if size and size > max_bytes:
            raise ValueError(f"单张图片不能超过 {settings.max_upload_mb} MB：{filename}")
        target = image_dir / f"view_{index:02d}{_safe_suffix(filename)}"
        try:
            file.file.seek(0)
        except Exception:
            pass
        with target.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        if target.stat().st_size > max_bytes:
            target.unlink(missing_ok=True)
            raise ValueError(f"单张图片不能超过 {settings.max_upload_mb} MB：{filename}")
        saved.append(str(target))
    return saved


def build_request_config(
    sample_id: str,
    params: Dict[str, Any],
    questions: Sequence[str],
    use_graph_rag: bool,
    run_dir: Path,
) -> Dict[str, Any]:
    cfg = settings.load_base_config()
    question = questions[0] if questions else ""
    cfg["sample_id"] = sample_id
    cfg["workpiece_id"] = sample_id
    cfg["params"] = params
    cfg["question"] = question
    cfg["questions"] = list(questions)
    cfg["use_graph_rag"] = bool(use_graph_rag)
    cfg["image_dir"] = str(run_dir / "images")
    cfg["visual_output_dir"] = str(run_dir / "visual")
    cfg["result_output_dir"] = str(run_dir / "result")
    cfg["save_intermediate_visual_files"] = True
    return cfg


def _relative_url_for_run_file(run_dir: Path, file_path: str | Path) -> str:
    rel = Path(file_path).resolve().relative_to(run_dir.resolve()).as_posix()
    return f"/api/runs/{run_dir.name}/files/{rel}"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _rounded(value: Any, digits: int = 3) -> float:
    return round(_as_float(value), digits)


def _defect_area(defect: Dict[str, Any]) -> float:
    for key in ("area_mm2", "Area (mm2)", "area", "area_mm^2"):
        if key in defect:
            return _as_float(defect.get(key))
    return 0.0


def _defect_diameter(defect: Dict[str, Any]) -> float:
    for key in ("eq_diameter_mm", "diameter_mm", "Diameter (mm)", "equivalent_diameter_mm"):
        if key in defect:
            return _as_float(defect.get(key))
    return 0.0


def _defect_confidence(defect: Dict[str, Any]) -> Any:
    for key in ("confidence", "score", "conf"):
        if key in defect:
            return defect.get(key)
    return ""


def build_region_cards(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    region_count = summary.get("region_count", {}) or {}
    region_area = summary.get("region_max_area_mm2", {}) or {}
    region_dia = summary.get("region_max_eq_diameter_mm", {}) or {}
    region_total_area = summary.get("region_total_area_mm2", {}) or summary.get("region_area_mm2", {}) or {}
    names = [name for name in REGION_ORDER if name in region_count or name in region_area or name in region_dia]
    names.extend(sorted(set(region_count) | set(region_area) | set(region_dia) | set(region_total_area) - set(names)))
    cards: List[Dict[str, Any]] = []
    for name in names:
        count = int(_as_float(region_count.get(name, 0)))
        high_impact = name in HIGH_IMPACT_REGIONS
        note = "高影响区域，建议优先结合参数链复核。" if high_impact and count else "作为整体波动参考。" if name == "一般区域" else "当前区域未见突出气孔统计。"
        cards.append({
            "region": name,
            "count": count,
            "total_area_mm2": _rounded(region_total_area.get(name, 0)),
            "max_area_mm2": _rounded(region_area.get(name, 0)),
            "max_eq_diameter_mm": _rounded(region_dia.get(name, 0)),
            "high_impact": high_impact,
            "note": note,
        })
    return cards


def build_defect_table(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_index, defects in enumerate(bundle.get("defect_meta_by_view", []) or [], start=1):
        if not isinstance(defects, list):
            continue
        for defect_index, defect in enumerate(defects, start=1):
            if not isinstance(defect, dict):
                continue
            bbox = defect.get("bbox") or defect.get("box") or []
            rows.append({
                "view_index": view_index,
                "defect_id": f"V{view_index:02d}-D{defect_index:02d}",
                "region_type": str(defect.get("region_type") or defect.get("region") or "未知区域"),
                "bbox": bbox,
                "area_mm2": _rounded(_defect_area(defect)),
                "eq_diameter_mm": _rounded(_defect_diameter(defect)),
                "confidence": "" if _defect_confidence(defect) == "" else _rounded(_defect_confidence(defect), 4),
            })
    return rows


def build_param_table(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for key, value in (params or {}).items():
        raw = "" if value is None else str(value)
        number = _as_float(value, None) if raw != "" else None
        table.append({
            "name": key,
            "value": value,
            "is_numeric": number is not None,
            "display": raw,
        })
    return table


def validate_answer_format(answer: str) -> Dict[str, Any]:
    text = answer or ""
    required = ["【结论】", "【溯源分析】", "【建议方向】"]
    missing = [item for item in required if item not in text]
    return {
        "valid": not missing and text.strip().startswith("【结论】"),
        "missing_sections": missing,
        "starts_with_conclusion": text.strip().startswith("【结论】"),
    }


def make_artifact_urls(run_dir: Path, index: int) -> Dict[str, str]:
    return {
        "prompt_url": _relative_url_for_run_file(run_dir, run_dir / "result" / "prompts" / f"q{index:02d}_prompt.txt"),
        "answer_url": _relative_url_for_run_file(run_dir, run_dir / "result" / "answers" / f"q{index:02d}_answer.txt"),
    }


def _draw_visual_results(run_dir: Path, image_paths: Sequence[str], bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    output_dir = run_dir / "result" / "annotated_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery: List[Dict[str, Any]] = []

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return gallery

    color_map = {
        "一般区域": "#2f80ed",
        "牙孔区域": "#d13f31",
        "焊接区域": "#9a5b12",
        "重点区域": "#7b3ff2",
    }
    defect_meta_by_view = bundle.get("defect_meta_by_view", []) or []

    for idx, image_path in enumerate(image_paths, start=1):
        src = Path(image_path)
        target = output_dir / f"view_{idx:02d}_svp_text.jpg"
        defects = defect_meta_by_view[idx - 1] if idx - 1 < len(defect_meta_by_view) else []
        try:
            image = Image.open(src).convert("RGB")
            draw = ImageDraw.Draw(image)
            regions = []
            max_area = 0.0
            max_diameter = 0.0
            for defect_idx, defect in enumerate(defects, start=1):
                box = defect.get("bbox") or []
                if len(box) != 4:
                    continue
                region = str(defect.get("region_type", "区域"))
                regions.append(region)
                max_area = max(max_area, _defect_area(defect))
                max_diameter = max(max_diameter, _defect_diameter(defect))
                color = color_map.get(region, "#0d766e")
                x1, y1, x2, y2 = [float(v) for v in box]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                label = f"D{defect_idx} {defect.get('diameter_mm', defect.get('Diameter (mm)', ''))}mm"
                text_y = max(0, y1 - 18)
                draw.rectangle([x1, text_y, min(image.width, x1 + 190), text_y + 18], fill=color)
                draw.text((x1 + 4, text_y + 2), label, fill="#ffffff")
            image.save(target, quality=92)
            gallery.append({
                "view_index": idx,
                "image_name": src.name,
                "defect_count": len(defects),
                "major_regions": sorted(set(regions)),
                "max_area_mm2": _rounded(max_area),
                "max_eq_diameter_mm": _rounded(max_diameter),
                "file": str(target),
                "url": _relative_url_for_run_file(run_dir, target),
            })
        except Exception:
            continue

    return gallery


def _write_download_bundle(run_dir: Path) -> Path:
    bundle_path = run_dir / "result" / "svp_text_outputs.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in (run_dir / "result").rglob("*"):
            if path.is_file() and path != bundle_path:
                zf.write(path, path.relative_to(run_dir))
        request_cfg = run_dir / "request_config.json"
        if request_cfg.exists():
            zf.write(request_cfg, request_cfg.relative_to(run_dir))
    return bundle_path


def _history_db_path() -> Path:
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    return settings.runtime_dir / HISTORY_DB


def _history_conn():
    conn = sqlite3.connect(_history_db_path())
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trace_history (
            run_id TEXT PRIMARY KEY,
            sample_id TEXT,
            created_at TEXT,
            question_count INTEGER,
            image_count INTEGER,
            download_url TEXT,
            result_file TEXT,
            summary TEXT
        )
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trace_history)").fetchall()}
    for name, ddl in {
        "use_graph_rag": "INTEGER DEFAULT 0",
        "status": "TEXT DEFAULT 'completed'",
        "duration_ms": "INTEGER DEFAULT 0",
    }.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE trace_history ADD COLUMN {name} {ddl}")
    return conn


def _append_history(record: Dict[str, Any]) -> None:
    with _history_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO trace_history
            (run_id, sample_id, created_at, question_count, image_count, download_url, result_file, summary, use_graph_rag, status, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("run_id", ""),
                record.get("sample_id", ""),
                record.get("created_at", ""),
                int(record.get("question_count", 0) or 0),
                int(record.get("image_count", 0) or 0),
                record.get("download_url", ""),
                record.get("result_file", ""),
                record.get("summary", ""),
                1 if record.get("use_graph_rag") else 0,
                record.get("status", "completed"),
                int(record.get("duration_ms", 0) or 0),
            ),
        )


def list_recent_history(days: int = 7) -> List[Dict[str, Any]]:
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_text = cutoff.isoformat(timespec="seconds")
    with _history_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT run_id, sample_id, created_at, question_count, image_count, download_url, result_file, summary,
                   use_graph_rag, status, duration_ms
            FROM trace_history
            WHERE created_at >= ?
            ORDER BY created_at DESC
            """,
            (cutoff_text,),
        ).fetchall()
    return [dict(row) for row in rows]


def resolve_run_file(run_id: str, rel_path: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id or ""):
        raise ValueError("非法运行编号。")
    run_dir = (settings.runtime_dir / run_id).resolve()
    target = (run_dir / rel_path).resolve()
    try:
        target.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("非法文件路径。") from exc
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError("运行目录不存在。")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError("文件不存在。")
    return target


def resolve_run_download(run_id: str) -> Path:
    return resolve_run_file(run_id, "result/svp_text_outputs.zip")


def load_run_result(run_id: str) -> Dict[str, Any]:
    return _load_json(resolve_run_file(run_id, "result/svp_text_result.json"), {})


def list_run_artifacts(run_id: str) -> List[Dict[str, Any]]:
    if not RUN_ID_PATTERN.fullmatch(run_id or ""):
        raise ValueError("非法运行编号。")
    run_dir = (settings.runtime_dir / run_id).resolve()
    result_dir = run_dir / "result"
    if not result_dir.exists():
        raise FileNotFoundError("结果目录不存在。")
    artifacts: List[Dict[str, Any]] = []
    for path in sorted(result_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.resolve().relative_to(run_dir).as_posix()
        except ValueError:
            continue
        artifacts.append({
            "name": path.name,
            "path": rel,
            "size": path.stat().st_size,
            "url": f"/api/runs/{run_id}/files/{rel}",
        })
    return artifacts


def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("压铸参数必须是 JSON 对象。")
    missing = [key for key in REQUIRED_PARAM_FIELDS if params.get(key) in ("", None)]
    non_numeric = []
    for key, value in params.items():
        if value in ("", None):
            continue
        if _as_float(value, None) is None:
            non_numeric.append(key)
    high_risk = [
        key for key in ("高速速度", "铸造压力", "建压时间", "压射压力")
        if key in params and params.get(key) not in ("", None)
    ]
    return {
        "ok": not missing,
        "missing_fields": missing,
        "non_numeric_fields": non_numeric,
        "suggested_fields": [key for key in REQUIRED_PARAM_FIELDS if key not in params],
        "high_risk_fields": high_risk,
        "field_count": len(params),
    }


class SVPTextTraceService:
    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._gpu_lock = threading.Lock()

    def _status_path(self, run_dir: Path) -> Path:
        return run_dir / "job_status.json"

    def _set_status(self, job: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
        job.update(updates)
        job["updated_at"] = datetime.now().isoformat(timespec="seconds")
        status = {
            "job_id": job["job_id"],
            "run_id": job["run_id"],
            "sample_id": job.get("sample_id", ""),
            "status": job.get("status", "running"),
            "stage": job.get("stage", ""),
            "progress": int(job.get("progress", 0)),
            "message": job.get("message", ""),
            "queued_position": int(job.get("queued_position", 0) or 0),
            "created_at": job.get("created_at", ""),
            "updated_at": job.get("updated_at", ""),
            "result": job.get("result"),
            "error": job.get("error", ""),
        }
        _save_json(self._status_path(Path(job["run_dir"])), status)
        return status

    def create_job(
        self,
        image_files: Iterable[Any],
        params: Dict[str, Any],
        questions_text: str,
        sample_id: str = "",
        use_graph_rag: bool = False,
        runtime_options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not isinstance(params, dict):
            raise ValueError("压铸参数必须是 JSON 对象。")
        questions = parse_questions(questions_text)
        run_dir = _make_run_dir(sample_id or params.get("样本编号") or params.get("探伤图编号"))
        image_paths = save_uploaded_images(image_files, run_dir / "images")
        sample_id = sample_id or str(params.get("样本编号") or params.get("探伤图编号") or run_dir.name)
        cfg_json = build_request_config(sample_id, params, questions, use_graph_rag, run_dir)
        for key in ("max_new_tokens", "max_seq_length", "graph_top_k"):
            if runtime_options and runtime_options.get(key) not in ("", None):
                cfg_json[key] = int(runtime_options[key])
        _save_json(run_dir / "request_config.json", cfg_json)

        job = {
            "job_id": run_dir.name,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "sample_id": sample_id,
            "image_paths": image_paths,
            "questions": questions,
            "params": params,
            "use_graph_rag": bool(use_graph_rag),
            "cfg_json": cfg_json,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "任务已创建，等待执行。",
            "queued_position": 0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "result": None,
            "error": "",
        }
        self.jobs[job["job_id"]] = job
        return self._set_status(job)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        if job_id in self.jobs:
            return self._set_status(self.jobs[job_id])
        status_path = settings.runtime_dir / job_id / "job_status.json"
        status = _load_json(status_path, None)
        if status:
            return status
        raise KeyError("任务不存在。")

    def run_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        try:
            self._set_status(
                job,
                status="queued",
                stage="queued",
                progress=3,
                queued_position=1 if self._gpu_lock.locked() else 0,
                message="任务进入 GPU 推理队列，等待视觉模型和 LLM 独占运行。",
            )
            with self._gpu_lock:
                self._set_status(
                    job,
                    status="running",
                    stage="queued",
                    progress=5,
                    queued_position=0,
                    message="已获得 GPU 推理资源，开始执行。",
                )
                result = self._analyze_saved_job(job)
            self._set_status(
                job,
                status="completed",
                stage="completed",
                progress=100,
                message="溯源任务已完成。",
                result=result,
            )
        except Exception as exc:
            self._set_status(
                job,
                status="failed",
                stage="failed",
                progress=int(job.get("progress", 0)),
                message="任务执行失败。",
                error=str(exc),
            )

    def analyze(
        self,
        image_files: Iterable[Any],
        params: Dict[str, Any],
        question: str,
        sample_id: str = "",
        use_graph_rag: bool = False,
    ) -> Dict[str, Any]:
        status = self.create_job(image_files, params, question, sample_id, use_graph_rag)
        job_id = status["job_id"]
        self.run_job(job_id)
        final_status = self.get_job(job_id)
        if final_status.get("status") == "failed":
            raise RuntimeError(final_status.get("error", "任务执行失败。"))
        return final_status["result"]

    def _analyze_saved_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = Path(job["run_dir"])
        cfg_json = dict(job["cfg_json"])
        image_paths = list(job["image_paths"])
        questions = list(job["questions"])
        params = dict(job["params"])
        sample_id = job["sample_id"]
        use_graph_rag = bool(job["use_graph_rag"])
        created_at = job.get("created_at", datetime.now().isoformat(timespec="seconds"))
        overall_start = time.perf_counter()
        pipeline: List[Dict[str, Any]] = []

        def add_stage(key: str, name: str, status: str, duration_start: float, message: str = "") -> None:
            pipeline.append({
                "key": key,
                "name": name,
                "status": status,
                "duration_ms": int((time.perf_counter() - duration_start) * 1000),
                "message": message,
            })

        upload_start = time.perf_counter()
        add_stage("upload", "图片上传完成", "completed", upload_start, f"已保存 {len(image_paths)} 张探伤图片。")
        self._set_status(job, status="running", stage="vision", progress=8, message="正在载入视觉层模型并识别图片。")
        vision_start = time.perf_counter()
        runtime = _e2e_runtime()
        vision = runtime["VisualStageRunner"](cfg_json)
        try:
            bundle = vision.run(image_paths)
            bundle = runtime["filter_visual_bundle"](bundle)
        finally:
            vision.release()
        add_stage("vision", "视觉识别", "completed", vision_start, "视觉模型识别与区域归属完成。")

        visual_text_start = time.perf_counter()
        visual_gallery = _draw_visual_results(run_dir, image_paths, bundle)
        visual_fact_block = build_visual_text(bundle["aggregate_summary"])
        region_cards = build_region_cards(bundle["aggregate_summary"])
        defect_table = build_defect_table(bundle)
        _save_text(run_dir / "result" / "visual_fact_block.txt", visual_fact_block)
        _save_json(run_dir / "result" / "visual_summary.json", bundle["aggregate_summary"])
        add_stage("visual_text", "视觉统计转文本", "completed", visual_text_start, "已生成 SVP-Text 视觉事实块。")
        self._set_status(job, stage="visual_summary", progress=30, message="视觉识别完成，正在生成文字总结。")

        llm_start = time.perf_counter()
        graph_total_ms = 0
        llm_paths = settings.resolve_llm_paths(cfg_json)
        llm = get_llm(
            LLMPaths(base_path=llm_paths["base_path"], adapter_path=llm_paths["adapter_path"]),
            generation_max_time_sec=float(cfg_json.get("generation_max_time_sec", settings.generation_max_time_sec)),
        )

        question_results = []
        answer_dir = run_dir / "result" / "answers"
        prompt_dir = run_dir / "result" / "prompts"
        answer_dir.mkdir(parents=True, exist_ok=True)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        total = max(1, len(questions))
        for index, question in enumerate(questions, start=1):
            base_progress = 35 + int(((index - 1) / total) * 55)
            self._set_status(
                job,
                stage="llm",
                progress=base_progress,
                message=f"正在生成第 {index}/{total} 个问题的溯源答案。",
            )
            question_type, question_type_hint, focus_regions, mechanism_candidates = classify_question(question)
            graph_context = ""
            graph_debug: Dict[str, Any] = {
                "enabled": bool(use_graph_rag),
                "available": False,
                "status": "disabled",
                "question_type": question_type,
                "focus_regions": focus_regions,
                "mechanism_candidates": mechanism_candidates,
            }
            if use_graph_rag:
                graph_start = time.perf_counter()
                try:
                    graph_context, graph_debug_base = runtime["retrieve_graph_context"](
                        cfg_json=cfg_json,
                        info_text=runtime["build_info_text_for_prompt"](cfg_json),
                        question=question,
                        graph_top_k=int(cfg_json.get("graph_top_k", 10)),
                    )
                    graph_debug.update(graph_debug_base)
                    graph_debug["question_type"] = question_type
                    graph_debug["focus_regions"] = focus_regions
                    graph_debug["mechanism_candidates"] = mechanism_candidates
                except Exception as exc:
                    graph_debug["status"] = f"graph retrieval failed: {exc}"
                finally:
                    graph_total_ms += int((time.perf_counter() - graph_start) * 1000)

            prompt_sections = build_prompt_sections(
                cfg_json=cfg_json,
                question=question,
                visual_fact_block=visual_fact_block,
                graph_context=graph_context,
                question_type_hint=question_type_hint,
            )
            user_prompt = prompt_sections["user_prompt"]
            generation = llm.generate_with_stats(
                system_prompt=SVP_TEXT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=int(cfg_json.get("max_new_tokens", settings.max_new_tokens)),
                max_seq_length=int(cfg_json.get("max_seq_length", settings.max_seq_length)),
            )
            raw_answer = generation["text"]
            try:
                answer = runtime["normalize_output"](raw_answer)
            except Exception:
                answer = _fallback_normalize_output(raw_answer)

            item = {
                "index": index,
                "question": question,
                "question_type": question_type,
                "focus_regions": focus_regions,
                "mechanism_candidates": mechanism_candidates,
                "graph_context": graph_context,
                "graph_debug": graph_debug,
                "user_prompt": user_prompt,
                "prompt_sections": prompt_sections,
                "prompt_char_count": generation["prompt_char_count"],
                "prompt_token_count": generation["prompt_token_count"],
                "max_seq_length": generation["max_seq_length"],
                "was_truncated": generation["was_truncated"],
                "format_check": validate_answer_format(answer),
                "evidence": {
                    "focus_regions": focus_regions,
                    "mechanism_candidates": mechanism_candidates,
                    "graph_used": bool(use_graph_rag and graph_context),
                },
                "artifacts": make_artifact_urls(run_dir, index),
                "raw_answer": raw_answer,
                "answer": answer,
            }
            question_results.append(item)
            _save_text(answer_dir / f"q{index:02d}_answer.txt", answer)
            _save_text(prompt_dir / f"q{index:02d}_prompt.txt", user_prompt)
        if use_graph_rag:
            pipeline.append({
                "key": "graph",
                "name": "图谱证据检索",
                "status": "completed",
                "duration_ms": graph_total_ms,
                "message": "已按问题检索图谱辅助证据。",
            })
        else:
            pipeline.append({
                "key": "graph",
                "name": "图谱证据检索",
                "status": "skipped",
                "duration_ms": 0,
                "message": "本次运行未启用图谱辅助。",
            })
        add_stage("llm", "LLM 溯源生成", "completed", llm_start, f"已生成 {len(question_results)} 个问题答案。")

        answers_md = ["# SVP-Text 溯源结果", ""]
        for item in question_results:
            answers_md.extend([
                f"## Q{item['index']:02d}",
                "",
                item["question"],
                "",
                item["answer"],
                "",
            ])
        _save_text(run_dir / "result" / "answers.md", "\n".join(answers_md))
        self._set_status(job, stage="packaging", progress=94, message="正在整理输出文件和下载包。")

        packaging_start = time.perf_counter()
        answer_items = []
        for item in question_results:
            answer_items.append({
                "index": item["index"],
                "question": item["question"],
                "question_type": item["question_type"],
                "answer": item["answer"],
                "evidence": item["evidence"],
                "graph_debug": item["graph_debug"],
                "format_check": item["format_check"],
                "prompt_sections": item["prompt_sections"],
                "prompt_char_count": item["prompt_char_count"],
                "prompt_token_count": item["prompt_token_count"],
                "max_seq_length": item["max_seq_length"],
                "was_truncated": item["was_truncated"],
                "artifacts": item["artifacts"],
            })

        result = {
            "method": "SVP-Text",
            "run_id": run_dir.name,
            "display_sample_id": sample_id,
            "sample_id": sample_id,
            "created_at": created_at,
            "run_dir": str(run_dir),
            "input": {
                "image_count": len(image_paths),
                "params": params,
                "questions": questions,
                "use_graph_rag": use_graph_rag,
            },
            "pipeline": pipeline,
            "visual": {
                "summary": bundle["aggregate_summary"],
                "fact_block": visual_fact_block,
                "gallery": visual_gallery,
                "region_cards": region_cards,
                "defect_table": defect_table,
            },
            "params": {
                "raw": params,
                "table": build_param_table(params),
                "validation": validate_params(params),
            },
            "answers": answer_items,
            "llm_base_path": llm_paths["base_path"],
            "llm_adapter_path": llm_paths["adapter_path"],
            "max_seq_length": int(cfg_json.get("max_seq_length", settings.max_seq_length)),
            "image_paths": image_paths,
            "questions": questions,
            "raw_params": params,
            "visual_summary": bundle["aggregate_summary"],
            "visual_fact_block": visual_fact_block,
            "visual_gallery": visual_gallery,
            "results": question_results,
            "answer": question_results[0]["answer"] if question_results else "",
        }
        _save_text(run_dir / "result" / "answer.txt", result["answer"])
        bundle_path = run_dir / "result" / "svp_text_outputs.zip"
        result["download_url"] = f"/api/runs/{run_dir.name}/download"
        result["download_file"] = str(bundle_path)
        add_stage("packaging", "输出打包", "completed", packaging_start, "已生成 JSON、Markdown、Prompt、答案和 zip 包。")
        result["pipeline"] = pipeline
        result["duration_ms"] = int((time.perf_counter() - overall_start) * 1000)

        _save_json(run_dir / "result" / "svp_text_result.json", result)
        _write_download_bundle(run_dir)
        _append_history({
            "run_id": run_dir.name,
            "sample_id": sample_id,
            "created_at": result["created_at"],
            "question_count": len(questions),
            "image_count": len(image_paths),
            "download_url": result["download_url"],
            "result_file": str(run_dir / "result" / "svp_text_result.json"),
            "summary": question_results[0]["answer"][:160] if question_results else "",
            "use_graph_rag": use_graph_rag,
            "status": "completed",
            "duration_ms": result["duration_ms"],
        })
        return result


def parse_params_json(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"压铸参数 JSON 格式错误: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("压铸参数 JSON 必须是对象。")
    return value
