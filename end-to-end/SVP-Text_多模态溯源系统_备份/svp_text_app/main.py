from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .bootstrap import find_region_route_root
from .config import SYSTEM_ROOT, settings
from .trace_service import (
    SVPTextTraceService,
    list_run_artifacts,
    list_recent_history,
    load_run_result,
    parse_params_json,
    resolve_run_download,
    resolve_run_file,
    validate_params,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="SVP-Text Traceability System", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

trace_service = SVPTextTraceService()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "method": "SVP-Text",
        "system_root": str(SYSTEM_ROOT),
        "region_route_root": str(find_region_route_root()),
        "config_json": str(settings.config_json),
        "runtime_dir": str(settings.runtime_dir),
    }


@app.get("/api/config")
def config_info():
    cfg = settings.load_base_config()
    paths = settings.resolve_llm_paths(cfg)
    return {
        "method": "SVP-Text",
        "system_root": str(SYSTEM_ROOT),
        "region_route_root": str(find_region_route_root()),
        "config_json": str(settings.config_json),
        "config_name": Path(settings.config_json).name,
        "runtime_dir": str(settings.runtime_dir),
        "visual_model_path": cfg.get("visual_model_path", ""),
        "visual_model_name": Path(str(cfg.get("visual_model_path", ""))).name,
        "llm_base_path": paths["base_path"],
        "llm_base_name": Path(paths["base_path"]).name,
        "llm_adapter_path": paths["adapter_path"],
        "llm_adapter_name": Path(paths["adapter_path"]).name,
        "use_graph_rag_default": bool(cfg.get("use_graph_rag", False)),
        "max_new_tokens": cfg.get("max_new_tokens"),
        "max_seq_length": cfg.get("max_seq_length"),
        "graph_top_k": cfg.get("graph_top_k"),
    }


@app.post("/api/trace/start")
def start_trace(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    params_json: str = Form(...),
    questions: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
    sample_id: str = Form(""),
    use_graph_rag: bool = Form(False),
    max_new_tokens: Optional[int] = Form(None),
    max_seq_length: Optional[int] = Form(None),
    graph_top_k: Optional[int] = Form(None),
):
    try:
        params = parse_params_json(params_json)
        validate_params(params)
        status = trace_service.create_job(
            image_files=images,
            params=params,
            questions_text=questions or question or "",
            sample_id=sample_id,
            use_graph_rag=use_graph_rag,
            runtime_options={
                "max_new_tokens": max_new_tokens,
                "max_seq_length": max_seq_length,
                "graph_top_k": graph_top_k,
            },
        )
        background_tasks.add_task(trace_service.run_job, status["job_id"])
        return status
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return trace_service.get_job(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/history")
def history(days: int = 7):
    return {
        "days": days,
        "items": list_recent_history(days=days),
    }


@app.get("/api/runs/{run_id}/result")
def run_result(run_id: str):
    try:
        result = load_run_result(run_id)
        if not result:
            raise FileNotFoundError("结果文件为空或不存在。")
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/artifacts")
def run_artifacts(run_id: str):
    try:
        return {"run_id": run_id, "items": list_run_artifacts(run_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/params/validate")
def validate_params_api(payload: dict):
    try:
        return validate_params(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/download")
def download_run(run_id: str):
    try:
        return FileResponse(
            resolve_run_download(run_id),
            media_type="application/zip",
            filename=f"{run_id}_svp_text_outputs.zip",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/files/{rel_path:path}")
def run_file(run_id: str, rel_path: str):
    try:
        return FileResponse(resolve_run_file(run_id, rel_path))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/trace")
def trace(
    images: List[UploadFile] = File(...),
    params_json: str = Form(...),
    question: str = Form(...),
    sample_id: str = Form(""),
    use_graph_rag: bool = Form(False),
):
    try:
        params = parse_params_json(params_json)
        validate_params(params)
        return trace_service.analyze(
            image_files=images,
            params=params,
            question=question,
            sample_id=sample_id,
            use_graph_rag=use_graph_rag,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
