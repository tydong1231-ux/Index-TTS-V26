from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, load_settings
from .jobs import JobManager
from .llm import LLMConfig, OpenAICompatibleClient
from .preprocess import PreprocessService
from .schemas import AutoMatchRequest, JobCreate, ModelListRequest, ProjectCreate, ProjectSettingsUpdate, RoleUpdate, SegmentUpdate
from .store import ConflictError, NotFoundError, ProjectStore
from .tts import IndexTTSAdapter, TTSService
from .voices import VoiceLibrary, VoiceMatcher


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = ProjectStore(settings)
        self.voices = VoiceLibrary(settings)
        self.matcher = VoiceMatcher(self.store, self.voices)
        self.preprocess = PreprocessService(self.store)
        self.tts_adapter = IndexTTSAdapter(settings)
        self.tts = TTSService(settings, self.store, self.voices, self.tts_adapter)
        self.jobs = JobManager(self.store, self.preprocess, self.matcher, self.tts)

    def shutdown(self) -> None:
        self.jobs.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = Services(load_settings())
    app.state.services = svc
    yield
    svc.shutdown()


app = FastAPI(title="VoiceBook Studio API", version="0.1.0", description="Project-based novel preprocessing and IndexTTS audiobook production API.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1", "http://localhost", "null"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def services(request: Request) -> Services:
    return request.app.state.services


@app.exception_handler(NotFoundError)
async def not_found_handler(_: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc).strip("'")})


@app.exception_handler(ConflictError)
async def conflict_handler(_: Request, exc: ConflictError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    svc = services(request)
    return {"status": "ok", "project_count": len(svc.store.list_projects()), "voice_count": len(svc.voices.list()), "tts": svc.tts_adapter.status(), "data_root": str(svc.settings.data_root), "frontend_dir": str(svc.settings.frontend_dir)}


@app.get("/api/config")
def safe_config(request: Request) -> dict[str, Any]:
    svc = services(request)
    return {"api_base_url": svc.settings.api_base_url, "model": svc.settings.preprocess_model, "api_key_configured": bool(svc.settings.api_key), "model_dir": str(svc.settings.model_dir), "voice_library_dir": str(svc.settings.voice_library_dir)}


@app.post("/api/models")
def list_models(payload: ModelListRequest, request: Request) -> dict[str, Any]:
    svc = services(request)
    client = OpenAICompatibleClient(LLMConfig(api_base_url=(payload.api_base_url or svc.settings.api_base_url).rstrip("/"), api_key=payload.api_key.get_secret_value() if payload.api_key else svc.settings.api_key, model=svc.settings.preprocess_model))
    return {"models": client.list_models()}


@app.get("/api/projects")
def list_projects(request: Request) -> dict[str, Any]:
    return {"projects": services(request).store.list_projects()}


@app.post("/api/projects", status_code=201)
def create_project(payload: ProjectCreate, request: Request) -> dict[str, Any]:
    return services(request).store.create_project(payload.name, payload.source_path, payload.api_base_url, payload.model)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str, request: Request) -> dict[str, Any]:
    return services(request).store.get_project(project_id)


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request):
    services(request).store.delete_project(project_id)
    return None


@app.patch("/api/projects/{project_id}/settings")
def update_project_settings(project_id: str, payload: ProjectSettingsUpdate, request: Request) -> dict[str, Any]:
    return services(request).store.update_settings(project_id, payload.model_dump(exclude_none=True))


@app.post("/api/projects/{project_id}/scan")
def scan_project_source(project_id: str, request: Request) -> dict[str, Any]:
    return services(request).store.scan_source(project_id)


@app.post("/api/projects/{project_id}/chapters/upload")
async def upload_chapters(project_id: str, request: Request, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    store, imported = services(request).store, []
    for file in files:
        imported.append(store.import_upload(project_id, file.filename or "chapter.txt", await file.read()))
    return {"chapters": imported, "project": store.get_project(project_id)}


@app.get("/api/projects/{project_id}/chapters/{chapter_id}")
def get_chapter(project_id: str, chapter_id: str, request: Request) -> dict[str, Any]:
    return services(request).store.get_chapter(project_id, chapter_id)


@app.patch("/api/projects/{project_id}/chapters/{chapter_id}/segments/{segment_id}")
def update_segment(project_id: str, chapter_id: str, segment_id: str, payload: SegmentUpdate, request: Request) -> dict[str, Any]:
    return services(request).store.update_segment(project_id, chapter_id, segment_id, payload.model_dump(exclude_none=True))


@app.post("/api/projects/{project_id}/chapters/{chapter_id}/confirm/{stage}")
def confirm_stage(project_id: str, chapter_id: str, stage: str, request: Request) -> dict[str, Any]:
    if stage not in {"source", "script", "voices", "generate", "qa"}:
        raise HTTPException(status_code=404, detail="Unknown stage")
    chapter = services(request).store.get_chapter(project_id, chapter_id)
    if stage == "generate" and not chapter.get("audio_path"):
        raise ValueError("Audio has not been generated")
    return services(request).store.update_chapter(project_id, chapter_id, {"status": "done" if stage == "qa" else chapter.get("status", "ready"), "stages": {stage: "done"}})


@app.get("/api/projects/{project_id}/roles")
def list_roles(project_id: str, request: Request) -> dict[str, Any]:
    return {"roles": list(services(request).store.get_project(project_id).get("roles", {}).values())}


@app.patch("/api/projects/{project_id}/roles/{role_name}")
def update_role(project_id: str, role_name: str, payload: RoleUpdate, request: Request) -> dict[str, Any]:
    return services(request).store.update_role(project_id, role_name, payload.model_dump(exclude_none=True))


@app.post("/api/projects/{project_id}/roles/auto-match")
def auto_match_roles(project_id: str, payload: AutoMatchRequest, request: Request) -> dict[str, Any]:
    return services(request).matcher.auto_match(project_id, overwrite_unlocked=payload.overwrite_unlocked)


@app.get("/api/voices")
def list_voices(request: Request) -> dict[str, Any]:
    return {"voices": services(request).voices.list()}


@app.post("/api/voices/upload", status_code=201)
async def upload_voice(request: Request, file: UploadFile = File(...), name: str | None = Form(default=None)) -> dict[str, Any]:
    return services(request).voices.save_upload(file.filename or "voice.wav", await file.read(), name)


@app.post("/api/projects/{project_id}/jobs", status_code=202)
def create_job(project_id: str, payload: JobCreate, request: Request) -> dict[str, Any]:
    svc = services(request)
    return svc.jobs.enqueue(project_id, payload.model_dump(exclude={"api_key"}), api_key=payload.api_key.get_secret_value() if payload.api_key else svc.settings.api_key)


@app.get("/api/projects/{project_id}/jobs")
def list_jobs(project_id: str, request: Request) -> dict[str, Any]:
    return {"jobs": services(request).store.list_jobs(project_id)}


@app.get("/api/projects/{project_id}/jobs/{job_id}")
def get_job(project_id: str, job_id: str, request: Request) -> dict[str, Any]:
    return services(request).store.get_job(project_id, job_id)


@app.post("/api/projects/{project_id}/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str, request: Request) -> dict[str, Any]:
    return services(request).jobs.cancel(project_id, job_id)


@app.get("/api/projects/{project_id}/chapters/{chapter_id}/audio")
def chapter_audio(project_id: str, chapter_id: str, request: Request):
    chapter = services(request).store.get_chapter(project_id, chapter_id)
    path_value = chapter.get("mp3_path") or chapter.get("audio_path")
    if not path_value:
        raise NotFoundError("Chapter audio is not available")
    path = Path(path_value)
    if not path.exists():
        raise NotFoundError("Chapter audio file is missing")
    return FileResponse(path, filename=path.name)


settings_for_mount = load_settings()
settings_for_mount.data_root.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(settings_for_mount.data_root)), name="media")
if settings_for_mount.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(settings_for_mount.frontend_dir), html=True), name="frontend")


def run() -> None:
    import uvicorn
    settings = load_settings()
    uvicorn.run("audiobook_server.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
