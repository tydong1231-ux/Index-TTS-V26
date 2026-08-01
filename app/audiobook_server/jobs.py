from __future__ import annotations

import queue
import threading
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from .llm import LLMConfig
from .preprocess import PreprocessService
from .store import ProjectStore
from .tts import TTSService
from .utils import utc_now
from .voices import VoiceMatcher


@dataclass
class QueuedJob:
    project_id: str
    job_id: str
    request: dict[str, Any]
    api_key: str


class JobManager:
    def __init__(self, store: ProjectStore, preprocess: PreprocessService, matcher: VoiceMatcher, tts: TTSService):
        self.store, self.preprocess, self.matcher, self.tts = store, preprocess, matcher, tts
        self._queue: queue.Queue[QueuedJob | None] = queue.Queue()
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._worker = threading.Thread(target=self._run, name="voicebook-job-worker", daemon=True)
        self._worker.start()

    def enqueue(self, project_id: str, request: dict[str, Any], api_key: str = "") -> dict[str, Any]:
        project = self.store.get_project(project_id)
        valid_ids = {chapter["id"] for chapter in project.get("chapters", [])}
        chapter_ids = [chapter_id for chapter_id in request.get("chapter_ids", []) if chapter_id in valid_ids]
        if not chapter_ids:
            raise ValueError("No valid chapter IDs were selected")
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        sanitized_request = dict(request)
        sanitized_request.pop("api_key", None)
        job = {"id": job_id, "project_id": project_id, "status": "queued", "created_at": utc_now(), "updated_at": utc_now(), "started_at": None, "completed_at": None, "request": sanitized_request, "chapter_ids": chapter_ids, "current_chapter_id": None, "current_stage": None, "progress": 0.0, "message": "Queued", "logs": [], "error": ""}
        self.store.save_job(project_id, job)
        self._queue.put(QueuedJob(project_id, job_id, {**request, "chapter_ids": chapter_ids}, api_key))
        return job

    def cancel(self, project_id: str, job_id: str) -> dict[str, Any]:
        with self._lock:
            self._cancelled.add(job_id)
        job = self.store.get_job(project_id, job_id)
        if job.get("status") in {"queued", "running"}:
            job.update(status="cancelling", message="Cancellation requested", updated_at=utc_now())
            self.store.save_job(project_id, job)
        return job

    def shutdown(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=3)

    def _run(self) -> None:
        while True:
            queued = self._queue.get()
            if queued is None:
                return
            try:
                self._execute(queued)
            except Exception:
                traceback.print_exc()
            finally:
                self._queue.task_done()

    def _execute(self, queued: QueuedJob) -> None:
        job = self.store.get_job(queued.project_id, queued.job_id)
        job.update(status="running", started_at=utc_now(), updated_at=utc_now(), message="Started")
        self.store.save_job(queued.project_id, job)
        stages = queued.request.get("stages") or ["script", "voices", "generate", "qa"]
        total_units, completed_units, matched_project = max(1, len(queued.request["chapter_ids"]) * len(stages)), 0, False
        try:
            for chapter_id in queued.request["chapter_ids"]:
                if self._is_cancelled(queued.job_id):
                    raise JobCancelled()
                for stage in stages:
                    if self._is_cancelled(queued.job_id):
                        raise JobCancelled()
                    job = self.store.get_job(queued.project_id, queued.job_id)
                    job.update(current_chapter_id=chapter_id, current_stage=stage, message=f"Running {stage}", updated_at=utc_now())
                    self.store.save_job(queued.project_id, job)
                    self._mark_stage(queued.project_id, chapter_id, stage, "running")

                    def report(stage_progress: float, message: str) -> None:
                        current = self.store.get_job(queued.project_id, queued.job_id)
                        current.update(progress=min(0.999, (completed_units + max(0.0, min(1.0, stage_progress))) / total_units), message=message, updated_at=utc_now())
                        self.store.save_job(queued.project_id, current)

                    if stage == "source":
                        self._mark_stage(queued.project_id, chapter_id, stage, "done")
                    elif stage == "script":
                        project = self.store.get_project(queued.project_id)
                        settings = project.get("settings", {})
                        llm_config = LLMConfig(api_base_url=str(queued.request.get("api_base_url") or settings.get("api_base_url") or "").rstrip("/"), api_key=queued.api_key, model=str(queued.request.get("model") or settings.get("model") or ""), endpoint=str(queued.request.get("endpoint") or "chat_completions"), reasoning_effort=str(queued.request.get("reasoning_effort") or "low"))
                        self.preprocess.preprocess_chapter(queued.project_id, chapter_id, llm_config, force=bool(queued.request.get("force")), progress=report)
                    elif stage == "voices":
                        if not matched_project:
                            self.matcher.auto_match(queued.project_id, overwrite_unlocked=False)
                            matched_project = True
                        project = self.store.get_project(queued.project_id)
                        ready = all(role.get("voice") for role in project.get("roles", {}).values())
                        self._mark_stage(queued.project_id, chapter_id, stage, "done" if ready else "attention")
                    elif stage == "generate":
                        project = self.store.get_project(queued.project_id)
                        mode = str(queued.request.get("generation_mode") or project.get("settings", {}).get("generation_mode") or "ordered")
                        self.tts.synthesize_chapter(queued.project_id, chapter_id, mode=mode, force=bool(queued.request.get("force")), progress=report)
                    elif stage == "qa":
                        chapter = self.store.get_chapter(queued.project_id, chapter_id)
                        if chapter.get("stages", {}).get("generate") != "done":
                            raise ValueError("QA cannot be completed before audio generation")
                        self.store.update_chapter(queued.project_id, chapter_id, {"status": "done", "stages": {"qa": "done"}})
                    completed_units += 1
                    job = self.store.get_job(queued.project_id, queued.job_id)
                    job["progress"] = completed_units / total_units
                    job.setdefault("logs", []).append({"time": utc_now(), "chapter_id": chapter_id, "stage": stage, "status": "done"})
                    job["updated_at"] = utc_now()
                    self.store.save_job(queued.project_id, job)
            job = self.store.get_job(queued.project_id, queued.job_id)
            job.update(status="completed", progress=1.0, message="Completed", current_chapter_id=None, current_stage=None, completed_at=utc_now(), updated_at=utc_now())
            self.store.save_job(queued.project_id, job)
        except JobCancelled:
            job = self.store.get_job(queued.project_id, queued.job_id)
            job.update(status="cancelled", message="Cancelled", completed_at=utc_now(), updated_at=utc_now())
            self.store.save_job(queued.project_id, job)
        except Exception as exc:
            if job.get("current_chapter_id") and job.get("current_stage"):
                self._mark_stage(queued.project_id, job["current_chapter_id"], job["current_stage"], "error")
            job = self.store.get_job(queued.project_id, queued.job_id)
            job.update(status="failed", message=str(exc), error=f"{type(exc).__name__}: {exc}", completed_at=utc_now(), updated_at=utc_now())
            job.setdefault("logs", []).append({"time": utc_now(), "status": "failed", "message": str(exc)})
            self.store.save_job(queued.project_id, job)
        finally:
            with self._lock:
                self._cancelled.discard(queued.job_id)

    def _mark_stage(self, project_id: str, chapter_id: str, stage: str, status: str) -> None:
        updates: dict[str, Any] = {"stages": {stage: status}}
        if status == "running":
            updates["status"] = "running"
        elif status == "error":
            updates["status"] = "attention"
        self.store.update_chapter(project_id, chapter_id, updates)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled


class JobCancelled(Exception):
    pass
