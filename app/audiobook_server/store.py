from __future__ import annotations

import shutil
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .utils import atomic_write_json, atomic_write_text, natural_sort_key, read_json, read_text, safe_join, slugify, utc_now

STAGES = ("source", "script", "voices", "generate", "qa")


class NotFoundError(KeyError):
    pass


class ConflictError(RuntimeError):
    pass


class ProjectStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)

    def project_dir(self, project_id: str) -> Path:
        return safe_join(self.root, project_id)

    def project_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project.json"

    def chapter_dir(self, project_id: str, chapter_id: str) -> Path:
        return safe_join(self.project_dir(project_id) / "chapters", chapter_id)

    def chapter_script_path(self, project_id: str, chapter_id: str) -> Path:
        return self.chapter_dir(project_id, chapter_id) / "script.json"

    def job_path(self, project_id: str, job_id: str) -> Path:
        return safe_join(self.project_dir(project_id) / "jobs", f"{job_id}.json")

    def list_projects(self) -> list[dict[str, Any]]:
        projects = []
        for path in self.root.glob("*/project.json"):
            project = read_json(path, {})
            if project:
                projects.append(self._project_summary(project))
        return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get_project(self, project_id: str) -> dict[str, Any]:
        project = read_json(self.project_path(project_id))
        if not project:
            raise NotFoundError(f"Project not found: {project_id}")
        return project

    def save_project(self, project: dict[str, Any]) -> dict[str, Any]:
        project_id = project["id"]
        with self._locks[project_id]:
            project["updated_at"] = utc_now()
            atomic_write_json(self.project_path(project_id), project)
        return project

    def create_project(self, name: str, source_path: str | None = None, api_base_url: str | None = None, model: str | None = None) -> dict[str, Any]:
        project_id = f"{slugify(name, 'novel')}_{uuid.uuid4().hex[:10]}"
        root = self.project_dir(project_id)
        (root / "chapters").mkdir(parents=True, exist_ok=True)
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        now = utc_now()
        project = {
            "id": project_id, "name": name.strip(),
            "source_path": str(Path(source_path).expanduser().resolve()) if source_path else "",
            "created_at": now, "updated_at": now,
            "settings": {"api_base_url": (api_base_url or self.settings.api_base_url).rstrip("/"), "model": model or self.settings.preprocess_model, "chunk_target_chars": 500, "chunk_max_chars": 800, "interval_ms": 450, "max_text_tokens_per_segment": 120, "generation_mode": "ordered"},
            "roles": {}, "chapters": [],
        }
        self.save_project(project)
        if source_path:
            self.scan_source(project_id)
        return self.get_project(project_id)

    def update_settings(self, project_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        project = self.get_project(project_id)
        for key, value in updates.items():
            if value is not None:
                project.setdefault("settings", {})[key] = value
        return self.save_project(project)

    def delete_project(self, project_id: str) -> None:
        root = self.project_dir(project_id)
        if not root.exists():
            raise NotFoundError(project_id)
        shutil.rmtree(root)

    def scan_source(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        source = Path(project.get("source_path") or "").expanduser()
        if not source.exists():
            raise NotFoundError(f"Source path does not exist: {source}")
        files = [path for path in source.rglob("*.txt") if path.is_file()] if source.is_dir() else ([source] if source.suffix.lower() == ".txt" else [])
        files.sort(key=lambda path: natural_sort_key(str(path.relative_to(source) if source.is_dir() else path.name)))
        return self.import_files(project_id, files)

    def import_files(self, project_id: str, files: Iterable[Path]) -> dict[str, Any]:
        project = self.get_project(project_id)
        existing_by_source = {chapter.get("source_path"): chapter for chapter in project.get("chapters", [])}
        existing_by_name = {chapter.get("source_filename"): chapter for chapter in project.get("chapters", [])}
        chapters = list(project.get("chapters", []))
        for file_path in sorted(files, key=lambda path: natural_sort_key(path.name)):
            file_path = Path(file_path).expanduser().resolve()
            if file_path.suffix.lower() != ".txt" or not file_path.exists() or str(file_path) in existing_by_source:
                continue
            text, filename, base_title = read_text(file_path), file_path.name, file_path.stem
            if filename in existing_by_name:
                base_title = f"{base_title}_{uuid.uuid4().hex[:4]}"
            index = len(chapters) + 1
            chapter_id = f"{index:04d}_{slugify(base_title, 'chapter')}_{uuid.uuid4().hex[:8]}"
            chapter_root = self.chapter_dir(project_id, chapter_id)
            chapter_root.mkdir(parents=True, exist_ok=True)
            atomic_write_text(chapter_root / "source.txt", text)
            chapter = self._new_chapter(chapter_id, index, base_title, filename, str(file_path), len(text))
            atomic_write_json(chapter_root / "chapter.json", chapter)
            chapters.append(chapter)
            existing_by_source[str(file_path)] = chapter
            existing_by_name[filename] = chapter
        for index, chapter in enumerate(sorted(chapters, key=lambda item: natural_sort_key(item.get("title", ""))), 1):
            chapter["index"] = index
        project["chapters"] = chapters
        return self.save_project(project)

    def import_upload(self, project_id: str, filename: str, data: bytes) -> dict[str, Any]:
        project = self.get_project(project_id)
        if not filename.lower().endswith(".txt"):
            raise ValueError("Only .txt files are supported")
        title, index = Path(filename).stem, len(project.get("chapters", [])) + 1
        chapter_id = f"{index:04d}_{slugify(title, 'chapter')}_{uuid.uuid4().hex[:8]}"
        chapter_root = self.chapter_dir(project_id, chapter_id)
        chapter_root.mkdir(parents=True, exist_ok=True)
        text = self._decode_upload(data)
        atomic_write_text(chapter_root / "source.txt", text)
        chapter = self._new_chapter(chapter_id, index, title, filename, "", len(text))
        atomic_write_json(chapter_root / "chapter.json", chapter)
        project.setdefault("chapters", []).append(chapter)
        self.save_project(project)
        return chapter

    @staticmethod
    def _new_chapter(chapter_id: str, index: int, title: str, filename: str, source_path: str, size: int) -> dict[str, Any]:
        return {"id": chapter_id, "index": index, "title": title, "source_filename": filename, "source_path": source_path, "source_chars": size, "status": "new", "stages": {"source": "done", "script": "pending", "voices": "pending", "generate": "pending", "qa": "pending"}, "segment_count": 0, "role_count": 0, "warning_count": 0, "audio_path": "", "mp3_path": "", "duration_seconds": None, "updated_at": utc_now()}

    def get_chapter(self, project_id: str, chapter_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        for chapter in project.get("chapters", []):
            if chapter.get("id") == chapter_id:
                result = dict(chapter)
                result["source_text"] = read_text(self.chapter_dir(project_id, chapter_id) / "source.txt")
                result["script"] = read_json(self.chapter_script_path(project_id, chapter_id), {})
                return result
        raise NotFoundError(f"Chapter not found: {chapter_id}")

    def update_chapter(self, project_id: str, chapter_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        project = self.get_project(project_id)
        chapter = next((item for item in project.get("chapters", []) if item.get("id") == chapter_id), None)
        if chapter is None:
            raise NotFoundError(chapter_id)
        for key, value in updates.items():
            if key == "stages" and isinstance(value, dict):
                chapter.setdefault("stages", {}).update(value)
            else:
                chapter[key] = value
        chapter["updated_at"] = utc_now()
        atomic_write_json(self.chapter_dir(project_id, chapter_id) / "chapter.json", chapter)
        self.save_project(project)
        return chapter

    def get_script(self, project_id: str, chapter_id: str) -> dict[str, Any]:
        script = read_json(self.chapter_script_path(project_id, chapter_id))
        if not script:
            raise NotFoundError("Chapter script has not been created")
        return script

    def save_script(self, project_id: str, chapter_id: str, script: dict[str, Any]) -> dict[str, Any]:
        atomic_write_json(self.chapter_script_path(project_id, chapter_id), script)
        atomic_write_text(self.chapter_dir(project_id, chapter_id) / "dialogue.txt", "\n".join(f"{line.get('speaker', '旁白')}：{line.get('text', '')}" for line in script.get("segments", [])))
        return script

    def update_segment(self, project_id: str, chapter_id: str, segment_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        script = self.get_script(project_id, chapter_id)
        segment = next((item for item in script.get("segments", []) if str(item.get("id")) == str(segment_id)), None)
        if segment is None:
            raise NotFoundError(segment_id)
        for key, value in updates.items():
            if value is not None:
                segment[key] = value
        script["updated_at"] = utc_now()
        self.save_script(project_id, chapter_id, script)
        self.sync_roles(project_id)
        return segment

    def update_role(self, project_id: str, role_name: str, updates: dict[str, Any]) -> dict[str, Any]:
        project = self.get_project(project_id)
        role = project.setdefault("roles", {}).setdefault(role_name, {"name": role_name, "count": 0, "gender": "u", "age_stage": "adult", "traits": [], "aliases": [], "voice": "", "voice_locked": False, "voice_score": None, "voice_reason": "", "note": ""})
        for key, value in updates.items():
            if value is not None:
                role[key] = value
        self.save_project(project)
        return role

    def sync_roles(self, project_id: str) -> dict[str, Any]:
        project, counts, hints = self.get_project(project_id), {}, {}
        current = project.get("roles", {})
        for chapter in project.get("chapters", []):
            script = read_json(self.chapter_script_path(project_id, chapter["id"]), {})
            for segment in script.get("segments", []):
                name = str(segment.get("speaker") or "旁白").strip() or "旁白"
                counts[name] = counts.get(name, 0) + 1
            for role in script.get("roles", []):
                name = str(role.get("name") or "").strip()
                if name:
                    hints[name] = role
        roles = {}
        for name, count in counts.items():
            old, hint = current.get(name, {}), hints.get(name, {})
            roles[name] = {"name": name, "count": count, "gender": old.get("gender") or hint.get("gender") or "u", "age_stage": old.get("age_stage") or hint.get("age_stage") or "adult", "traits": old.get("traits") or hint.get("traits") or [], "aliases": old.get("aliases") or hint.get("aliases") or [], "voice": old.get("voice", ""), "voice_locked": bool(old.get("voice_locked", False)), "voice_score": old.get("voice_score"), "voice_reason": old.get("voice_reason", ""), "note": old.get("note") or hint.get("note") or ""}
        project["roles"] = dict(sorted(roles.items(), key=lambda item: (-item[1]["count"], item[0])))
        self.save_project(project)
        return project

    def save_job(self, project_id: str, job: dict[str, Any]) -> dict[str, Any]:
        atomic_write_json(self.job_path(project_id, job["id"]), job)
        return job

    def get_job(self, project_id: str, job_id: str) -> dict[str, Any]:
        job = read_json(self.job_path(project_id, job_id))
        if not job:
            raise NotFoundError(job_id)
        return job

    def list_jobs(self, project_id: str) -> list[dict[str, Any]]:
        jobs = [read_json(path, {}) for path in (self.project_dir(project_id) / "jobs").glob("*.json")]
        return sorted([job for job in jobs if job], key=lambda item: item.get("created_at", ""), reverse=True)

    @staticmethod
    def _decode_upload(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _project_summary(project: dict[str, Any]) -> dict[str, Any]:
        chapters = project.get("chapters", [])
        return {"id": project.get("id"), "name": project.get("name"), "source_path": project.get("source_path", ""), "created_at": project.get("created_at"), "updated_at": project.get("updated_at"), "chapter_count": len(chapters), "completed_chapters": sum(1 for chapter in chapters if chapter.get("status") == "done"), "attention_chapters": sum(1 for chapter in chapters if chapter.get("status") == "attention"), "role_count": len(project.get("roles", {})), "confirmed_roles": sum(1 for role in project.get("roles", {}).values() if role.get("voice"))}
