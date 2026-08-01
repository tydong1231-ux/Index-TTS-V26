from pathlib import Path

from audiobook_server.config import Settings
from audiobook_server.store import ProjectStore
from audiobook_server.voices import VoiceLibrary, VoiceMatcher


def make_settings(tmp_path: Path) -> Settings:
    repo = tmp_path / "repo"
    app = repo / "app"
    frontend = repo / "frontend-prototype"
    data = repo / "projects"
    voices = app / "prompts" / "library"
    model = app / "checkpoints"
    for path in (frontend, data, voices, model):
        path.mkdir(parents=True, exist_ok=True)
    return Settings(repo_root=repo, data_root=data, frontend_dir=frontend, model_dir=model, voice_library_dir=voices, api_base_url="https://api.openai.com", api_key="", preprocess_model="gpt-5.6-luna", host="127.0.0.1", port=7861, use_fp16=False, use_deepspeed=False, use_cuda_kernel=False)


def test_project_folder_import(tmp_path: Path):
    settings = make_settings(tmp_path)
    source = tmp_path / "novel"
    source.mkdir()
    (source / "第2章.txt").write_text("第二章", encoding="utf-8")
    (source / "第10章.txt").write_text("第十章", encoding="utf-8")
    store = ProjectStore(settings)
    project = store.create_project("测试小说", str(source))
    assert len(project["chapters"]) == 2
    assert project["chapters"][0]["source_chars"] > 0
    assert project["chapters"][0]["stages"]["source"] == "done"


def test_voice_matcher_avoids_reuse(tmp_path: Path):
    settings = make_settings(tmp_path)
    for name in ("青年男声_01.wav", "青年男声_02.wav", "御姐音_01.wav"):
        (settings.voice_library_dir / name).write_bytes(b"not-a-real-wave")
    store = ProjectStore(settings)
    project = store.create_project("测试小说")
    project["roles"] = {
        "林默": {"name": "林默", "count": 100, "gender": "m", "age_stage": "young", "traits": ["冷静"], "aliases": [], "voice": "", "voice_locked": False, "voice_score": None, "voice_reason": "", "note": ""},
        "列车员": {"name": "列车员", "count": 40, "gender": "m", "age_stage": "young", "traits": [], "aliases": [], "voice": "", "voice_locked": False, "voice_score": None, "voice_reason": "", "note": ""},
        "周岚": {"name": "周岚", "count": 90, "gender": "f", "age_stage": "young", "traits": ["冷静"], "aliases": [], "voice": "", "voice_locked": False, "voice_score": None, "voice_reason": "", "note": ""},
    }
    store.save_project(project)
    matched = VoiceMatcher(store, VoiceLibrary(settings)).auto_match(project["id"])
    voices = [role["voice"] for role in matched["roles"].values()]
    assert len(set(voices)) == 3
    assert matched["roles"]["周岚"]["voice"].startswith("御姐音")
