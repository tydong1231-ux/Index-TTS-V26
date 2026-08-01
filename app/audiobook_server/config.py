from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_root: Path
    frontend_dir: Path
    model_dir: Path
    voice_library_dir: Path
    api_base_url: str
    api_key: str
    preprocess_model: str
    host: str
    port: int
    use_fp16: bool
    use_deepspeed: bool
    use_cuda_kernel: bool


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    app_root = repo_root / "app"
    data_root = Path(os.environ.get("VOICEBOOK_DATA_ROOT", repo_root / "projects")).expanduser().resolve()
    frontend_dir = Path(os.environ.get("VOICEBOOK_FRONTEND_DIR", repo_root / "frontend-prototype")).expanduser().resolve()
    model_dir = Path(os.environ.get("INDEXTTS_MODEL_DIR", app_root / "checkpoints")).expanduser().resolve()
    voice_library_dir = Path(
        os.environ.get("VOICEBOOK_VOICE_LIBRARY", app_root / "prompts" / "library")
    ).expanduser().resolve()
    return Settings(
        repo_root=repo_root,
        data_root=data_root,
        frontend_dir=frontend_dir,
        model_dir=model_dir,
        voice_library_dir=voice_library_dir,
        api_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/"),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        preprocess_model=os.environ.get("VOICEBOOK_PREPROCESS_MODEL", "gpt-5.6-luna"),
        host=os.environ.get("VOICEBOOK_HOST", "127.0.0.1"),
        port=int(os.environ.get("VOICEBOOK_PORT", "7861")),
        use_fp16=os.environ.get("INDEXTTS_FP16", "1") not in {"0", "false", "False"},
        use_deepspeed=os.environ.get("INDEXTTS_DEEPSPEED", "0") in {"1", "true", "True"},
        use_cuda_kernel=os.environ.get("INDEXTTS_CUDA_KERNEL", "1") not in {"0", "false", "False"},
    )
