# VoiceBook Studio Backend

FastAPI backend for the two-level VoiceBook Studio frontend:

- one project page manages chapters, project roles, batch execution, settings, and export state;
- one chapter page validates source, script, voices, generation, and final audio.

The backend reuses the repository's existing IndexTTS implementation through a lazy adapter. The API can start without model weights; only synthesis endpoints require a ready model directory.

## Run

From the repository root on Windows:

```bat
pip install -r app\requirements-audiobook.txt
start_audiobook_studio.bat
```

On macOS/Linux:

```bash
pip install -r app/requirements-audiobook.txt
./start_audiobook_studio.sh
```

Open `http://127.0.0.1:7861`.

## Configuration

- `OPENAI_BASE_URL`: OpenAI-compatible base URL.
- `OPENAI_API_KEY`: default API key. Keys entered in the UI are sent only with the job and are not persisted.
- `VOICEBOOK_PREPROCESS_MODEL`: free-form model ID, default `gpt-5.6-luna`.
- `VOICEBOOK_DATA_ROOT`: managed project directory.
- `VOICEBOOK_VOICE_LIBRARY`: reference voice directory. File names are used as AI matching tags.
- `INDEXTTS_MODEL_DIR`: IndexTTS checkpoint directory.
- `VOICEBOOK_HOST` / `VOICEBOOK_PORT`: server bind address.

## Data layout

```text
projects/<project_id>/
  project.json
  chapters/<chapter_id>/
    source.txt
    chapter.json
    script.json
    dialogue.txt
    audio_cache/
    outputs/<run_id>/
      manifest.json
      <chapter>.wav
      <chapter>.mp3
  jobs/<job_id>.json
```

Only Project, Chapter, Role, Segment, and Job are API entities. Chunks, cached WAV files, and manifests remain chapter-internal artifacts.

## Production notes

- The built-in queue is a single-process serial worker, matching the requested local desktop workflow.
- For multi-machine deployment, replace `JobManager` with Redis/Celery or another durable queue while retaining the service interfaces.
- Project JSON writes are atomic, but this MVP is intended for a single local operator rather than concurrent multi-user editing.
