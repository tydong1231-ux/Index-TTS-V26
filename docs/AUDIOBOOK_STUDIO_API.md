# VoiceBook Studio API

Base URL: `http://127.0.0.1:7861`

## Main flow

1. `POST /api/projects` creates a project and optionally scans a server-accessible folder of TXT files.
2. `POST /api/projects/{id}/chapters/upload` imports TXT files selected in the browser.
3. `POST /api/projects/{id}/jobs` runs selected chapters serially through any combination of `script`, `voices`, `generate`, and `qa`.
4. `GET /api/projects/{id}/jobs/{job_id}` polls progress.
5. The chapter page reads `GET /api/projects/{id}/chapters/{chapter_id}` and edits a segment with `PATCH .../segments/{segment_id}`.

## Project endpoints

- `GET /api/projects`
- `POST /api/projects`
- `GET /api/projects/{project_id}`
- `PATCH /api/projects/{project_id}/settings`
- `POST /api/projects/{project_id}/scan`
- `POST /api/projects/{project_id}/chapters/upload`

Create project body:

```json
{
  "name": "雾城来信",
  "source_path": "D:\\Voicebook\\雾城来信",
  "api_base_url": "https://api.openai.com",
  "model": "gpt-5.6-luna"
}
```

`source_path` is optional. When omitted, chapters can be uploaded from the frontend.

## Jobs

```json
POST /api/projects/{project_id}/jobs
{
  "chapter_ids": ["0001_chapter_xxx", "0002_chapter_xxx"],
  "stages": ["script", "voices", "generate", "qa"],
  "api_base_url": "https://api.openai.com",
  "api_key": "sk-...",
  "model": "gpt-5.6-luna",
  "endpoint": "chat_completions",
  "reasoning_effort": "low",
  "generation_mode": "ordered",
  "force": false
}
```

The API key is retained only in process memory for the queued job and is excluded from persisted job JSON.

Generation modes:

- `ordered`: generate in original segment order.
- `grouped`: generate consecutive segments by role to reuse IndexTTS speaker conditioning, while still writing one WAV per segment and reassembling by the original manifest order.

## Roles and voices

- `GET /api/voices`
- `POST /api/voices/upload`
- `GET /api/projects/{project_id}/roles`
- `PATCH /api/projects/{project_id}/roles/{role_name}`
- `POST /api/projects/{project_id}/roles/auto-match`

Voice matching scores role gender, age stage, narrative function, and character traits. Already-assigned voices receive a reuse penalty so different characters prefer different voices.

## Runtime status

`GET /api/health` always works even when IndexTTS weights are absent. It returns missing checkpoint file names and whether the model has been loaded.
