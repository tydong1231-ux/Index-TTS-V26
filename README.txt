# IndexTTS Windows launcher (source-only)

This repository contains the Windows launcher, support tools, and a frozen copy of the IndexTTS application source used by this package. It intentionally excludes model weights, Hugging Face caches, the bundled Python/CUDA runtime, generated audio, and other large binaries.

## VoiceBook Studio

This branch also includes a project-based audiobook production application:

- `frontend-prototype/`: the confirmed two-level project/chapter frontend.
- `app/audiobook_server/`: FastAPI project, LLM preprocessing, voice matching, job queue, and IndexTTS orchestration backend.
- `start_audiobook_studio.bat` / `start_audiobook_studio.sh`: launch the integrated frontend and API at `http://127.0.0.1:7861`.
- `docs/AUDIOBOOK_STUDIO_API.md`: API and data-flow documentation.

Install the lightweight server dependencies after the existing IndexTTS environment:

```bash
pip install -r app/requirements-audiobook.txt
```

The server can start before model weights are installed. Project management and LLM preprocessing remain available; synthesis reports the exact missing model files until `INDEXTTS_MODEL_DIR` is ready.

## Upstream project and license

IndexTTS is maintained upstream at https://github.com/index-tts/index-tts. The included `app/LICENSE` is the upstream **bilibili Model Use License Agreement** and applies to the included project material and model use. Please read it before use. This repository is an unofficial Windows packaging/launcher project, not an official IndexTTS distribution.

## What's included

- `app/`: IndexTTS source and documentation
- `tools/windows_local/`: Windows packaging, support, and benchmark scripts
- `start_webui_auto.bat`, `start_webui_small_gpu.bat`: original IndexTTS launchers
- `start_audiobook_studio.bat`, `start_audiobook_studio.sh`: VoiceBook Studio launchers
- `check_support.bat`: environment compatibility check

## Required local files

Before TTS synthesis can run, obtain the required resources from the official project:

1. Download IndexTTS model checkpoints from https://huggingface.co/IndexTeam/IndexTTS-2 or https://modelscope.cn/models/IndexTeam/IndexTTS-2, then place them in the model directory used by `INDEXTTS_MODEL_DIR`.
2. Set up a compatible Python/CUDA environment and required dependencies according to `app/docs/README_zh.md` (or the upstream README).
3. Run `start_audiobook_studio.bat` for the novel workflow, or the original WebUI launchers for the upstream interface.

## Git hygiene

`.gitignore` deliberately prevents model files, caches, runtime binaries, generated audio, logs, and archives from being committed. Do not add them to Git history or Git LFS without confirming their upstream license and distribution terms.
