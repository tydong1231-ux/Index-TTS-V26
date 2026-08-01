# IndexTTS Windows launcher (source-only)

This repository contains the Windows launcher, support tools, and a frozen copy of the IndexTTS application source used by this package.  It intentionally excludes model weights, Hugging Face caches, the bundled Python/CUDA runtime, generated audio, and other large binaries.

## Upstream project and license

IndexTTS is maintained upstream at https://github.com/index-tts/index-tts.  The included `app/LICENSE` is the upstream **bilibili Model Use License Agreement** and applies to the included project material and model use.  Please read it before use. This repository is an unofficial Windows packaging/launcher project, not an official IndexTTS distribution.

## What's included

- `app/`: IndexTTS source and documentation
- `tools/windows_local/`: Windows packaging, support, and benchmark scripts
- `start_webui_auto.bat`, `start_webui_small_gpu.bat`: launchers
- `check_support.bat`: environment compatibility check

## Required local files

Before the launchers can run, obtain the required resources from the official project:

1. Download IndexTTS model checkpoints from https://huggingface.co/IndexTeam/IndexTTS-2 or https://modelscope.cn/models/IndexTeam/IndexTTS-2, then place them in `data/checkpoints/`.
2. Set up a compatible Python/CUDA environment and required dependencies according to `app/docs/README_zh.md` (or the upstream README).
3. Run `start_webui_auto.bat`; use `start_webui_small_gpu.bat` when appropriate.

## Git hygiene

`.gitignore` deliberately prevents model files, caches, runtime binaries, generated audio, logs, and archives from being committed. Do not add them to Git history or Git LFS without confirming their upstream license and distribution terms.
