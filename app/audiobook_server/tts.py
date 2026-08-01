from __future__ import annotations

import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .store import ProjectStore
from .utils import atomic_write_json, hash_parts, utc_now
from .voices import VoiceLibrary

EMOTION_VECTORS = {"高兴": [0.8,0,0,0,0,0,0,0], "愤怒": [0,0.8,0,0,0,0,0,0], "悲伤": [0,0,0.8,0,0,0,0,0], "恐惧": [0,0,0,0.8,0,0,0,0], "厌恶": [0,0,0,0,0.8,0,0,0], "低落": [0,0,0,0,0,0.8,0,0], "惊喜": [0,0,0,0,0,0,0.8,0], "平静": [0,0,0,0,0,0,0,0.8]}


class TTSUnavailableError(RuntimeError):
    pass


class IndexTTSAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        required = ["config.yaml", "bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt"]
        missing = [name for name in required if not (self.settings.model_dir / name).exists()]
        return {"ready": not missing, "loaded": self._model is not None, "model_dir": str(self.settings.model_dir), "missing_files": missing, "fp16": self.settings.use_fp16, "deepspeed": self.settings.use_deepspeed, "cuda_kernel": self.settings.use_cuda_kernel}

    def infer(self, voice_path: Path, text: str, emotion: str, output_path: Path, max_tokens: int) -> Path:
        model = self._get_model()
        vector = EMOTION_VECTORS.get(emotion)
        if vector is not None:
            vector = model.normalize_emo_vec(vector, apply_bias=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._infer_lock:
            result = model.infer(spk_audio_prompt=str(voice_path), text=text, output_path=str(output_path), emo_audio_prompt=None, emo_alpha=0.65, emo_vector=vector, use_emo_text=False, emo_text=None, use_random=False, verbose=False, max_text_tokens_per_segment=max_tokens, do_sample=True, top_p=0.8, top_k=30, temperature=0.8, length_penalty=0.0, num_beams=3, repetition_penalty=10.0, max_mel_tokens=1500)
        resolved = Path(str(result or output_path))
        if not resolved.exists():
            raise RuntimeError(f"IndexTTS did not create output: {output_path}")
        return resolved

    def _get_model(self):
        status = self.status()
        if not status["ready"]:
            raise TTSUnavailableError("IndexTTS model is not ready. Missing: " + ", ".join(status["missing_files"]))
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from indextts.infer_v2 import IndexTTS2
                    self._model = IndexTTS2(model_dir=str(self.settings.model_dir), cfg_path=str(self.settings.model_dir / "config.yaml"), use_fp16=self.settings.use_fp16, use_deepspeed=self.settings.use_deepspeed, use_cuda_kernel=self.settings.use_cuda_kernel)
        return self._model


class TTSService:
    def __init__(self, settings: Settings, store: ProjectStore, voices: VoiceLibrary, adapter: IndexTTSAdapter):
        self.settings, self.store, self.voices, self.adapter = settings, store, voices, adapter

    def synthesize_chapter(self, project_id: str, chapter_id: str, mode: str = "ordered", force: bool = False, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
        project, chapter = self.store.get_project(project_id), self.store.get_chapter(project_id, chapter_id)
        script = chapter.get("script") or self.store.get_script(project_id, chapter_id)
        segments = list(script.get("segments", []))
        if not segments:
            raise ValueError("Chapter has no script segments")
        role_map, resolved_voices, missing = project.get("roles", {}), {}, []
        for segment in segments:
            speaker = str(segment.get("speaker") or "旁白")
            if speaker in resolved_voices:
                continue
            voice_path = self.voices.resolve(str((role_map.get(speaker) or {}).get("voice") or ""))
            if voice_path is None:
                missing.append(speaker)
            else:
                resolved_voices[speaker] = voice_path
        if missing:
            raise ValueError("Roles without a usable voice: " + "、".join(sorted(set(missing))))
        settings = project.get("settings", {})
        interval_ms, max_tokens = int(settings.get("interval_ms", 450)), int(settings.get("max_text_tokens_per_segment", 120))
        generation_order = list(segments)
        if mode == "grouped":
            generation_order.sort(key=lambda item: (str(item.get("speaker", "")), int(item.get("order", 0))))
        chapter_root = self.store.chapter_dir(project_id, chapter_id)
        cache_dir = chapter_root / "audio_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        run_dir = chapter_root / "outputs" / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        generated_by_id, cache_hits = {}, {}
        for position, segment in enumerate(generation_order, 1):
            segment_id, speaker = str(segment.get("id") or f"{position:04d}"), str(segment.get("speaker") or "旁白")
            voice_path, emotion = resolved_voices[speaker], str(segment.get("emotion") or "默认")
            fingerprint = hash_parts([str(voice_path), str(voice_path.stat().st_mtime_ns), str(segment.get("text") or ""), emotion, str(max_tokens)])
            cached = cache_dir / f"{fingerprint}.wav"
            if progress:
                progress((position - 1) / len(generation_order), f"{speaker} {position}/{len(generation_order)}")
            if force or not cached.exists():
                self.adapter.infer(voice_path, str(segment.get("text") or ""), emotion, cached, max_tokens)
                cache_hits[segment_id] = False
            else:
                cache_hits[segment_id] = True
            generated_by_id[segment_id] = cached
        ordered = sorted(segments, key=lambda item: int(item.get("order", 0)))
        ordered_paths = [generated_by_id[str(segment.get("id"))] for segment in ordered]
        final_name = f"{int(chapter.get('index', 0)):04d}_{chapter.get('title', chapter_id)}"
        final_wav = run_dir / f"{final_name}.wav"
        combine_wav_files(ordered_paths, final_wav, interval_ms)
        final_mp3, duration = convert_to_mp3(final_wav), wav_duration(final_wav)
        manifest = []
        for segment in ordered:
            segment_id = str(segment.get("id"))
            manifest.append({"id": segment_id, "order": segment.get("order"), "speaker": segment.get("speaker"), "emotion": segment.get("emotion"), "text": segment.get("text"), "voice": str((role_map.get(str(segment.get("speaker"))) or {}).get("voice") or ""), "audio_path": str(generated_by_id[segment_id]), "cache_hit": cache_hits[segment_id]})
        atomic_write_json(run_dir / "manifest.json", {"project_id": project_id, "chapter_id": chapter_id, "mode": mode, "created_at": utc_now(), "final_wav": str(final_wav), "final_mp3": str(final_mp3) if final_mp3 else "", "segments": manifest})
        self.store.update_chapter(project_id, chapter_id, {"status": "done", "audio_path": str(final_wav), "mp3_path": str(final_mp3) if final_mp3 else "", "duration_seconds": duration, "stages": {"generate": "done", "qa": "pending"}})
        if progress:
            progress(1.0, "Chapter synthesis complete")
        return {"audio_path": str(final_mp3 or final_wav), "wav_path": str(final_wav), "mp3_path": str(final_mp3) if final_mp3 else "", "duration_seconds": duration, "manifest_path": str(run_dir / "manifest.json"), "cache_hits": sum(1 for value in cache_hits.values() if value), "segment_count": len(segments)}


def combine_wav_files(paths: list[Path], output_path: Path, interval_ms: int = 450) -> None:
    if not paths:
        raise ValueError("No audio segments were generated")
    with wave.open(str(paths[0]), "rb") as first:
        params, channels, sample_width, frame_rate = first.getparams(), first.getnchannels(), first.getsampwidth(), first.getframerate()
    silence = b"\x00" * int(frame_rate * max(0, interval_ms) / 1000) * channels * sample_width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as segment:
                if segment.getnchannels() != channels or segment.getsampwidth() != sample_width or segment.getframerate() != frame_rate:
                    raise ValueError(f"Audio format mismatch: {path}")
                output.writeframes(segment.readframes(segment.getnframes()))
            if index < len(paths) - 1 and silence:
                output.writeframes(silence)


def convert_to_mp3(wav_path: Path) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)], check=True)
    return mp3_path if mp3_path.exists() else None


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / float(audio.getframerate())
