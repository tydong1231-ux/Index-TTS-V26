import html
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*GPT2InferenceModel has generative capabilities.*",
    category=UserWarning,
)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
import indextts.gpt.transformers_modeling_utils as transformers_modeling_utils
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

# Silence the upstream GenerationMixin advisory during app startup.
transformers_modeling_utils.logger.warning_once = lambda *args, **kwargs: None

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features

os.makedirs(os.path.join(ROOT_DIR, "outputs", "tasks"), exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def format_glossary_markdown():
    """将词汇表转换为Markdown表格格式"""
    if not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join(ROOT_DIR, "outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(
        f"<div class='premium-warning'>"
        f"<span class='premium-warning__dot'></span>"
        f"<span class='premium-warning__text'>{html.escape(warning_text)}</span>"
        f"</div>"
    )

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))


# ------------------------------------------------------------------
# Voice Library (shared between speaker-ref and emotion-ref slots)
# storage: ./prompts/library/<name>.<ext>
# ------------------------------------------------------------------
VOICE_LIB_DIR = os.path.join("prompts", "library")
VOICE_LIB_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
os.makedirs(VOICE_LIB_DIR, exist_ok=True)


def _safe_voice_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '', name)
    return name[:60]


def voice_lib_list():
    if not os.path.isdir(VOICE_LIB_DIR):
        return []
    items = []
    for fn in sorted(os.listdir(VOICE_LIB_DIR)):
        if fn.lower().endswith(VOICE_LIB_EXTS):
            items.append(os.path.splitext(fn)[0])
    return items


def voice_lib_path(name: str):
    safe = _safe_voice_name(name)
    if not safe:
        return None
    for ext in VOICE_LIB_EXTS:
        p = os.path.join(VOICE_LIB_DIR, safe + ext)
        if os.path.exists(p):
            return p
    return None


def _voice_lib_save(audio_path, name):
    """Returns (saved_name_or_None, choices_list). Emits toast."""
    if not audio_path:
        gr.Warning(i18n("请先上传或录制一段音频"))
        return None, voice_lib_list()
    safe = _safe_voice_name(name)
    if not safe:
        gr.Warning(i18n("请填写音色名称"))
        return None, voice_lib_list()
    ext = os.path.splitext(audio_path)[1].lower() or ".wav"
    if ext not in VOICE_LIB_EXTS:
        ext = ".wav"
    dst = os.path.join(VOICE_LIB_DIR, safe + ext)
    try:
        for old_ext in VOICE_LIB_EXTS:
            old = os.path.join(VOICE_LIB_DIR, safe + old_ext)
            if old_ext != ext and os.path.exists(old):
                os.remove(old)
        shutil.copy2(audio_path, dst)
        gr.Info(i18n("音色已保存：") + safe, duration=2)
    except Exception as e:
        gr.Error(i18n("保存失败：") + str(e))
        return None, voice_lib_list()
    return safe, voice_lib_list()


def on_save_to_library(audio_path, name):
    saved, choices = _voice_lib_save(audio_path, name)
    value = saved if saved else gr.update()
    return gr.update(choices=choices, value=value), gr.update(choices=choices)


def on_load_from_library(name):
    if not name:
        gr.Warning(i18n("请先从音色库选择一个音色"))
        return gr.update()
    p = voice_lib_path(name)
    if not p:
        gr.Warning(i18n("音色文件不存在，请刷新列表"))
        return gr.update()
    return gr.update(value=p)


def on_delete_from_library(name):
    if not name:
        gr.Warning(i18n("请先选择要删除的音色"))
        choices = voice_lib_list()
        return gr.update(choices=choices), gr.update(choices=choices)
    p = voice_lib_path(name)
    if p and os.path.exists(p):
        try:
            os.remove(p)
            gr.Info(i18n("已删除：") + name, duration=2)
        except Exception as e:
            gr.Error(i18n("删除失败：") + str(e))
    choices = voice_lib_list()
    return gr.update(choices=choices, value=None), gr.update(choices=choices, value=None)


def on_refresh_library():
    choices = voice_lib_list()
    return gr.update(choices=choices), gr.update(choices=choices)


# Multi-role dialogue generation.
DIALOGUE_ROLE_PATTERN = re.compile(r"^\s*([^:\uff1a]{1,32})\s*[:\uff1a]\s*(.+?)\s*$")
DIALOGUE_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "dialogue")
DIALOGUE_ROLE_SLOT_COUNT = 8
DIALOGUE_ROLE_SLOT_FIELD_COUNT = 3
DIALOGUE_EMOTION_CHOICES = [
    "默认",
    "高兴",
    "愤怒",
    "悲伤",
    "恐惧",
    "厌恶",
    "低落",
    "惊喜",
    "平静",
]
DIALOGUE_EMOTION_VECTORS = {
    "高兴": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "愤怒": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "悲伤": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    "恐惧": [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
    "厌恶": [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
    "低落": [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
    "惊喜": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
    "平静": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
}


def parse_dialogue_script(script: str):
    lines = []
    for raw_line in (script or "").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        match = DIALOGUE_ROLE_PATTERN.match(raw_line)
        if not match:
            continue
        role = match.group(1).strip()
        text = match.group(2).strip()
        if role and text:
            lines.append({"role": role, "text": text})
    return lines


def dialogue_preview_data(script: str):
    lines = parse_dialogue_script(script)
    return [[idx + 1, item["role"], item["text"]] for idx, item in enumerate(lines)]


def dialogue_roles(lines):
    roles = []
    seen = set()
    for item in lines:
        role = item["role"]
        if role not in seen:
            roles.append(role)
            seen.add(role)
    return roles


def resolve_role_voice(voice_ref: str):
    if not voice_ref:
        return None
    if os.path.exists(voice_ref):
        return voice_ref
    lib_path = voice_lib_path(voice_ref)
    if lib_path:
        return lib_path
    return None


def build_role_slots_from_script(script: str, *current_values):
    lines = parse_dialogue_script(script)
    preview = gr.update(value=dialogue_preview_data(script))
    updates = []
    if not lines:
        for idx in range(DIALOGUE_ROLE_SLOT_COUNT):
            base = idx * DIALOGUE_ROLE_SLOT_FIELD_COUNT
            role_value = current_values[base] if base < len(current_values) else ""
            voice_value = current_values[base + 1] if base + 1 < len(current_values) else None
            emotion_value = current_values[base + 2] if base + 2 < len(current_values) else "默认"
            updates.extend([
                gr.update(visible=False),
                gr.update(value=role_value or ""),
                gr.update(value=voice_value),
                gr.update(value=emotion_value or "默认"),
            ])
        return (*updates, preview)

    existing = {}
    for idx in range(DIALOGUE_ROLE_SLOT_COUNT):
        base = idx * DIALOGUE_ROLE_SLOT_FIELD_COUNT
        role_value = current_values[base] if base < len(current_values) else ""
        voice_value = current_values[base + 1] if base + 1 < len(current_values) else None
        emotion_value = current_values[base + 2] if base + 2 < len(current_values) else "默认"
        if role_value:
            existing[str(role_value).strip()] = {
                "voice": str(voice_value).strip() if voice_value else "",
                "emotion": str(emotion_value).strip() if emotion_value else "默认",
            }

    role_names = dialogue_roles(lines)
    available_voices = voice_lib_list()
    for idx in range(DIALOGUE_ROLE_SLOT_COUNT):
        role = role_names[idx] if idx < len(role_names) else ""
        base = idx * DIALOGUE_ROLE_SLOT_FIELD_COUNT
        current_role = current_values[base] if base < len(current_values) else ""
        if not role and current_role:
            role = str(current_role).strip()
        existing_item = existing.get(role, {})
        voice = existing_item.get("voice", "")
        emotion = existing_item.get("emotion", "默认")
        if not voice and role in available_voices:
            voice = role
        if not role:
            voice = None
            emotion = "默认"
        updates.extend([
            gr.update(visible=bool(role)),
            gr.update(value=role),
            gr.update(value=voice if role else None),
            gr.update(value=emotion),
        ])
    return (*updates, preview)


def refresh_dialogue_voice_library():
    voices = voice_lib_list()
    voice_updates = [gr.update(choices=voices) for _ in range(DIALOGUE_ROLE_SLOT_COUNT)]
    emotion_updates = [gr.update() for _ in range(DIALOGUE_ROLE_SLOT_COUNT)]
    return (*voice_updates, *emotion_updates)


def build_role_voice_map_from_slots(*slot_values):
    mapping = {}
    for idx in range(DIALOGUE_ROLE_SLOT_COUNT):
        base = idx * DIALOGUE_ROLE_SLOT_FIELD_COUNT
        role = slot_values[base] if base < len(slot_values) else ""
        voice = slot_values[base + 1] if base + 1 < len(slot_values) else ""
        emotion = slot_values[base + 2] if base + 2 < len(slot_values) else "默认"
        role = str(role or "").strip()
        voice = str(voice or "").strip()
        emotion = str(emotion or "默认").strip()
        if role and voice:
            mapping[role] = {"voice": voice, "emotion": emotion}
    return mapping


def combine_wav_files(wav_paths, output_path, interval_ms=450):
    if not wav_paths:
        raise ValueError("No dialogue audio segments were generated.")

    with wave.open(wav_paths[0], "rb") as first:
        params = first.getparams()
        nchannels = first.getnchannels()
        sampwidth = first.getsampwidth()
        framerate = first.getframerate()

    silence_frames = int(framerate * max(0, int(interval_ms)) / 1000)
    silence = b"\x00" * silence_frames * nchannels * sampwidth
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with wave.open(output_path, "wb") as output:
        output.setparams(params)
        for idx, path in enumerate(wav_paths):
            with wave.open(path, "rb") as segment:
                if segment.getnchannels() != nchannels or segment.getsampwidth() != sampwidth or segment.getframerate() != framerate:
                    raise ValueError(f"Audio format mismatch while combining: {path}")
                output.writeframes(segment.readframes(segment.getnframes()))
            if idx < len(wav_paths) - 1 and silence:
                output.writeframes(silence)


def gen_dialogue(script, role_mapping, interval_ms, max_text_tokens_per_segment=120, *args, progress=gr.Progress()):
    lines = parse_dialogue_script(script)
    if not lines:
        gr.Warning(i18n("没有解析到角色台词，请使用“角色：台词”格式"))
        return gr.update(value=None), gr.update(value=[]), gr.update(value="")

    missing_roles = []
    resolved_voices = {}
    resolved_emotions = {}
    for role in dialogue_roles(lines):
        role_config = role_mapping.get(role, {})
        voice_ref = role_config.get("voice", "") if isinstance(role_config, dict) else role_config
        emotion = role_config.get("emotion", "默认") if isinstance(role_config, dict) else "默认"
        voice_path = resolve_role_voice(voice_ref)
        if not voice_path:
            missing_roles.append(role)
        else:
            resolved_voices[role] = voice_path
            resolved_emotions[role] = emotion

    if missing_roles:
        gr.Warning(i18n("这些角色还没有绑定可用音色：") + "、".join(missing_roles))
        return gr.update(value=None), gr.update(value=dialogue_preview_data(script)), gr.update(value="")

    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }

    task_id = time.strftime("dialogue_%Y%m%d_%H%M%S")
    task_dir = os.path.join(DIALOGUE_OUTPUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    generated_paths = []
    manifest_rows = []
    total = len(lines)
    for idx, item in enumerate(lines):
        role = item["role"]
        text = item["text"]
        emotion = resolved_emotions.get(role, "默认")
        emo_vector = DIALOGUE_EMOTION_VECTORS.get(emotion)
        if emo_vector is not None:
            emo_vector = tts.normalize_emo_vec(emo_vector, apply_bias=True)
        safe_role = _safe_voice_name(role) or f"role_{idx + 1}"
        output_path = os.path.join(task_dir, f"{idx + 1:03d}_{safe_role}.wav")
        progress((idx + 1) / total, desc=f"{role} {idx + 1}/{total}")
        result = tts.infer(
            spk_audio_prompt=resolved_voices[role],
            text=text,
            output_path=output_path,
            emo_audio_prompt=None,
            emo_alpha=0.65,
            emo_vector=emo_vector,
            use_emo_text=False,
            emo_text=None,
            use_random=False,
            verbose=cmd_args.verbose,
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
            **kwargs,
        )
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, tuple, list)):
            result = list(result)[-1]
        generated_paths.append(output_path)
        manifest_rows.append([idx + 1, role, emotion, text, output_path])

    final_path = os.path.join(task_dir, "dialogue_full.wav")
    combine_wav_files(generated_paths, final_path, interval_ms=interval_ms)
    manifest_path = os.path.join(task_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, ensure_ascii=False, indent=2)

    status = i18n("生成完成：") + final_path
    return gr.update(value=final_path, visible=True), gr.update(value=manifest_rows), gr.update(value=status)


# ------------------------------------------------------------------
# Novel audiobook project layer.
# Project -> many txt tasks -> GPT script rows -> IndexTTS dialogue synthesis.
# ------------------------------------------------------------------
NOVEL_PROJECTS_DIR = os.path.join(ROOT_DIR, "projects")
NOVEL_PROMPT_FILENAME = "prompt.md"
NOVEL_PROJECT_META = "project.json"
NOVEL_TASK_META = "task.json"
NOVEL_SCRIPT_JSON = "script.json"
NOVEL_DIALOGUE_TXT = "dialogue.txt"
NOVEL_DEFAULT_CHUNK_TARGET = 500
NOVEL_DEFAULT_CHUNK_MAX = 800
NOVEL_SCHEMA_EMOTIONS = DIALOGUE_EMOTION_CHOICES
os.makedirs(NOVEL_PROJECTS_DIR, exist_ok=True)


NOVEL_DEFAULT_PROMPT = """# IndexTTS V26 中文小说多人有声书脚本整理器

你要把输入的中文小说原文整理成 IndexTTS 多人对话脚本。
输出必须是严格 JSON，不要输出 Markdown，不要解释。

核心规则：
1. 严格保留输入文本的顺序和内容，不改写、不总结、不扩写、不遗漏。
2. 对话只移除最外层成对引号，其他文字、标点、书名号、专名都要保留。
3. 旁白、场景、动作、心理描写、说话前后的动作和“某某道/问/说”等归入“旁白”。
4. 角色台词使用角色主中文名；无法确定时用 Unknown1、UnknownMale1 或 UnknownFemale1。
5. 每条 line 只放一个说话者或旁白片段，不要把旁白和台词混在同一条。
6. emo 只能从这些标签中选择：默认、高兴、愤怒、悲伤、恐惧、厌恶、低落、惊喜、平静。
7. 旁白通常用“平静”或“默认”；对白按语义选择最贴近的 IndexTTS 情绪。
8. mem 只用于下一块连续理解，80 个中文字符以内，不能替代任何正文。
9. chars 只记录本块出现或用于消解称呼的角色，note 用中文简短说明身份线索。
10. 绝不能丢弃说话引导语、动作描写或台词后的旁白。例如“纸侠显得颇为兴奋，在那儿自顾自地说着：”必须单独输出为“旁白”。
11. 一个原文段落如果包含“旁白 + 台词 + 旁白 + 台词”，必须按原顺序拆成多条 line，不能只保留台词。

输出 JSON 字段：
- mem: 下一块记忆。
- chars: 角色数组，每项 n=主名，g=m/f/u，note=身份线索。
- lines: 有序脚本数组，每项 spk=旁白或角色名，txt=原文片段，emo=IndexTTS 情绪。
"""


def novel_llm_defaults():
    defaults = {
        "api_base_url": os.environ.get("AIGC_BASE_URL", "https://aigc789.top").rstrip("/"),
        "api_key": os.environ.get("AIGC_API_KEY", ""),
        "preprocess_model": os.environ.get("AIGC_PREPROCESS_MODEL", "gpt-5.4-mini"),
        "chunk_target_chars": NOVEL_DEFAULT_CHUNK_TARGET,
        "chunk_max_chars": NOVEL_DEFAULT_CHUNK_MAX,
    }
    old_config = os.path.join(os.path.dirname(ROOT_DIR), "audiobook_tool", "config", "local_settings.json")
    try:
        with open(old_config, "r", encoding="utf-8") as f:
            old = json.load(f)
        defaults["api_base_url"] = os.environ.get("AIGC_BASE_URL", old.get("api_base_url") or defaults["api_base_url"]).rstrip("/")
        defaults["api_key"] = os.environ.get("AIGC_API_KEY", old.get("api_key") or defaults["api_key"])
        defaults["preprocess_model"] = os.environ.get("AIGC_PREPROCESS_MODEL", old.get("preprocess_model") or defaults["preprocess_model"])
    except Exception:
        pass
    return defaults


def read_json_file(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def write_text_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_text_safely(path):
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def safe_project_name(name):
    safe = _safe_voice_name(name)
    safe = re.sub(r"\s+", "_", safe)
    return safe or "novel"


def natural_sort_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def project_meta_path(project_id):
    return os.path.join(NOVEL_PROJECTS_DIR, project_id, NOVEL_PROJECT_META)


def load_project(project_id):
    if not project_id:
        return None
    return read_json_file(project_meta_path(project_id))


def save_project(meta):
    write_json_file(project_meta_path(meta["id"]), meta)


def project_dir(project_id):
    return os.path.join(NOVEL_PROJECTS_DIR, project_id)


def task_dir(project_id, task_id):
    return os.path.join(project_dir(project_id), "tasks", task_id)


def novel_project_choices():
    choices = []
    for fn in sorted(os.listdir(NOVEL_PROJECTS_DIR), key=natural_sort_key):
        meta = load_project(fn)
        if meta:
            choices.append((f"{meta.get('name', fn)} [{fn}]", fn))
    return choices


def parse_project_choice(choice):
    if not choice:
        return ""
    if isinstance(choice, (list, tuple)) and choice:
        choice = choice[-1]
    text = str(choice).strip()
    if os.path.exists(project_meta_path(text)):
        return text
    match = re.search(r"\[([^\[\]]+)\]\s*$", str(choice))
    return match.group(1) if match else str(choice).strip()


def task_choice_label(task):
    return f"{int(task.get('index', 0)):03d} {task.get('title', task.get('id', ''))}"


def task_choices(meta):
    return [(task_choice_label(task), task.get("id", "")) for task in meta.get("tasks", [])]


def parse_task_choice(choice, valid_ids=None):
    if not choice:
        return ""
    if isinstance(choice, (list, tuple)) and choice:
        choice = choice[-1]
    text = str(choice).strip()
    valid_ids = list(valid_ids or [])
    if text in valid_ids:
        return text
    for task_id in valid_ids:
        if task_id and task_id in text:
            return task_id
    match = re.search(r"\[([^\[\]]+)\]\s*$", str(choice))
    return match.group(1) if match else text


def parse_task_choices(values, valid_ids=None):
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    parsed = []
    for value in values:
        task_id = parse_task_choice(value, valid_ids=valid_ids)
        if task_id:
            parsed.append(task_id)
    return parsed


def ensure_project_prompt(project_id):
    path = os.path.join(project_dir(project_id), NOVEL_PROMPT_FILENAME)
    if not os.path.exists(path):
        write_text_file(path, NOVEL_DEFAULT_PROMPT)
    return path


def find_task(meta, task_id):
    for task in meta.get("tasks", []):
        if task.get("id") == task_id:
            return task
    return None


def novel_tasks_table(meta):
    rows = []
    if not meta:
        return rows
    for task in meta.get("tasks", []):
        script_path = os.path.join(task_dir(meta["id"], task["id"]), NOVEL_SCRIPT_JSON)
        script = read_json_file(script_path, {}) if os.path.exists(script_path) else {}
        line_count = len(script.get("lines", [])) if script else 0
        rows.append([
            task.get("index", 0),
            task.get("id", ""),
            task.get("title", ""),
            task.get("status", "new"),
            task.get("source_chars", 0),
            line_count,
            task.get("audio_path", ""),
        ])
    return rows


def novel_chars_table(meta):
    rows = []
    if not meta:
        return rows
    chars = meta.get("characters", {})
    for name, info in sorted(chars.items(), key=lambda item: (-int(item[1].get("count", 0)), item[0])):
        rows.append([
            name,
            int(info.get("count", 0)),
            info.get("gender", "u"),
            info.get("voice", ""),
            info.get("note", ""),
        ])
    return rows


def dataframe_rows(data):
    if data is None:
        return []
    if isinstance(data, pd.DataFrame):
        return data.fillna("").values.tolist()
    if isinstance(data, dict) and "data" in data:
        return data.get("data") or []
    if isinstance(data, list):
        return data
    return []


def normalize_gender(value):
    value = str(value or "u").strip().lower()
    if value in ("m", "男", "male"):
        return "m"
    if value in ("f", "女", "female"):
        return "f"
    return "u"


def guess_gender_from_name(name, note=""):
    text = f"{name} {note}"
    if any(word in text for word in ("女", "小姐", "夫人", "姑娘", "少女", "女人", "母亲", "姐姐", "妹妹")):
        return "f"
    if any(word in text for word in ("男", "先生", "少年", "男人", "父亲", "哥哥", "弟弟", "老头")):
        return "m"
    return "u"


def auto_match_voice(role, gender, used_voices=None):
    voices = voice_lib_list()
    used_voices = used_voices or set()
    if not voices:
        return ""
    if role in voices and role not in used_voices:
        return role
    for voice in voices:
        if voice in used_voices:
            continue
        if role and (voice.startswith(role) or role.startswith(voice)):
            return voice
    if role == "旁白":
        for voice in voices:
            if voice not in used_voices and voice.startswith("旁白"):
                return voice
    gender_words = {"m": ("男", "男声", "叔", "少年"), "f": ("女", "女声", "少女", "温柔")}
    for voice in voices:
        if voice in used_voices:
            continue
        if any(word in voice for word in gender_words.get(gender, ())):
            return voice
    for voice in voices:
        if voice not in used_voices:
            return voice
    return voices[0]


def normalize_dialogue_text(text):
    text = str(text or "").strip()
    text = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", "", text)
    all_quote_chars = "“”‘’\"'「」『』"
    quote_pairs = {
        "“": "”",
        "‘": "’",
        "\"": "\"",
        "'": "'",
        "「": "」",
        "『": "』",
    }
    changed = True
    while changed and len(text) >= 2:
        changed = False
        first = text[0]
        last = text[-1]
        if first in quote_pairs and quote_pairs[first] == last:
            text = text[1:-1].strip()
            changed = True
        elif first in quote_pairs:
            text = text[1:].strip()
            changed = True
        elif last in quote_pairs.values():
            text = text[:-1].strip()
            changed = True
    text = text.translate(str.maketrans("", "", all_quote_chars)).strip()
    return text


def normalize_coverage_text(text):
    text = re.sub(r"\s+", "", str(text or ""))
    return text.translate(str.maketrans("", "", "“”‘’\"「」『』"))


def normalized_text_with_raw_map(text):
    chars = []
    raw_indexes = []
    quote_chars = set("“”‘’\"'「」『』")
    for idx, char in enumerate(str(text or "")):
        if char.isspace() or char in quote_chars:
            continue
        chars.append(char)
        raw_indexes.append(idx)
    return "".join(chars), raw_indexes


def insert_missing_narration_fragments(source_text, lines):
    source_norm, source_map = normalized_text_with_raw_map(source_text)
    line_norms = [normalized_text_with_raw_map(line.get("txt", ""))[0] for line in lines]
    output_norm = "".join(line_norms)
    if not source_norm or source_norm == output_norm:
        return lines, []

    repaired = [dict(line) for line in lines]
    missing = []
    inserted = 0
    matcher = difflib.SequenceMatcher(a=source_norm, b=output_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "delete" or i1 >= i2:
            continue
        raw_start = source_map[i1]
        raw_end = source_map[i2 - 1] + 1
        fragment = str(source_text or "")[raw_start:raw_end].strip()
        fragment_norm = normalize_coverage_text(fragment)
        if not fragment_norm:
            continue
        if len(fragment_norm) <= 1 and fragment.strip("，。？！：；、,.!?;:"):
            continue
        local_start = max(0, j1 - len(fragment_norm) - 30)
        local_end = min(len(output_norm), j1 + len(fragment_norm) + 30)
        if fragment_norm in output_norm[local_start:local_end]:
            continue

        insert_at = 0
        acc = 0
        for line_norm in line_norms:
            if acc + len(line_norm) <= j1:
                insert_at += 1
                acc += len(line_norm)
            else:
                break
        insert_at = min(len(repaired), insert_at + inserted)
        repaired.insert(insert_at, {"spk": "旁白", "txt": fragment, "emo": "平静"})
        inserted += 1
        missing.append(fragment)
    return repaired, missing


def split_long_unit(unit, target_chars, max_chars):
    unit = unit.strip()
    chunks = []
    punctuation = "。！？!?；;，,"
    while len(unit) > max_chars:
        window = unit[:max_chars]
        cut = -1
        for mark in punctuation:
            cut = max(cut, window.rfind(mark))
        if cut < max(80, int(target_chars * 0.45)):
            cut = max_chars - 1
        chunks.append(unit[:cut + 1].strip())
        unit = unit[cut + 1:].strip()
    if unit:
        chunks.append(unit)
    return chunks


def split_novel_text(text, target_chars=NOVEL_DEFAULT_CHUNK_TARGET, max_chars=NOVEL_DEFAULT_CHUNK_MAX):
    target_chars = max(180, int(target_chars or NOVEL_DEFAULT_CHUNK_TARGET))
    max_chars = max(target_chars, int(max_chars or NOVEL_DEFAULT_CHUNK_MAX))
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks = []
    current = ""
    for para in paragraphs:
        units = split_long_unit(para, target_chars, max_chars) if len(para) > max_chars else [para]
        for unit in units:
            sep = "\n\n" if current else ""
            if current and len(current) + len(sep) + len(unit) > max_chars:
                chunks.append(current.strip())
                current = unit
            elif current and len(current) >= target_chars:
                chunks.append(current.strip())
                current = unit
            else:
                current = current + sep + unit if current else unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def novel_response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "indextts_novel_chunk",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mem", "chars", "lines"],
                "properties": {
                    "mem": {"type": "string"},
                    "chars": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["n", "g", "note"],
                            "properties": {
                                "n": {"type": "string"},
                                "g": {"type": "string", "enum": ["m", "f", "u"]},
                                "note": {"type": "string"},
                            },
                        },
                    },
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["spk", "txt", "emo"],
                            "properties": {
                                "spk": {"type": "string"},
                                "txt": {"type": "string"},
                                "emo": {"type": "string", "enum": NOVEL_SCHEMA_EMOTIONS},
                            },
                        },
                    },
                },
            },
        },
    }


def extract_json_object(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def call_structured_llm(api_base_url, api_key, model, system_prompt, user_content, max_tokens=6000):
    api_base_url = str(api_base_url or "").rstrip("/")
    defaults = novel_llm_defaults()
    if not api_key:
        api_key = defaults.get("api_key", "")
    if not api_base_url:
        raise ValueError("请填写 API Base URL")
    if not api_key:
        raise ValueError("请填写 API Key")
    candidate_models = []
    for candidate in [model, defaults.get("preprocess_model"), "gpt-5.4-mini"]:
        candidate = str(candidate or "").strip()
        if candidate and candidate not in candidate_models:
            candidate_models.append(candidate)
    payload_base = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": int(max_tokens),
        "stream": False,
        "response_format": novel_response_format(),
    }
    last_error = ""
    for candidate_model in candidate_models:
        payload = dict(payload_base)
        payload["model"] = candidate_model
        req = urllib.request.Request(
            api_base_url + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"].get("content", "")
            result = extract_json_object(content)
            usage = data.get("usage", {})
            if isinstance(usage, dict):
                usage["_model"] = candidate_model
            return result, usage
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_error = f"LLM HTTP {e.code} model={candidate_model}: {body[:800]}"
            if e.code in (400, 404, 503) and "model_not_found" in body and candidate_model != candidate_models[-1]:
                continue
            raise RuntimeError(last_error) from e
        except Exception as e:
            last_error = f"LLM call failed model={candidate_model}: {e}"
            raise RuntimeError(last_error) from e
    raise RuntimeError(last_error or "LLM call failed")


def sync_project_characters(project_id, auto_assign=False):
    meta = load_project(project_id)
    if not meta:
        return None
    existing = meta.get("characters", {})
    counts = {}
    notes = {}
    genders = {}
    for task in meta.get("tasks", []):
        script_path = os.path.join(task_dir(project_id, task["id"]), NOVEL_SCRIPT_JSON)
        script = read_json_file(script_path, {})
        for line in script.get("lines", []):
            role = str(line.get("spk") or "").strip()
            if not role:
                continue
            counts[role] = counts.get(role, 0) + 1
        for char in script.get("chars", []):
            name = str(char.get("n") or "").strip()
            if not name:
                continue
            if char.get("note"):
                notes[name] = char.get("note", "")
            if char.get("g") in ("m", "f", "u"):
                genders[name] = char.get("g")
    used_voices = set()
    characters = {}
    for role, count in counts.items():
        old = existing.get(role, {})
        gender = old.get("gender") or genders.get(role) or guess_gender_from_name(role, notes.get(role, ""))
        voice = old.get("voice", "")
        if auto_assign and not voice:
            voice = auto_match_voice(role, gender, used_voices)
        if voice:
            used_voices.add(voice)
        characters[role] = {
            "count": count,
            "gender": normalize_gender(gender),
            "voice": voice,
            "note": old.get("note") or notes.get(role, ""),
        }
    meta["characters"] = dict(sorted(characters.items(), key=lambda item: (-int(item[1].get("count", 0)), item[0])))
    save_project(meta)
    return meta


def create_novel_project(project_name, input_path, api_base_url, api_key, model):
    input_path = os.path.abspath(str(input_path or "").strip().strip('"'))
    if not os.path.exists(input_path):
        raise ValueError("TXT 文件或文件夹不存在")
    if os.path.isdir(input_path):
        txt_files = [
            os.path.join(root, fn)
            for root, _, files in os.walk(input_path)
            for fn in files
            if fn.lower().endswith(".txt")
        ]
    else:
        txt_files = [input_path] if input_path.lower().endswith(".txt") else []
    txt_files = sorted(txt_files, key=natural_sort_key)
    if not txt_files:
        raise ValueError("没有找到 txt 文件")
    name = project_name.strip() if project_name else os.path.splitext(os.path.basename(input_path))[0]
    project_id = f"{safe_project_name(name)}_{uuid.uuid4().hex[:10]}"
    root = project_dir(project_id)
    os.makedirs(os.path.join(root, "tasks"), exist_ok=True)
    meta = {
        "id": project_id,
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_path": input_path,
        "settings": {
            "api_base_url": api_base_url,
            "preprocess_model": model or novel_llm_defaults().get("preprocess_model", "gpt-5.4-mini"),
            "chunk_target_chars": NOVEL_DEFAULT_CHUNK_TARGET,
            "chunk_max_chars": NOVEL_DEFAULT_CHUNK_MAX,
            "interval_ms": 450,
            "max_text_tokens_per_segment": 120,
        },
        "characters": {},
        "tasks": [],
    }
    for idx, src in enumerate(txt_files, 1):
        title = os.path.splitext(os.path.basename(src))[0]
        task_id = f"{idx:04d}_{safe_project_name(title)}_{uuid.uuid4().hex[:8]}"
        td = task_dir(project_id, task_id)
        os.makedirs(td, exist_ok=True)
        text = read_text_safely(src)
        source_copy = os.path.join(td, "source.txt")
        write_text_file(source_copy, text)
        task_meta = {
            "id": task_id,
            "index": idx,
            "title": title,
            "source_path": src,
            "source_copy": source_copy,
            "source_chars": len(text),
            "status": "new",
            "script_path": "",
            "dialogue_path": "",
            "audio_path": "",
            "mp3_path": "",
        }
        write_json_file(os.path.join(td, NOVEL_TASK_META), task_meta)
        meta["tasks"].append(task_meta)
    save_project(meta)
    ensure_project_prompt(project_id)
    return meta


def preprocess_novel_task(project_id, task_id, api_base_url, api_key, model, target_chars, max_chars, progress=None):
    meta = load_project(project_id)
    if not meta:
        raise ValueError("项目不存在")
    task = find_task(meta, task_id)
    if not task:
        raise ValueError("任务不存在")
    td = task_dir(project_id, task_id)
    source_path = task.get("source_copy") or os.path.join(td, "source.txt")
    source = read_text_safely(source_path)
    chunks = split_novel_text(source, target_chars, max_chars)
    prompt_path = ensure_project_prompt(project_id)
    system_prompt = read_text_safely(prompt_path)
    all_lines = []
    all_chars = []
    warnings_list = []
    mem = ""
    known_chars = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    actual_model = model or novel_llm_defaults().get("preprocess_model", "gpt-5.4-mini")
    for idx, chunk in enumerate(chunks, 1):
        if progress:
            progress(idx / max(1, len(chunks)), desc=f"GPT 预处理 {idx}/{len(chunks)}")
        known_text = "、".join(known_chars[:80])
        user_content = (
            f"<mem>{mem}</mem>\n"
            f"<known_chars>{known_text}</known_chars>\n"
            f"<text>\n{chunk}\n</text>"
        )
        result, usage = call_structured_llm(
            api_base_url,
            api_key,
            model,
            system_prompt,
            user_content,
            max_tokens=6000,
        )
        actual_model = usage.get("_model", actual_model) if isinstance(usage, dict) else actual_model
        mem = str(result.get("mem", ""))[:120]
        chunk_lines = []
        for line in result.get("lines", []):
            role = str(line.get("spk") or "").strip() or "旁白"
            text = normalize_dialogue_text(line.get("txt", ""))
            emotion = str(line.get("emo") or "默认").strip()
            if emotion not in NOVEL_SCHEMA_EMOTIONS:
                emotion = "默认"
            if text:
                chunk_lines.append({"spk": role, "txt": text, "emo": emotion})
                if role not in known_chars:
                    known_chars.append(role)
        for char in result.get("chars", []):
            name = str(char.get("n") or "").strip()
            if name and name not in known_chars:
                known_chars.append(name)
            if name:
                all_chars.append({
                    "n": name,
                    "g": normalize_gender(char.get("g")),
                    "note": str(char.get("note") or "").strip(),
                })
        chunk_lines, repaired_fragments = insert_missing_narration_fragments(chunk, chunk_lines)
        if repaired_fragments:
            warnings_list.append(f"第 {idx} 块自动补回遗漏旁白 {len(repaired_fragments)} 段。")
        joined = "".join(item["txt"] for item in chunk_lines)
        if normalize_coverage_text(joined) != normalize_coverage_text(chunk):
            warnings_list.append(f"第 {idx} 块覆盖校验不一致，请人工检查。")
        for line in chunk_lines:
            line["idx"] = len(all_lines) + 1
            all_lines.append(line)
        for key in total_tokens:
            total_tokens[key] += int(usage.get(key, 0) or 0)
    script = {
        "task_id": task_id,
        "title": task.get("title", ""),
        "source_chars": len(source),
        "chunk_count": len(chunks),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": actual_model,
        "usage": total_tokens,
        "warnings": warnings_list,
        "chars": all_chars,
        "lines": all_lines,
    }
    script_path = os.path.join(td, NOVEL_SCRIPT_JSON)
    dialogue_path = os.path.join(td, NOVEL_DIALOGUE_TXT)
    write_json_file(script_path, script)
    write_text_file(dialogue_path, "\n".join(f"{line['spk']}：{line['txt']}" for line in all_lines))
    task["script_path"] = script_path
    task["dialogue_path"] = dialogue_path
    task["status"] = "preprocessed"
    task["chunk_count"] = len(chunks)
    save_project(meta)
    meta = sync_project_characters(project_id, auto_assign=True)
    return meta, script


def save_project_characters_from_table(project_id, table):
    meta = load_project(project_id)
    if not meta:
        raise ValueError("项目不存在")
    characters = meta.get("characters", {})
    updated = {}
    for row in dataframe_rows(table):
        if len(row) < 1:
            continue
        role = str(row[0] or "").strip()
        if not role:
            continue
        old = characters.get(role, {})
        count = int(float(row[1])) if len(row) > 1 and str(row[1] or "").strip() else int(old.get("count", 0))
        gender = normalize_gender(row[2] if len(row) > 2 else old.get("gender"))
        voice = str(row[3] or "").strip() if len(row) > 3 else old.get("voice", "")
        note = str(row[4] or "").strip() if len(row) > 4 else old.get("note", "")
        updated[role] = {"count": count, "gender": gender, "voice": voice, "note": note}
    meta["characters"] = dict(sorted(updated.items(), key=lambda item: (-int(item[1].get("count", 0)), item[0])))
    save_project(meta)
    return meta


def script_rows_table(script):
    return [[line.get("idx", i + 1), line.get("spk", ""), line.get("emo", "默认"), line.get("txt", "")]
            for i, line in enumerate((script or {}).get("lines", []))]


def save_task_script_from_table(project_id, task_id, table):
    td = task_dir(project_id, task_id)
    script_path = os.path.join(td, NOVEL_SCRIPT_JSON)
    script = read_json_file(script_path, {})
    lines = []
    for idx, row in enumerate(dataframe_rows(table), 1):
        if len(row) < 4:
            continue
        role = str(row[1] or "").strip()
        text = normalize_dialogue_text(row[3])
        emotion = str(row[2] or "默认").strip()
        if emotion not in NOVEL_SCHEMA_EMOTIONS:
            emotion = "默认"
        if role and text:
            lines.append({"idx": idx, "spk": role, "emo": emotion, "txt": text})
    script["lines"] = lines
    script["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json_file(script_path, script)
    write_text_file(os.path.join(td, NOVEL_DIALOGUE_TXT), "\n".join(f"{line['spk']}：{line['txt']}" for line in lines))
    meta = sync_project_characters(project_id, auto_assign=False)
    return meta, script


def convert_wav_to_mp3(wav_path, mp3_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return ""
    os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "2", mp3_path]
    subprocess.run(cmd, check=True)
    return mp3_path if os.path.exists(mp3_path) else ""


def synthesize_dialogue_rows(rows, role_mapping, output_dir, final_name, interval_ms, max_text_tokens_per_segment,
                             generation_args, progress=None):
    if not rows:
        raise ValueError("脚本为空")
    missing_roles = []
    resolved_voices = {}
    for row in rows:
        role = row.get("spk", "")
        if role in resolved_voices:
            continue
        voice_ref = (role_mapping.get(role) or {}).get("voice", "")
        voice_path = resolve_role_voice(voice_ref)
        if not voice_path:
            missing_roles.append(role)
        else:
            resolved_voices[role] = voice_path
    if missing_roles:
        raise ValueError("这些角色还没有绑定可用音色：" + "、".join(sorted(set(missing_roles))))

    do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens = generation_args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []
    manifest_rows = []
    total = len(rows)
    for idx, row in enumerate(rows, 1):
        role = row.get("spk", "")
        text = row.get("txt", "")
        emotion = row.get("emo", "默认")
        if emotion not in NOVEL_SCHEMA_EMOTIONS:
            emotion = "默认"
        emo_vector = DIALOGUE_EMOTION_VECTORS.get(emotion)
        if emo_vector is not None:
            emo_vector = tts.normalize_emo_vec(emo_vector, apply_bias=True)
        safe_role = _safe_voice_name(role) or f"role_{idx}"
        output_path = os.path.join(output_dir, f"{idx:04d}_{safe_role}.wav")
        if progress:
            progress(idx / total, desc=f"{role} {idx}/{total}")
        tts.infer(
            spk_audio_prompt=resolved_voices[role],
            text=text,
            output_path=output_path,
            emo_audio_prompt=None,
            emo_alpha=0.65,
            emo_vector=emo_vector,
            use_emo_text=False,
            emo_text=None,
            use_random=False,
            verbose=cmd_args.verbose,
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
            **kwargs,
        )
        generated_paths.append(output_path)
        manifest_rows.append([idx, role, emotion, text, output_path])
    final_wav = os.path.join(output_dir, final_name + ".wav")
    combine_wav_files(generated_paths, final_wav, interval_ms=interval_ms)
    final_mp3 = ""
    try:
        final_mp3 = convert_wav_to_mp3(final_wav, os.path.join(output_dir, final_name + ".mp3"))
    except Exception as e:
        print(">> mp3 conversion failed:", e)
    write_json_file(os.path.join(output_dir, "manifest.json"), manifest_rows)
    return final_wav, final_mp3, manifest_rows


def synthesize_novel_task(project_id, task_id, interval_ms, max_text_tokens_per_segment, generation_args, progress=None):
    meta = load_project(project_id)
    if not meta:
        raise ValueError("项目不存在")
    task = find_task(meta, task_id)
    if not task:
        raise ValueError("任务不存在")
    script_path = os.path.join(task_dir(project_id, task_id), NOVEL_SCRIPT_JSON)
    script = read_json_file(script_path, {})
    rows = script.get("lines", [])
    role_mapping = {role: {"voice": info.get("voice", "")} for role, info in meta.get("characters", {}).items()}
    output_dir = os.path.join(task_dir(project_id, task_id), "outputs", time.strftime("%Y%m%d_%H%M%S"))
    final_name = safe_project_name(task.get("title", task_id))
    final_wav, final_mp3, manifest_rows = synthesize_dialogue_rows(
        rows,
        role_mapping,
        output_dir,
        final_name,
        interval_ms,
        max_text_tokens_per_segment,
        generation_args,
        progress=progress,
    )
    task["status"] = "synthesized"
    task["audio_path"] = final_wav
    task["mp3_path"] = final_mp3
    save_project(meta)
    return meta, final_mp3 or final_wav, manifest_rows


PREMIUM_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
    font=["Inter", "Segoe UI Variable", "Segoe UI", "Microsoft YaHei UI", "PingFang SC", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["Cascadia Mono", "SFMono-Regular", "Consolas", "ui-monospace", "monospace"],
)

PREMIUM_CSS = """
:root {
    --wb-bg: #0d1117;
    --wb-bg-soft: #111722;
    --wb-panel: #151b24;
    --wb-panel-raised: #19212d;
    --wb-panel-solid: #101620;
    --wb-input: #0b1018;
    --wb-border: rgba(226, 232, 240, 0.12);
    --wb-border-strong: rgba(226, 232, 240, 0.22);
    --wb-text: #f8fafc;
    --wb-muted: #a7b0bf;
    --wb-faint: #707b8d;
    --wb-blue: #3b82f6;
    --wb-blue-hover: #60a5fa;
    --wb-green: #22c55e;
    --wb-green-soft: rgba(34, 197, 94, 0.12);
    --wb-red: #ef4444;
    --wb-red-soft: rgba(239, 68, 68, 0.12);
    --wb-amber: #f59e0b;
    --wb-radius: 8px;
    --wb-radius-sm: 6px;
    --wb-shadow: 0 18px 52px rgba(0, 0, 0, 0.30), inset 0 1px 0 rgba(255,255,255,0.04);
    --wb-font: "Inter", "Segoe UI Variable", "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
    --wb-mono: "Cascadia Mono", "SFMono-Regular", Consolas, ui-monospace, monospace;
}

html, body, .gradio-container {
    background: var(--wb-bg) !important;
    color: var(--wb-text) !important;
    font-family: var(--wb-font) !important;
    letter-spacing: 0 !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.gradio-container {
    max-width: 1640px !important;
    min-height: 100vh;
    margin: 0 auto !important;
    padding: 18px 24px 34px !important;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.035) 0, rgba(255,255,255,0) 170px),
        var(--wb-bg) !important;
}

.gradio-container::before { display: none !important; }
.gradio-container > * { position: relative; z-index: 1; }

.console-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    min-height: 66px;
    margin: 0 0 14px;
    padding: 12px 16px;
    border: 1px solid var(--wb-border);
    border-radius: var(--wb-radius);
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018)), var(--wb-panel);
    box-shadow: var(--wb-shadow);
}

.console-brand {
    display: flex;
    align-items: center;
    gap: 13px;
    min-width: 0;
}

.console-mark {
    width: 40px;
    height: 40px;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    border-radius: var(--wb-radius-sm);
    border: 1px solid rgba(59,130,246,0.45);
    background: linear-gradient(180deg, rgba(59,130,246,0.22), rgba(59,130,246,0.08)), #101826;
    color: #bfdbfe;
    font-family: var(--wb-mono);
    font-size: 13px;
    font-weight: 800;
}

.console-title {
    color: var(--wb-text);
    font-size: 20px;
    line-height: 1.15;
    font-weight: 760;
    letter-spacing: 0 !important;
}

.console-subtitle {
    margin-top: 3px;
    color: var(--wb-faint);
    font-family: var(--wb-mono);
    font-size: 11px;
    letter-spacing: 0 !important;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.console-status {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    flex-wrap: wrap;
}

.console-status span {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 5px 10px;
    border: 1px solid var(--wb-border);
    border-radius: 999px;
    background: rgba(11, 16, 24, 0.76);
    color: var(--wb-muted);
    font-family: var(--wb-mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0 !important;
}

.console-status a {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 5px 11px;
    border: 1px solid rgba(96,165,250,0.46);
    border-radius: 999px;
    background: rgba(59,130,246,0.13);
    color: #bfdbfe !important;
    font-family: var(--wb-font);
    font-size: 12px;
    font-weight: 760;
    letter-spacing: 0 !important;
    text-decoration: none !important;
}

.console-status a::after {
    content: ">";
    margin-left: 7px;
    color: #93c5fd;
    font-family: var(--wb-mono);
}

.console-status a:hover {
    border-color: rgba(147,197,253,0.72);
    background: rgba(59,130,246,0.20);
    color: #ffffff !important;
}

.console-status span:first-child::before {
    content: "";
    width: 7px;
    height: 7px;
    margin-right: 7px;
    border-radius: 999px;
    background: var(--wb-green);
    box-shadow: 0 0 0 4px rgba(34,197,94,0.12);
}

.premium-hero, .premium-footer { display: none !important; }

.gradio-container .tab-nav,
.gradio-container .tabs > .tab-nav {
    margin: 0 0 12px !important;
    padding: 0 !important;
    gap: 0 !important;
    border: 0 !important;
    border-bottom: 1px solid var(--wb-border) !important;
    background: transparent !important;
}

.gradio-container .tab-nav button {
    padding: 10px 14px !important;
    border: 0 !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    color: var(--wb-muted) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
}

.gradio-container .tab-nav button.selected {
    color: var(--wb-text) !important;
    border-bottom-color: var(--wb-blue) !important;
    box-shadow: none !important;
}

.studio-command-grid,
.gradio-container .studio-command-grid {
    display: grid !important;
    grid-template-columns: minmax(240px, 0.62fr) minmax(500px, 1.48fr) minmax(320px, 0.9fr) !important;
    gap: 14px !important;
    align-items: start !important;
}

.studio-command-grid > .gr-column,
.studio-command-grid > .block,
.studio-command-grid > div { min-width: 0 !important; }

.studio-card,
.studio-command-grid > .block,
.studio-command-grid > div {
    padding: 14px !important;
    border: 1px solid var(--wb-border) !important;
    border-radius: var(--wb-radius) !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.038), rgba(255,255,255,0.012)), var(--wb-panel) !important;
    box-shadow: var(--wb-shadow) !important;
}

.studio-card--voice {
    padding: 12px !important;
    border-top: 2px solid rgba(148,163,184,0.50) !important;
}
.studio-card--compose { border-top: 2px solid rgba(59,130,246,0.62) !important; }
.studio-card--output,
.studio-command-grid > .block:last-child,
.studio-command-grid > div:last-child { border-top: 2px solid rgba(34,197,94,0.58) !important; }

.studio-section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin: 0 0 12px !important;
    padding: 0 0 9px !important;
    border-bottom: 1px solid var(--wb-border) !important;
}

.studio-section-title__main {
    color: var(--wb-text) !important;
    font-size: 13px !important;
    font-weight: 780 !important;
    line-height: 1.2;
    letter-spacing: 0 !important;
}

.studio-section-title__meta {
    color: var(--wb-faint) !important;
    font-family: var(--wb-mono);
    font-size: 10.5px !important;
    font-weight: 700;
    letter-spacing: 0 !important;
}

.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .block,
.gradio-container .form,
.gradio-container details {
    border-color: var(--wb-border) !important;
    border-radius: var(--wb-radius) !important;
    background: var(--wb-panel-raised) !important;
    box-shadow: none !important;
}

.studio-card .block,
.studio-card .form,
.studio-card .gr-form,
.studio-card .gr-box,
.studio-card .gr-panel {
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}

.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .gr-form > label {
    color: var(--wb-muted) !important;
    font-size: 12px !important;
    font-weight: 680 !important;
    letter-spacing: 0 !important;
}

.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container .gr-textbox textarea,
.gradio-container .gr-textbox input,
.gradio-container select {
    border: 1px solid var(--wb-border) !important;
    border-radius: var(--wb-radius-sm) !important;
    background: var(--wb-input) !important;
    color: var(--wb-text) !important;
    font-family: var(--wb-font) !important;
    font-size: 14px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.035) !important;
}

.gradio-container textarea {
    min-height: 228px !important;
    line-height: 1.58 !important;
    resize: vertical;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: rgba(59,130,246,0.70) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.16), inset 0 1px 0 rgba(255,255,255,0.035) !important;
    outline: none !important;
}

.gradio-container ::placeholder { color: var(--wb-faint) !important; opacity: 1; }

.gradio-container button:not(.selected),
.gradio-container .gr-button:not(.selected),
.voice-lib-row button {
    min-height: 34px !important;
    border: 1px solid var(--wb-border-strong) !important;
    border-radius: var(--wb-radius-sm) !important;
    background: #232b38 !important;
    color: var(--wb-text) !important;
    font-size: 13px !important;
    font-weight: 720 !important;
    letter-spacing: 0 !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.045) !important;
    transition: background 0.14s ease, border-color 0.14s ease, color 0.14s ease;
}

.gradio-container button:not(.selected):hover,
.gradio-container .gr-button:not(.selected):hover,
.voice-lib-row button:hover {
    background: #2c3747 !important;
    border-color: rgba(226,232,240,0.32) !important;
    color: #ffffff !important;
}

#gen_button_wrap button,
button#component-gen_button,
button[id*="gen_button"],
.gradio-container button.primary,
.gradio-container button[variant="primary"] {
    width: 100% !important;
    min-height: 48px !important;
    border: 1px solid rgba(147,197,253,0.55) !important;
    border-radius: var(--wb-radius-sm) !important;
    background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 820 !important;
    text-shadow: none !important;
    box-shadow: 0 14px 28px rgba(37,99,235,0.20), inset 0 1px 0 rgba(255,255,255,0.22) !important;
}

#gen_button_wrap button:hover,
button#component-gen_button:hover,
button[id*="gen_button"]:hover {
    background: linear-gradient(180deg, #60a5fa 0%, #2f6fec 100%) !important;
    color: #ffffff !important;
}

.voice-lib-row {
    gap: 6px !important;
    margin-top: 6px !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

.voice-lib-row button {
    min-width: 54px !important;
    min-height: 30px !important;
    padding: 5px 9px !important;
    font-size: 11.5px !important;
}

.voice-name-input {
    flex: 0 0 148px !important;
    width: 148px !important;
    max-width: 148px !important;
    min-width: 112px !important;
}

.voice-name-input input,
.voice-name-input textarea {
    min-height: 30px !important;
    height: 30px !important;
    padding: 5px 9px !important;
    font-size: 12px !important;
    line-height: 18px !important;
}

.voice-btn-save button,
button.voice-btn-save {
    border-color: rgba(34,197,94,0.42) !important;
    background: var(--wb-green-soft) !important;
    color: #bbf7d0 !important;
}

.voice-btn-danger button,
button.voice-btn-danger {
    border-color: rgba(239,68,68,0.42) !important;
    background: var(--wb-red-soft) !important;
    color: #fecaca !important;
}

.gradio-container .gr-audio {
    min-height: 168px !important;
    overflow: hidden !important;
    border: 1px solid var(--wb-border) !important;
    border-radius: var(--wb-radius) !important;
    background: var(--wb-input) !important;
}

.studio-card--voice .gr-audio {
    min-height: 108px !important;
}

.studio-card--voice .studio-section-title {
    margin-bottom: 8px !important;
    padding-bottom: 7px !important;
}

.studio-card--voice input[type="text"],
.studio-card--voice .gr-textbox input,
.studio-card--voice select {
    min-height: 30px !important;
    padding: 6px 9px !important;
    font-size: 12.5px !important;
}

.studio-card--output .gr-audio,
.studio-command-grid > .block:last-child .gr-audio,
.studio-command-grid > div:last-child .gr-audio {
    min-height: 228px !important;
}

.gradio-container input[type="range"] { accent-color: var(--wb-blue) !important; }
.gradio-container input[type="checkbox"],
.gradio-container input[type="radio"] { accent-color: var(--wb-blue) !important; }

.gradio-container .gr-radio label,
.gradio-container .gr-checkbox label {
    padding: 8px 12px !important;
    border: 1px solid var(--wb-border) !important;
    border-radius: var(--wb-radius-sm) !important;
    background: #202838 !important;
    color: var(--wb-muted) !important;
    font-weight: 700 !important;
}

.studio-controls-row {
    margin-top: 12px !important;
    padding: 10px 12px !important;
    border: 1px solid var(--wb-border) !important;
    border-radius: var(--wb-radius) !important;
    background: var(--wb-panel) !important;
    gap: 14px !important;
    align-items: center !important;
}

.gradio-container details {
    margin-top: 12px !important;
    overflow: hidden !important;
}

.gradio-container details > summary,
.gradio-container .gr-accordion > summary,
.gradio-container .label-wrap {
    padding: 11px 13px !important;
    border-bottom-color: var(--wb-border) !important;
    background: #151b24 !important;
    color: var(--wb-text) !important;
    font-weight: 760 !important;
}

.gradio-container table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    background: transparent !important;
}

.gradio-container table th {
    background: #151b24 !important;
    color: var(--wb-muted) !important;
    font-size: 12px !important;
    font-weight: 760 !important;
    border-bottom: 1px solid var(--wb-border) !important;
}

.gradio-container table td {
    color: var(--wb-text) !important;
    border-bottom: 1px solid var(--wb-border) !important;
}

.gradio-container .prose,
.gradio-container .md,
.gradio-container .info-text,
.gradio-container [data-testid="info"],
.gradio-container .gr-info {
    color: var(--wb-muted) !important;
    font-size: 12.5px !important;
}

.gradio-container .prose strong,
.gradio-container .md strong { color: var(--wb-text) !important; }
.gradio-container .prose a,
.gradio-container .md a { color: #93c5fd !important; text-decoration: none !important; }

.premium-warning {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 9px 12px;
    border: 1px solid rgba(245,158,11,0.36);
    border-radius: var(--wb-radius-sm);
    background: rgba(245,158,11,0.10);
    color: #fcd34d;
    font-size: 12.5px;
    font-weight: 700;
}

.premium-warning__dot {
    width: 7px;
    height: 7px;
    flex: 0 0 auto;
    border-radius: 999px;
    background: var(--wb-amber);
}

.examples-accordion table,
.examples-accordion .table-wrap { font-size: 12px !important; }

.console-footer {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 20px;
    padding-top: 12px;
    border-top: 1px solid var(--wb-border);
    color: var(--wb-faint);
    font-family: var(--wb-mono);
    font-size: 11px;
    letter-spacing: 0 !important;
}

.gradio-container ::-webkit-scrollbar { width: 10px; height: 10px; }
.gradio-container ::-webkit-scrollbar-track { background: transparent; }
.gradio-container ::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.22);
    border: 2px solid transparent;
    border-radius: 999px;
    background-clip: padding-box;
}
.gradio-container ::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.34); background-clip: padding-box; }

@media (max-width: 1180px) {
    .studio-command-grid,
    .gradio-container .studio-command-grid { grid-template-columns: 1fr !important; }
    .console-hero { align-items: flex-start; flex-direction: column; }
    .console-status { justify-content: flex-start; }
    .console-subtitle { white-space: normal; }
}

@media (max-width: 720px) {
    .gradio-container { padding: 12px 12px 24px !important; }
    .console-hero { min-height: 0; padding: 11px 12px; }
    .console-title { font-size: 18px; }
    .console-status span { font-size: 10px; }
    .studio-card,
    .studio-command-grid > .block,
    .studio-command-grid > div { padding: 12px !important; }
    .console-footer { flex-direction: column; }
}
"""

PREMIUM_HERO_HTML = """
<div class="console-hero">
  <div class="console-brand">
    <div class="console-mark">IT</div>
    <div>
      <div class="console-title">IndexTTS-V26</div>
      <div class="console-subtitle">By 王知风</div>
    </div>
  </div>
  <div class="console-status">
    <span>LOCAL READY</span>
    <span>INDEXTTS 2</span>
    <span>CUDA 12.8</span>
    <a href="https://wangzhifeng.vip/" target="_blank" rel="noopener noreferrer">更多AI工具</a>
  </div>
</div>
"""

PREMIUM_FOOTER_HTML = """
<div class="console-footer">
  <span>IndexTTS-V26 / By 王知风</span>
  <span>Local inference package / V26</span>
</div>
"""
with gr.Blocks(title="IndexTTS-V26", theme=PREMIUM_THEME, css=PREMIUM_CSS) as demo:
    mutex = threading.Lock()
    novel_defaults = novel_llm_defaults()
    gr.HTML(PREMIUM_HERO_HTML)

    with gr.Tab(i18n("音频生成")):
        with gr.Row(elem_classes=["studio-command-grid"]):
            os.makedirs("prompts",exist_ok=True)
            with gr.Column(elem_classes=["voice-col", "studio-card", "studio-card--voice"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">Voice Asset</span><span class="studio-section-title__meta">REFERENCE</span></div>')
                prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                        sources=["upload","microphone"],type="filepath")
                with gr.Row(elem_classes=["voice-lib-row"]):
                    voice_save_name = gr.Textbox(
                        show_label=False, placeholder=i18n("给当前音色起个名字"),
                        scale=1, container=False, elem_classes=["voice-name-input"],
                    )
                    voice_save_btn = gr.Button(i18n("保存音色"), scale=1, size="sm", elem_classes=["voice-btn-save"])
                with gr.Row(elem_classes=["voice-lib-row"]):
                    voice_dropdown = gr.Dropdown(
                        choices=voice_lib_list(), show_label=False,
                        value=None, scale=3, container=False,
                        allow_custom_value=False,
                    )
                    voice_load_btn = gr.Button(i18n("加载"), scale=1, size="sm")
                    voice_delete_btn = gr.Button(i18n("删除"), scale=1, size="sm", elem_classes=["voice-btn-danger"])
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column(elem_classes=["studio-card", "studio-card--compose"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">Script Composer</span><span class="studio-section-title__meta">TEXT TO SPEECH</span></div>')
                input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button", interactive=True, variant="primary", size="lg", elem_id="gen_button_wrap")
            with gr.Column(elem_classes=["studio-card", "studio-card--output"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">Synthesis Result</span><span class="studio-section-title__meta">OUTPUT</span></div>')
                output_audio = gr.Audio(label=i18n("生成结果"), visible=True, key="output_audio", show_download_button=True)

        with gr.Row(elem_classes=["studio-controls-row"]):
            experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
            glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary)
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)  # do not render
        # 情感参考音频部分
        with gr.Group(visible=False, elem_classes=["emo-control-narrow"]) as emotion_reference_group:
            with gr.Column(elem_classes=["voice-col"]):
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")
                with gr.Row(elem_classes=["voice-lib-row"]):
                    emo_save_name = gr.Textbox(
                        show_label=False, placeholder=i18n("给当前情感音色起个名字"),
                        scale=1, container=False, elem_classes=["voice-name-input"],
                    )
                    emo_save_btn = gr.Button(i18n("保存音色"), scale=1, size="sm", elem_classes=["voice-btn-save"])
                with gr.Row(elem_classes=["voice-lib-row"]):
                    emo_dropdown = gr.Dropdown(
                        choices=voice_lib_list(), show_label=False,
                        value=None, scale=3, container=False,
                        allow_custom_value=False,
                    )
                    emo_load_btn = gr.Button(i18n("加载"), scale=1, size="sm")
                    emo_delete_btn = gr.Button(i18n("删除"), scale=1, size="sm", elem_classes=["voice-btn-danger"])

        # 情感随机采样
        with gr.Row(visible=False, elem_classes=["emo-control-narrow"]) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False, elem_classes=["emo-control-narrow"]) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

        with gr.Row(visible=False, elem_classes=["emo-control-narrow"]) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        # 术语词汇表管理
        with gr.Accordion(i18n("自定义术语词汇读音"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(
                        label=i18n("术语"),
                        placeholder="IndexTTS2",
                        lines=1, max_lines=1,
                    )
                    glossary_reading_zh = gr.Textbox(
                        label=i18n("中文读法"),
                        placeholder="Index T-T-S 二",
                        lines=1, max_lines=1,
                    )
                    glossary_reading_en = gr.Textbox(
                        label=i18n("英文读法"),
                        placeholder="Index T-T-S two",
                        lines=1, max_lines=1,
                    )
                    btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(
                        value=format_glossary_markdown()
                    )

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
        # binds tightly to UI and always restores the initial state of all components,
        # such as the list of available choices in emo_control_method.
        with gr.Accordion(i18n("示例 Examples"), open=False, elem_classes=["examples-accordion"]):
            example_table = gr.Dataset(label="Examples",
                samples_per_page=20,
                samples=get_example_cases(include_experimental=False),
                type="values",
                # these components are NOT "connected". it just reads the column labels/available
                # states from them, so we MUST link to the "all options" versions of all components,
                # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
                components=[prompt_audio,
                            emo_control_method_all,  # important: support all mode labels!
                            input_text_single,
                            emo_upload,
                            emo_weight,
                            emo_text,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            )

    with gr.Tab(i18n("多人对话")):
        with gr.Row(elem_classes=["studio-command-grid"]):
            with gr.Column(elem_classes=["studio-card", "studio-card--compose"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">Dialogue Script</span><span class="studio-section-title__meta">MULTI ROLE</span></div>')
                dialogue_script = gr.TextArea(
                    label=i18n("对话脚本"),
                    lines=14,
                    value="旁白：夜色已经很深了。\n小明：你怎么还没睡？\n小红：我在等一个答案。\n小明：什么答案？\n小红：你明天还会来吗？",
                    placeholder=i18n("每行一个角色台词，例如：小明：你好"),
                )
                with gr.Row():
                    dialogue_parse_btn = gr.Button(i18n("解析角色"), size="sm")
                    dialogue_generate_btn = gr.Button(i18n("生成多人对话"), variant="primary", size="lg")
            with gr.Column(elem_classes=["studio-card", "studio-card--voice"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">Role Voices</span><span class="studio-section-title__meta">VOICE LIBRARY</span></div>')
                dialogue_parse_roles_btn = gr.Button(i18n("解析角色并填入下表"), size="sm", variant="secondary")
                dialogue_role_rows = []
                dialogue_role_inputs = []
                dialogue_voice_dropdowns = []
                dialogue_emotion_dropdowns = []
                initial_dialogue_voices = voice_lib_list()
                for slot_idx in range(DIALOGUE_ROLE_SLOT_COUNT):
                    with gr.Row(visible=False) as role_row:
                        role_input = gr.Textbox(
                            label=i18n("角色"),
                            container=True,
                            max_lines=1,
                            scale=1,
                            min_width=160,
                        )
                        role_voice_dd = gr.Dropdown(
                            choices=initial_dialogue_voices,
                            label=i18n("音色"),
                            value=None,
                            allow_custom_value=True,
                            container=True,
                            scale=1,
                            min_width=160,
                        )
                        role_emotion_dd = gr.Dropdown(
                            choices=DIALOGUE_EMOTION_CHOICES,
                            label=i18n("情绪"),
                            value="默认",
                            allow_custom_value=False,
                            container=True,
                            scale=1,
                            min_width=120,
                        )
                    dialogue_role_rows.append(role_row)
                    dialogue_role_inputs.append(role_input)
                    dialogue_voice_dropdowns.append(role_voice_dd)
                    dialogue_emotion_dropdowns.append(role_emotion_dd)
                dialogue_interval = gr.Slider(
                    label=i18n("句间停顿 ms"),
                    minimum=0,
                    maximum=2000,
                    value=450,
                    step=50,
                )
                gr.Markdown(i18n("解析角色后，为每个角色选择音色和情绪。音色也可手动填写 wav/mp3/flac 文件路径。"))
            with gr.Column(elem_classes=["studio-card", "studio-card--output"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">Dialogue Result</span><span class="studio-section-title__meta">OUTPUT</span></div>')
                dialogue_output_audio = gr.Audio(label=i18n("完整对话音频"), visible=True, show_download_button=True)
                dialogue_status = gr.Textbox(label=i18n("状态"), interactive=False)

        with gr.Accordion(i18n("解析预览"), open=True):
            dialogue_preview = gr.Dataframe(
                headers=[i18n("序号"), i18n("角色"), i18n("台词")],
                wrap=True,
                interactive=False,
            )
        with gr.Accordion(i18n("生成明细"), open=False):
            dialogue_manifest = gr.Dataframe(
                headers=[i18n("序号"), i18n("角色"), i18n("情绪"), i18n("台词"), i18n("音频文件")],
                wrap=True,
                interactive=False,
            )

    with gr.Tab(i18n("小说项目")):
        with gr.Row(elem_classes=["studio-command-grid"]):
            with gr.Column(elem_classes=["studio-card", "studio-card--compose"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">Project Intake</span><span class="studio-section-title__meta">TXT TO SCRIPT</span></div>')
                novel_project_name = gr.Textbox(label=i18n("项目名"), placeholder=i18n("例如：贩罪 第二卷"), max_lines=1)
                novel_input_path = gr.Textbox(label=i18n("TXT 文件或文件夹路径"), placeholder=r"X:\tts\book")
                with gr.Row():
                    novel_create_btn = gr.Button(i18n("新建/导入项目"), variant="primary")
                    novel_refresh_btn = gr.Button(i18n("刷新项目"))
            with gr.Column(elem_classes=["studio-card", "studio-card--voice"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">LLM Settings</span><span class="studio-section-title__meta">STRUCTURED OUTPUT</span></div>')
                novel_api_base = gr.Textbox(label="API Base URL", value=novel_defaults.get("api_base_url", ""), max_lines=1)
                novel_api_key = gr.Textbox(label="API Key", value="", type="password", max_lines=1, placeholder=i18n("留空则使用本地配置或环境变量"))
                novel_model = gr.Textbox(label=i18n("预处理模型"), value=novel_defaults.get("preprocess_model", "gpt-5.4-mini"), max_lines=1)
                with gr.Row():
                    novel_chunk_target = gr.Number(label=i18n("目标切块字数"), value=NOVEL_DEFAULT_CHUNK_TARGET, precision=0)
                    novel_chunk_max = gr.Number(label=i18n("弹性最大字数"), value=NOVEL_DEFAULT_CHUNK_MAX, precision=0)
            with gr.Column(elem_classes=["studio-card", "studio-card--output"]):
                gr.HTML('<div class="studio-section-title"><span class="studio-section-title__main">Project Status</span><span class="studio-section-title__meta">QUEUE</span></div>')
                novel_project_dropdown = gr.Dropdown(label=i18n("当前项目"), choices=novel_project_choices(), value=None, allow_custom_value=False)
                novel_status = gr.Textbox(label=i18n("状态"), interactive=False, lines=6)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Accordion(i18n("章节任务"), open=True):
                    novel_tasks_df = gr.Dataframe(
                        headers=[i18n("序号"), "Task ID", i18n("标题"), i18n("状态"), i18n("原文字数"), i18n("脚本行数"), i18n("音频")],
                        wrap=True,
                        interactive=False,
                    )
                    novel_task_select = gr.Dropdown(label=i18n("选择任务"), choices=[], multiselect=True, allow_custom_value=True)
                    with gr.Row():
                        novel_load_project_btn = gr.Button(i18n("加载项目"))
                        novel_preprocess_selected_btn = gr.Button(i18n("预处理选中"))
                        novel_preprocess_all_btn = gr.Button(i18n("预处理全部未处理"))
                        novel_synth_selected_btn = gr.Button(i18n("合成选中"), variant="primary")
            with gr.Column(scale=2):
                with gr.Accordion(i18n("项目角色音色"), open=True):
                    novel_chars_df = gr.Dataframe(
                        headers=[i18n("角色"), i18n("出现次数"), i18n("性别 m/f/u"), i18n("音色"), i18n("备注")],
                        wrap=True,
                        interactive=True,
                    )
                    with gr.Row():
                        novel_save_chars_btn = gr.Button(i18n("保存角色音色"))
                        novel_auto_voice_btn = gr.Button(i18n("自动匹配音色"))

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Accordion(i18n("单章脚本与逐句情绪"), open=True):
                    novel_preview_task = gr.Dropdown(label=i18n("预览任务"), choices=[], allow_custom_value=True)
                    with gr.Row():
                        novel_load_script_btn = gr.Button(i18n("加载脚本"))
                        novel_save_script_btn = gr.Button(i18n("保存脚本修改"))
                    novel_script_df = gr.Dataframe(
                        headers=[i18n("序号"), i18n("角色"), i18n("情绪"), i18n("文本")],
                        wrap=True,
                        interactive=True,
                    )
                    novel_dialogue_text = gr.TextArea(label=i18n("IndexTTS 对话文本"), lines=12, interactive=False)
            with gr.Column(scale=2):
                with gr.Accordion(i18n("小说合成参数"), open=True):
                    novel_interval = gr.Slider(label=i18n("句间停顿 ms"), minimum=0, maximum=2000, value=450, step=50)
                    novel_output_audio = gr.Audio(label=i18n("章节音频"), visible=True, show_download_button=True)
                    novel_manifest_df = gr.Dataframe(
                        headers=[i18n("序号"), i18n("角色"), i18n("情绪"), i18n("台词"), i18n("音频文件")],
                        wrap=True,
                        interactive=False,
                    )

    gr.HTML(PREMIUM_FOOTER_HTML)

    def on_example_click(example):
        print(f"Example clicked: ({len(example)} values) = {example!r}")
        return (
            gr.update(value=example[0]),
            gr.update(value=example[1]),
            gr.update(value=example[2]),
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        )

    # click() event works on both desktop and mobile UI
    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=[prompt_audio,
                                 emo_control_method,
                                 input_text_single,
                                 emo_upload,
                                 emo_weight,
                                 emo_text,
                                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    # 术语词汇表事件处理函数
    def on_add_glossary_term(term, reading_zh, reading_en):
        """添加术语到词汇表并自动保存"""
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()

        if not term:
            gr.Warning(i18n("请输入术语"))
            return gr.update()
            
        if not reading_zh and not reading_en:
            gr.Warning(i18n("请至少输入一种读法"))
            return gr.update()
        

        # 构建读法数据
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        elif reading_en:
            reading = {"en": reading_en}
        else:
            reading = reading_zh or reading_en

        # 添加到词汇表
        tts.normalizer.term_glossary[term] = reading

        # 自动保存到文件
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(i18n("词汇表已更新"), duration=1)
        except Exception as e:
            gr.Error(i18n("保存词汇表时出错"))
            print(f"Error details: {e}")
            return gr.update()

        # 更新Markdown表格
        return gr.update(value=format_glossary_markdown())
        

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    def project_label(meta):
        return meta.get("id", "")

    def pack_novel_project_outputs(meta, status):
        choices = task_choices(meta) if meta else []
        preview_value = choices[0] if choices else None
        return (
            gr.update(value=novel_tasks_table(meta)),
            gr.update(choices=choices, value=[]),
            gr.update(choices=choices, value=preview_value),
            gr.update(value=novel_chars_table(meta)),
            gr.update(value=status),
        )

    def novel_create_project_ui(project_name, input_path, api_base, api_key, model):
        try:
            meta = create_novel_project(project_name, input_path, api_base, api_key, model)
            status = f"项目已创建：{meta['name']}，导入 {len(meta.get('tasks', []))} 个 txt。"
            return (
                gr.update(choices=novel_project_choices(), value=project_label(meta)),
                *pack_novel_project_outputs(meta, status),
            )
        except Exception as e:
            return (
                gr.update(choices=novel_project_choices()),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value="创建失败：" + str(e)),
            )

    def novel_refresh_projects_ui():
        choices = novel_project_choices()
        value = choices[0] if choices else None
        return gr.update(choices=choices, value=value), gr.update(value=f"已刷新，项目数：{len(choices)}")

    def novel_load_project_ui(project_choice):
        project_id = parse_project_choice(project_choice)
        meta = load_project(project_id)
        if not meta:
            return pack_novel_project_outputs(None, "请选择项目")
        return pack_novel_project_outputs(meta, f"已加载项目：{meta.get('name', project_id)}")

    def novel_save_chars_ui(project_choice, table):
        project_id = parse_project_choice(project_choice)
        try:
            meta = save_project_characters_from_table(project_id, table)
            return gr.update(value=novel_chars_table(meta)), gr.update(value="角色音色已保存。")
        except Exception as e:
            return gr.update(), gr.update(value="保存失败：" + str(e))

    def novel_auto_voice_ui(project_choice, table):
        project_id = parse_project_choice(project_choice)
        try:
            meta = save_project_characters_from_table(project_id, table)
            used = set()
            for role, info in meta.get("characters", {}).items():
                if info.get("voice"):
                    used.add(info["voice"])
            for role, info in meta.get("characters", {}).items():
                if not info.get("voice"):
                    voice = auto_match_voice(role, info.get("gender", "u"), used)
                    if voice:
                        info["voice"] = voice
                        used.add(voice)
            save_project(meta)
            return gr.update(value=novel_chars_table(meta)), gr.update(value="已自动匹配空缺音色，可继续手动调整。")
        except Exception as e:
            return gr.update(), gr.update(value="自动匹配失败：" + str(e))

    def novel_preprocess_ui(project_choice, selected_tasks, api_base, api_key, model, target_chars, max_chars, only_new, progress=gr.Progress()):
        project_id = parse_project_choice(project_choice)
        meta = load_project(project_id)
        if not meta:
            return (*pack_novel_project_outputs(None, "请选择项目"), gr.update(), gr.update())
        valid_ids = [task["id"] for task in meta.get("tasks", [])]
        selected_ids = parse_task_choices(selected_tasks, valid_ids=valid_ids)
        if only_new:
            selected_ids = [task["id"] for task in meta.get("tasks", []) if task.get("status") in ("new", "", None)]
        if not selected_ids:
            return (*pack_novel_project_outputs(meta, "没有可预处理的任务。"), gr.update(), gr.update())
        logs = []
        last_script = None
        try:
            for idx, task_id in enumerate(selected_ids, 1):
                task = find_task(meta, task_id)
                if not task:
                    logs.append(f"{task_id}: 任务不存在，已跳过。")
                    continue
                title = task.get("title", task_id)
                progress(idx / max(1, len(selected_ids)), desc=f"预处理 {idx}/{len(selected_ids)} {title}")
                meta, script = preprocess_novel_task(project_id, task_id, api_base, api_key, model, target_chars, max_chars, progress=progress)
                last_script = script
                warnings = script.get("warnings", [])
                coverage_count = len([item for item in warnings if "覆盖校验不一致" in item])
                repair_count = len([item for item in warnings if "自动补回" in item])
                logs.append(f"{title}: {script.get('chunk_count', 0)} 块，{len(script.get('lines', []))} 行，覆盖不一致 {coverage_count} 条，自动补回 {repair_count} 条。")
        except Exception as e:
            latest = load_project(project_id) or meta
            status = "预处理失败：" + str(e)
            if logs:
                status += "\n已完成：\n" + "\n".join(logs)
            return (
                *pack_novel_project_outputs(latest, status),
                gr.update(value=script_rows_table(last_script) if last_script else []),
                gr.update(value="\n".join(f"{line['spk']}：{line['txt']}" for line in last_script.get("lines", [])) if last_script else ""),
            )
        if not last_script:
            status = "没有任务被成功预处理。\n" + "\n".join(logs)
            return (*pack_novel_project_outputs(meta, status), gr.update(value=[]), gr.update(value=""))
        status = "预处理完成。\n" + "\n".join(logs)
        return (
            *pack_novel_project_outputs(meta, status),
            gr.update(value=script_rows_table(last_script)),
            gr.update(value="\n".join(f"{line['spk']}：{line['txt']}" for line in last_script.get("lines", []))),
        )

    def novel_load_script_ui(project_choice, task_choice):
        project_id = parse_project_choice(project_choice)
        meta = load_project(project_id)
        valid_ids = [task["id"] for task in meta.get("tasks", [])] if meta else []
        task_id = parse_task_choice(task_choice, valid_ids=valid_ids)
        if not project_id or not task_id:
            return gr.update(value=[]), gr.update(value=""), gr.update(value="请选择要预览的任务。")
        script_path = os.path.join(task_dir(project_id, task_id), NOVEL_SCRIPT_JSON)
        script = read_json_file(script_path, {})
        if not script:
            return gr.update(value=[]), gr.update(value=""), gr.update(value="这个任务还没有预处理脚本。")
        dialogue = "\n".join(f"{line['spk']}：{line['txt']}" for line in script.get("lines", []))
        warn = "\n".join(script.get("warnings", []))
        return gr.update(value=script_rows_table(script)), gr.update(value=dialogue), gr.update(value=warn or "脚本已加载。")

    def novel_save_script_ui(project_choice, task_choice, table):
        project_id = parse_project_choice(project_choice)
        meta = load_project(project_id)
        valid_ids = [task["id"] for task in meta.get("tasks", [])] if meta else []
        task_id = parse_task_choice(task_choice, valid_ids=valid_ids)
        try:
            meta, script = save_task_script_from_table(project_id, task_id, table)
            dialogue = "\n".join(f"{line['spk']}：{line['txt']}" for line in script.get("lines", []))
            return (
                gr.update(value=script_rows_table(script)),
                gr.update(value=dialogue),
                gr.update(value=novel_chars_table(meta)),
                gr.update(value="脚本修改已保存，项目角色统计已重新同步。"),
            )
        except Exception as e:
            return gr.update(), gr.update(), gr.update(), gr.update(value="保存脚本失败：" + str(e))

    def novel_synth_selected_ui(project_choice, selected_tasks, interval_ms, novel_seg_tokens, *generation_args, progress=gr.Progress()):
        project_id = parse_project_choice(project_choice)
        meta = load_project(project_id)
        if not meta:
            return (*pack_novel_project_outputs(None, "请选择项目"), gr.update(value=None), gr.update(value=[]))
        valid_ids = [task["id"] for task in meta.get("tasks", [])]
        selected_ids = parse_task_choices(selected_tasks, valid_ids=valid_ids)
        if not selected_ids:
            return (*pack_novel_project_outputs(meta, "请至少选择一个已预处理任务。"), gr.update(value=None), gr.update(value=[]))
        logs = []
        last_audio = None
        last_manifest = []
        try:
            with mutex:
                for idx, task_id in enumerate(selected_ids, 1):
                    task = find_task(meta, task_id)
                    title = task.get("title", task_id) if task else task_id
                    progress(idx / max(1, len(selected_ids)), desc=f"合成 {idx}/{len(selected_ids)} {title}")
                    meta, audio_path, manifest_rows = synthesize_novel_task(
                        project_id,
                        task_id,
                        interval_ms,
                        novel_seg_tokens,
                        generation_args,
                        progress=progress,
                    )
                    last_audio = audio_path
                    last_manifest = manifest_rows
                    logs.append(f"{title}: {audio_path}")
            status = "合成完成。\n" + "\n".join(logs)
            return (*pack_novel_project_outputs(meta, status), gr.update(value=last_audio, visible=True), gr.update(value=last_manifest))
        except Exception as e:
            latest = load_project(project_id) or meta
            return (*pack_novel_project_outputs(latest, "合成失败：" + str(e)), gr.update(value=last_audio), gr.update(value=last_manifest))

    novel_create_btn.click(
        novel_create_project_ui,
        inputs=[novel_project_name, novel_input_path, novel_api_base, novel_api_key, novel_model],
        outputs=[novel_project_dropdown, novel_tasks_df, novel_task_select, novel_preview_task, novel_chars_df, novel_status],
    )
    novel_refresh_btn.click(
        novel_refresh_projects_ui,
        inputs=[],
        outputs=[novel_project_dropdown, novel_status],
    )
    novel_load_project_btn.click(
        novel_load_project_ui,
        inputs=[novel_project_dropdown],
        outputs=[novel_tasks_df, novel_task_select, novel_preview_task, novel_chars_df, novel_status],
    )
    novel_project_dropdown.change(
        novel_load_project_ui,
        inputs=[novel_project_dropdown],
        outputs=[novel_tasks_df, novel_task_select, novel_preview_task, novel_chars_df, novel_status],
    )
    novel_save_chars_btn.click(
        novel_save_chars_ui,
        inputs=[novel_project_dropdown, novel_chars_df],
        outputs=[novel_chars_df, novel_status],
    )
    novel_auto_voice_btn.click(
        novel_auto_voice_ui,
        inputs=[novel_project_dropdown, novel_chars_df],
        outputs=[novel_chars_df, novel_status],
    )
    novel_preprocess_selected_btn.click(
        lambda project_choice, selected_tasks, api_base, api_key, model, target_chars, max_chars, progress=gr.Progress():
            novel_preprocess_ui(project_choice, selected_tasks, api_base, api_key, model, target_chars, max_chars, False, progress),
        inputs=[novel_project_dropdown, novel_task_select, novel_api_base, novel_api_key, novel_model, novel_chunk_target, novel_chunk_max],
        outputs=[novel_tasks_df, novel_task_select, novel_preview_task, novel_chars_df, novel_status, novel_script_df, novel_dialogue_text],
    )
    novel_preprocess_all_btn.click(
        lambda project_choice, selected_tasks, api_base, api_key, model, target_chars, max_chars, progress=gr.Progress():
            novel_preprocess_ui(project_choice, selected_tasks, api_base, api_key, model, target_chars, max_chars, True, progress),
        inputs=[novel_project_dropdown, novel_task_select, novel_api_base, novel_api_key, novel_model, novel_chunk_target, novel_chunk_max],
        outputs=[novel_tasks_df, novel_task_select, novel_preview_task, novel_chars_df, novel_status, novel_script_df, novel_dialogue_text],
    )
    novel_load_script_btn.click(
        novel_load_script_ui,
        inputs=[novel_project_dropdown, novel_preview_task],
        outputs=[novel_script_df, novel_dialogue_text, novel_status],
    )
    novel_save_script_btn.click(
        novel_save_script_ui,
        inputs=[novel_project_dropdown, novel_preview_task, novel_script_df],
        outputs=[novel_script_df, novel_dialogue_text, novel_chars_df, novel_status],
    )
    novel_synth_selected_btn.click(
        novel_synth_selected_ui,
        inputs=[
            novel_project_dropdown,
            novel_task_select,
            novel_interval,
            max_text_tokens_per_segment,
            *advanced_params,
        ],
        outputs=[novel_tasks_df, novel_task_select, novel_preview_task, novel_chars_df, novel_status, novel_output_audio, novel_manifest_df],
    )
    demo.load(
        novel_refresh_projects_ui,
        inputs=[],
        outputs=[novel_project_dropdown, novel_status],
    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table]
    )

    def on_glossary_checkbox_change(is_enabled):
        """控制术语词汇表的可见性"""
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    glossary_checkbox.change(
        on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    # Voice library events — shared between speaker and emotion reference slots.
    voice_save_btn.click(
        on_save_to_library,
        inputs=[prompt_audio, voice_save_name],
        outputs=[voice_dropdown, emo_dropdown],
    )
    voice_load_btn.click(
        on_load_from_library,
        inputs=[voice_dropdown],
        outputs=[prompt_audio],
    )
    voice_delete_btn.click(
        on_delete_from_library,
        inputs=[voice_dropdown],
        outputs=[voice_dropdown, emo_dropdown],
    )

    emo_save_btn.click(
        on_save_to_library,
        inputs=[emo_upload, emo_save_name],
        outputs=[emo_dropdown, voice_dropdown],
    )
    emo_load_btn.click(
        on_load_from_library,
        inputs=[emo_dropdown],
        outputs=[emo_upload],
    )
    emo_delete_btn.click(
        on_delete_from_library,
        inputs=[emo_dropdown],
        outputs=[emo_dropdown, voice_dropdown],
    )

    def on_demo_load():
        """页面加载时重新加载glossary数据"""
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(i18n("加载词汇表时出错"))
            print(f"Failed to reload glossary on page load: {e}")
        return gr.update(value=format_glossary_markdown())

    # 术语词汇表事件绑定
    btn_add_term.click(
        on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table]
    )

    # 页面加载时重新加载glossary
    demo.load(
        on_demo_load,
        inputs=[],
        outputs=[glossary_table]
    )

    # 页面加载时刷新音色库下拉
    demo.load(
        on_refresh_library,
        inputs=[],
        outputs=[voice_dropdown, emo_dropdown],
    )

    dialogue_parse_btn.click(
        build_role_slots_from_script,
        inputs=[
            dialogue_script,
            *[component for pair in zip(dialogue_role_inputs, dialogue_voice_dropdowns, dialogue_emotion_dropdowns) for component in pair],
        ],
        outputs=[
            *[component for triple in zip(dialogue_role_rows, dialogue_role_inputs, dialogue_voice_dropdowns, dialogue_emotion_dropdowns) for component in triple],
            dialogue_preview,
        ],
    )

    dialogue_parse_roles_btn.click(
        build_role_slots_from_script,
        inputs=[
            dialogue_script,
            *[component for pair in zip(dialogue_role_inputs, dialogue_voice_dropdowns, dialogue_emotion_dropdowns) for component in pair],
        ],
        outputs=[
            *[component for triple in zip(dialogue_role_rows, dialogue_role_inputs, dialogue_voice_dropdowns, dialogue_emotion_dropdowns) for component in triple],
            dialogue_preview,
        ],
    )

    def gen_dialogue_locked(script, interval_ms, max_text_tokens_per_segment, *args, progress=gr.Progress()):
        slot_values = args[:DIALOGUE_ROLE_SLOT_COUNT * DIALOGUE_ROLE_SLOT_FIELD_COUNT]
        generation_args = args[DIALOGUE_ROLE_SLOT_COUNT * DIALOGUE_ROLE_SLOT_FIELD_COUNT:]
        role_mapping = build_role_voice_map_from_slots(*slot_values)
        with mutex:
            return gen_dialogue(
                script,
                role_mapping,
                interval_ms,
                max_text_tokens_per_segment,
                *generation_args,
                progress=progress,
            )

    dialogue_generate_btn.click(
        gen_dialogue_locked,
        inputs=[
            dialogue_script,
            dialogue_interval,
            max_text_tokens_per_segment,
            *[component for pair in zip(dialogue_role_inputs, dialogue_voice_dropdowns, dialogue_emotion_dropdowns) for component in pair],
            *advanced_params,
        ],
        outputs=[dialogue_output_audio, dialogue_manifest, dialogue_status],
    )

    demo.load(
        build_role_slots_from_script,
        inputs=[
            dialogue_script,
            *[component for pair in zip(dialogue_role_inputs, dialogue_voice_dropdowns, dialogue_emotion_dropdowns) for component in pair],
        ],
        outputs=[
            *[component for triple in zip(dialogue_role_rows, dialogue_role_inputs, dialogue_voice_dropdowns, dialogue_emotion_dropdowns) for component in triple],
            dialogue_preview,
        ],
    )

    demo.load(
        refresh_dialogue_voice_library,
        inputs=[],
        outputs=dialogue_voice_dropdowns + dialogue_emotion_dropdowns,
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, inbrowser=True,
                allowed_paths=[os.path.join(ROOT_DIR, "outputs")])




