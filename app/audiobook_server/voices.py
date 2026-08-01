from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .store import ProjectStore
from .utils import slugify

VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

GENDER_TERMS = {
    "m": ("男", "男声", "少年", "青年", "叔", "大叔", "老年男", "爷爷", "父亲", "哥哥"),
    "f": ("女", "女声", "少女", "御姐", "温柔女", "老年女", "奶奶", "母亲", "姐姐"),
}
AGE_TERMS = {
    "child": ("儿童", "童年", "孩童", "幼年", "小孩", "童声"),
    "teen": ("少年", "少女", "青春", "十几岁"),
    "young": ("青年", "年轻", "少男", "少女", "御姐"),
    "middle": ("中年", "大叔", "成熟", "成熟女"),
    "old": ("老年", "老人", "爷爷", "奶奶", "苍老"),
    "adult": ("成年", "成人"),
}
STYLE_TERMS = {
    "narrator": ("旁白", "叙述", "讲述", "沉稳", "纪录片"),
    "broadcast": ("广播", "播音", "站内", "新闻"),
    "calm": ("冷静", "克制", "平静", "清冷"),
    "warm": ("温柔", "温暖", "治愈"),
    "bright": ("活泼", "明亮", "元气"),
    "deep": ("低沉", "厚重", "磁性", "沉稳"),
    "villain": ("反派", "阴沉", "危险", "沙哑"),
}


@dataclass
class VoiceDescriptor:
    name: str
    path: str
    gender: str
    ages: set[str]
    styles: set[str]
    tokens: set[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "gender": self.gender,
            "ages": sorted(self.ages),
            "styles": sorted(self.styles),
            "tokens": sorted(self.tokens),
        }


class VoiceLibrary:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.voice_library_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        return [descriptor.as_dict() for descriptor in self.descriptors()]

    def descriptors(self) -> list[VoiceDescriptor]:
        items: list[VoiceDescriptor] = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS:
                items.append(self.describe(path))
        return items

    def resolve(self, name_or_path: str) -> Path | None:
        value = str(name_or_path or "").strip()
        if not value:
            return None
        candidate = Path(value).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
        for path in self.root.iterdir():
            if path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS and path.stem == value:
                return path.resolve()
        return None

    def save_upload(self, filename: str, data: bytes, name: str | None = None) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in VOICE_EXTENSIONS:
            raise ValueError(f"Unsupported voice file type: {suffix}")
        stem = slugify(name or Path(filename).stem, "voice")
        destination = self.root / f"{stem}{suffix}"
        destination.write_bytes(data)
        return self.describe(destination).as_dict()

    def copy_file(self, source: Path, name: str | None = None) -> dict[str, Any]:
        source = source.expanduser().resolve()
        if not source.exists() or source.suffix.lower() not in VOICE_EXTENSIONS:
            raise ValueError("Voice file does not exist or has an unsupported extension")
        destination = self.root / f"{slugify(name or source.stem, 'voice')}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        return self.describe(destination).as_dict()

    @staticmethod
    def describe(path: Path) -> VoiceDescriptor:
        stem = path.stem
        normalized = stem.lower().replace("_", " ").replace("-", " ")
        gender = "u"
        for key, terms in GENDER_TERMS.items():
            if any(term.lower() in normalized for term in terms):
                gender = key
                break
        ages = {key for key, terms in AGE_TERMS.items() if any(term.lower() in normalized for term in terms)}
        styles = {key for key, terms in STYLE_TERMS.items() if any(term.lower() in normalized for term in terms)}
        tokens = {token for token in re.split(r"[\s_\-·]+", normalized) if token}
        return VoiceDescriptor(stem, str(path.resolve()), gender, ages, styles, tokens)


class VoiceMatcher:
    def __init__(self, store: ProjectStore, library: VoiceLibrary):
        self.store = store
        self.library = library

    def auto_match(self, project_id: str, overwrite_unlocked: bool = False) -> dict[str, Any]:
        project = self.store.get_project(project_id)
        roles = project.get("roles", {})
        voices = self.library.descriptors()
        if not voices:
            raise ValueError(f"No voice files found in {self.library.root}")
        used: set[str] = set()
        for role in roles.values():
            if role.get("voice") and role.get("voice_locked"):
                used.add(str(role["voice"]))
        ordered_roles = sorted(roles.values(), key=lambda role: (-int(role.get("count", 0)), role.get("name", "")))
        for role in ordered_roles:
            if role.get("voice_locked") and role.get("voice"):
                continue
            if role.get("voice") and not overwrite_unlocked:
                used.add(str(role["voice"]))
                continue
            scored = [self._score(role, voice, voice.name in used) for voice in voices]
            scored.sort(key=lambda item: (item[0], item[1].name), reverse=True)
            score, best, reasons = scored[0]
            role["voice"] = best.name
            role["voice_score"] = max(0, min(100, int(round(score))))
            role["voice_reason"] = "；".join(reasons[:4]) or "按可用音色顺序分配"
            used.add(best.name)
        self.store.save_project(project)
        self._update_voice_stage(project_id)
        return self.store.get_project(project_id)

    def _score(self, role: dict[str, Any], voice: VoiceDescriptor, already_used: bool) -> tuple[float, VoiceDescriptor, list[str]]:
        score = 20.0
        reasons: list[str] = []
        role_name = str(role.get("name") or "")
        if role_name and (role_name in voice.name or voice.name in role_name):
            score += 80
            reasons.append("名称直接匹配")
        if role_name == "旁白" or "旁白" in role_name:
            if "narrator" in voice.styles:
                score += 45
                reasons.append("旁白风格匹配")
            else:
                score -= 20
        gender = str(role.get("gender") or "u")
        if gender != "u":
            if voice.gender == gender:
                score += 24
                reasons.append("性别匹配")
            elif voice.gender != "u":
                score -= 36
        age = self._normalize_age(str(role.get("age_stage") or "adult"))
        if age in voice.ages:
            score += 24
            reasons.append("年龄阶段匹配")
        elif voice.ages and age not in {"adult", "unknown"}:
            score -= 14
        traits_text = " ".join(str(item) for item in role.get("traits", [])) + " " + str(role.get("note") or "")
        for style, terms in STYLE_TERMS.items():
            if any(term in traits_text for term in terms) and style in voice.styles:
                score += 11
                reasons.append(f"{style} 风格匹配")
        if already_used:
            score -= 48
            reasons.append("已被其他人物使用，降低优先级")
        if "未知" in role_name:
            score -= 5
        return score, voice, reasons

    @staticmethod
    def _normalize_age(value: str) -> str:
        value = value.lower()
        if any(term in value for term in ("童", "幼", "child")):
            return "child"
        if any(term in value for term in ("少年", "少女", "teen")):
            return "teen"
        if any(term in value for term in ("青年", "young")):
            return "young"
        if any(term in value for term in ("中年", "middle")):
            return "middle"
        if any(term in value for term in ("老", "old")):
            return "old"
        return "adult"

    def _update_voice_stage(self, project_id: str) -> None:
        project = self.store.get_project(project_id)
        all_ready = bool(project.get("roles")) and all(role.get("voice") for role in project.get("roles", {}).values())
        for chapter in project.get("chapters", []):
            if chapter.get("stages", {}).get("script") in {"done", "attention"}:
                chapter.setdefault("stages", {})["voices"] = "done" if all_ready else "attention"
                if not all_ready and chapter.get("status") == "ready":
                    chapter["status"] = "attention"
        self.store.save_project(project)
