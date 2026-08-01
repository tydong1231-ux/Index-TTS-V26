from __future__ import annotations

import difflib
import re
from typing import Any, Callable

from .llm import LLMConfig, OpenAICompatibleClient
from .store import ProjectStore
from .utils import utc_now

EMOTIONS = ["默认", "高兴", "愤怒", "悲伤", "恐惧", "厌恶", "低落", "惊喜", "平静"]
SYSTEM_PROMPT = """你是中文小说有声书脚本整理器。把原文严格拆成有序旁白与角色台词，供多人 TTS 使用。
1. 保留原文顺序、内容和标点，不改写、不总结、不扩写、不遗漏。
2. 对话只移除最外层成对引号；动作、心理、场景和说话引导语归入旁白。
3. 每条 segment 只能有一个说话人，旁白与对白必须拆开。
4. 同一人物使用稳定主名；声音明显不同的年龄时期用“主名 · 童年/老年”等变体名。
5. 无法确定人物时用“未知男人”“未知女人”或“未知角色”，并降低 confidence。
6. emotion 只能使用：默认、高兴、愤怒、悲伤、恐惧、厌恶、低落、惊喜、平静。
7. confidence 是 0-100 的说话人判断置信度；低于 85 时 needs_review=true。
8. roles 只返回本块新增或补充的人物信息；traits 用短词描述声音匹配相关特征。
9. memory 最多 120 个中文字符，只用于下一块消解人物，不能替代正文。
10. 只输出符合 JSON Schema 的 JSON。"""


def response_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": ["memory", "roles", "segments"], "properties": {"memory": {"type": "string"}, "roles": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "gender", "age_stage", "traits", "aliases", "note"], "properties": {"name": {"type": "string"}, "gender": {"type": "string", "enum": ["m", "f", "u"]}, "age_stage": {"type": "string"}, "traits": {"type": "array", "items": {"type": "string"}}, "aliases": {"type": "array", "items": {"type": "string"}}, "note": {"type": "string"}}}}, "segments": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["speaker", "text", "emotion", "confidence", "needs_review"], "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}, "emotion": {"type": "string", "enum": EMOTIONS}, "confidence": {"type": "integer", "minimum": 0, "maximum": 100}, "needs_review": {"type": "boolean"}}}}}}


def split_novel_text(text: str, target_chars: int = 500, max_chars: int = 800) -> list[str]:
    target_chars, max_chars = max(180, int(target_chars)), max(max(180, int(target_chars)), int(max_chars))
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs or [normalized]:
        for unit in _split_long_unit(paragraph, target_chars, max_chars):
            separator = "\n\n" if current else ""
            if current and (len(current) + len(separator) + len(unit) > max_chars or len(current) >= target_chars):
                chunks.append(current.strip())
                current = unit
            else:
                current = current + separator + unit if current else unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_long_unit(text: str, target_chars: int, max_chars: int) -> list[str]:
    remaining, parts, punctuation = text.strip(), [], "。！？!?；;，,"
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = max(window.rfind(mark) for mark in punctuation)
        if cut < max(80, int(target_chars * 0.45)):
            cut = max_chars - 1
        parts.append(remaining[:cut + 1].strip())
        remaining = remaining[cut + 1:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def normalize_dialogue_text(text: Any) -> str:
    value = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", "", str(text or "")).strip()
    pairs = {"“": "”", "‘": "’", '"': '"', "'": "'", "「": "」", "『": "』"}
    while len(value) >= 2 and value[0] in pairs and value[-1] == pairs[value[0]]:
        value = value[1:-1].strip()
    return value


def normalize_coverage(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).translate(str.maketrans("", "", "“”‘’\"'「」『』"))


def _normalized_with_map(text: Any) -> tuple[str, list[int]]:
    chars, indexes, ignored = [], [], set("“”‘’\"'「」『』")
    for index, char in enumerate(str(text or "")):
        if not char.isspace() and char not in ignored:
            chars.append(char)
            indexes.append(index)
    return "".join(chars), indexes


def repair_missing_fragments(source_text: str, segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    source_norm, source_map = _normalized_with_map(source_text)
    segment_norms = [_normalized_with_map(segment.get("text", ""))[0] for segment in segments]
    output_norm = "".join(segment_norms)
    if not source_norm or source_norm == output_norm:
        return segments, []
    repaired, missing, inserted = [dict(segment) for segment in segments], [], 0
    for tag, i1, i2, j1, _ in difflib.SequenceMatcher(a=source_norm, b=output_norm, autojunk=False).get_opcodes():
        if tag != "delete" or i1 >= i2:
            continue
        fragment = source_text[source_map[i1]:source_map[i2 - 1] + 1].strip()
        if len(normalize_coverage(fragment)) < 2:
            continue
        insert_at, consumed = 0, 0
        for segment_norm in segment_norms:
            if consumed + len(segment_norm) <= j1:
                insert_at += 1
                consumed += len(segment_norm)
            else:
                break
        repaired.insert(min(len(repaired), insert_at + inserted), {"speaker": "旁白", "text": fragment, "emotion": "平静", "confidence": 100, "needs_review": True, "repair_reason": "LLM omitted source text; automatically restored as narration"})
        inserted += 1
        missing.append(fragment)
    return repaired, missing


class PreprocessService:
    def __init__(self, store: ProjectStore):
        self.store = store

    def preprocess_chapter(self, project_id: str, chapter_id: str, llm_config: LLMConfig, force: bool = False, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
        project, chapter = self.store.get_project(project_id), self.store.get_chapter(project_id, chapter_id)
        if chapter.get("script") and not force:
            return chapter["script"]
        settings, source = project.get("settings", {}), chapter.get("source_text", "")
        chunks = split_novel_text(source, int(settings.get("chunk_target_chars", 500)), int(settings.get("chunk_max_chars", 800)))
        if not chunks:
            raise ValueError("Chapter source is empty")
        client, memory = OpenAICompatibleClient(llm_config), ""
        known_roles, all_segments, all_roles, warnings, usage_total = list(project.get("roles", {}).keys()), [], {}, [], {}
        for chunk_index, chunk in enumerate(chunks, 1):
            if progress:
                progress((chunk_index - 1) / len(chunks), f"LLM preprocessing {chunk_index}/{len(chunks)}")
            prompt = f"<memory>{memory}</memory>\n<known_roles>{'、'.join(known_roles[:120])}</known_roles>\n<source>\n{chunk}\n</source>"
            result, usage = client.generate_json(SYSTEM_PROMPT, prompt, "voicebook_novel_chunk", response_schema())
            memory = str(result.get("memory", ""))[:120]
            chunk_segments = []
            for raw in result.get("segments", []):
                speaker, text = str(raw.get("speaker") or "旁白").strip() or "旁白", normalize_dialogue_text(raw.get("text"))
                if not text:
                    continue
                confidence = max(0, min(100, int(raw.get("confidence", 90))))
                emotion = str(raw.get("emotion") or "默认")
                if emotion not in EMOTIONS:
                    emotion = "默认"
                chunk_segments.append({"speaker": speaker, "text": text, "emotion": emotion, "confidence": confidence, "needs_review": bool(raw.get("needs_review")) or confidence < 85 or speaker.startswith("未知")})
                if speaker not in known_roles:
                    known_roles.append(speaker)
            chunk_segments, repaired = repair_missing_fragments(chunk, chunk_segments)
            if repaired:
                warnings.append(f"Chunk {chunk_index}: restored {len(repaired)} omitted fragment(s) as narration")
            if normalize_coverage("".join(item["text"] for item in chunk_segments)) != normalize_coverage(chunk):
                warnings.append(f"Chunk {chunk_index}: source coverage differs and needs review")
            for role in result.get("roles", []):
                name = str(role.get("name") or "").strip()
                if name:
                    all_roles[name] = {"name": name, "gender": role.get("gender") if role.get("gender") in {"m", "f", "u"} else "u", "age_stage": str(role.get("age_stage") or "adult"), "traits": [str(item).strip() for item in role.get("traits", []) if str(item).strip()], "aliases": [str(item).strip() for item in role.get("aliases", []) if str(item).strip()], "note": str(role.get("note") or "").strip()}
            for key, value in (usage or {}).items():
                if isinstance(value, int):
                    usage_total[key] = usage_total.get(key, 0) + value
            all_segments.extend(chunk_segments)
        for index, segment in enumerate(all_segments, 1):
            segment["id"], segment["order"] = f"{index:04d}", index
        script = {"chapter_id": chapter_id, "title": chapter.get("title", ""), "model": llm_config.model, "created_at": utc_now(), "updated_at": utc_now(), "source_chars": len(source), "chunk_count": len(chunks), "roles": list(all_roles.values()), "segments": all_segments, "warnings": warnings, "usage": usage_total}
        self.store.save_script(project_id, chapter_id, script)
        review_count = sum(1 for segment in all_segments if segment.get("needs_review"))
        self.store.update_chapter(project_id, chapter_id, {"status": "attention" if review_count or warnings else "ready", "segment_count": len(all_segments), "role_count": len({segment["speaker"] for segment in all_segments}), "warning_count": len(warnings), "stages": {"script": "attention" if review_count or warnings else "done", "voices": "pending", "generate": "pending", "qa": "pending"}})
        self.store.sync_roles(project_id)
        if progress:
            progress(1.0, "LLM preprocessing complete")
        return script
