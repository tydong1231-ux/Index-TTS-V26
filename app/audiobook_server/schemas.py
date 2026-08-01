from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

StageName = Literal["source", "script", "voices", "generate", "qa"]
GenerationMode = Literal["ordered", "grouped"]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_path: str | None = None
    api_base_url: str | None = None
    model: str | None = None


class ProjectSettingsUpdate(BaseModel):
    api_base_url: str | None = None
    model: str | None = None
    chunk_target_chars: int | None = Field(default=None, ge=180, le=4000)
    chunk_max_chars: int | None = Field(default=None, ge=180, le=8000)
    interval_ms: int | None = Field(default=None, ge=0, le=5000)
    max_text_tokens_per_segment: int | None = Field(default=None, ge=20, le=600)
    generation_mode: GenerationMode | None = None


class RoleUpdate(BaseModel):
    gender: Literal["m", "f", "u"] | None = None
    age_stage: str | None = None
    traits: list[str] | None = None
    aliases: list[str] | None = None
    voice: str | None = None
    voice_locked: bool | None = None
    note: str | None = None


class SegmentUpdate(BaseModel):
    speaker: str | None = None
    text: str | None = None
    emotion: str | None = None
    confidence: int | None = Field(default=None, ge=0, le=100)
    needs_review: bool | None = None


class LLMConnection(BaseModel):
    api_base_url: str | None = None
    api_key: SecretStr | None = None
    model: str | None = None
    endpoint: Literal["chat_completions", "responses"] = "chat_completions"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"


class JobCreate(LLMConnection):
    chapter_ids: list[str] = Field(min_length=1)
    stages: list[StageName] = Field(default_factory=lambda: ["script", "voices", "generate", "qa"])
    generation_mode: GenerationMode | None = None
    force: bool = False

    @field_validator("stages")
    @classmethod
    def unique_stages(cls, value: list[StageName]) -> list[StageName]:
        seen: set[str] = set()
        result: list[StageName] = []
        for item in value:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


class AutoMatchRequest(BaseModel):
    overwrite_unlocked: bool = False


class ModelListRequest(BaseModel):
    api_base_url: str | None = None
    api_key: SecretStr | None = None


class VoiceUploadMeta(BaseModel):
    name: str
