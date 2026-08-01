from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    api_base_url: str
    api_key: str
    model: str
    endpoint: str = "chat_completions"
    reasoning_effort: str = "low"
    timeout_seconds: int = 240


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def list_models(self) -> list[str]:
        data = self._request("GET", "/v1/models")
        models = data.get("data", []) if isinstance(data, dict) else []
        return sorted({str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")})

    def generate_json(self, system_prompt: str, user_prompt: str, schema_name: str, schema: dict[str, Any], max_output_tokens: int = 12000) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.config.api_key:
            raise LLMError("API key is required for LLM preprocessing")
        if not self.config.model:
            raise LLMError("Model ID is required for LLM preprocessing")
        if self.config.endpoint == "responses":
            return self._responses_json(system_prompt, user_prompt, schema_name, schema, max_output_tokens)
        return self._chat_json(system_prompt, user_prompt, schema_name, schema, max_output_tokens)

    def _chat_json(self, system_prompt: str, user_prompt: str, schema_name: str, schema: dict[str, Any], max_output_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {"model": self.config.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "reasoning_effort": self.config.reasoning_effort, "max_completion_tokens": max_output_tokens, "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}}
        try:
            data = self._request("POST", "/v1/chat/completions", payload)
        except LLMError as exc:
            message = str(exc)
            if "response_format" not in message and "json_schema" not in message and "max_completion_tokens" not in message:
                raise
            data = self._request("POST", "/v1/chat/completions", {"model": self.config.model, "messages": payload["messages"], "temperature": 0.1, "max_tokens": max_output_tokens})
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected chat completion response: {data}") from exc
        return self._parse_json(content), data.get("usage", {}) if isinstance(data, dict) else {}

    def _responses_json(self, system_prompt: str, user_prompt: str, schema_name: str, schema: dict[str, Any], max_output_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {"model": self.config.model, "input": [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}, {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}], "reasoning": {"effort": self.config.reasoning_effort}, "max_output_tokens": max_output_tokens, "text": {"format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema}}}
        data = self._request("POST", "/v1/responses", payload)
        content = data.get("output_text") if isinstance(data, dict) else None
        if not content and isinstance(data, dict):
            texts = []
            for output in data.get("output", []):
                if isinstance(output, dict):
                    for item in output.get("content", []):
                        if isinstance(item, dict) and item.get("text"):
                            texts.append(str(item["text"]))
            content = "".join(texts)
        if not content:
            raise LLMError(f"Unexpected responses API payload: {data}")
        return self._parse_json(content), data.get("usage", {}) if isinstance(data, dict) else {}

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        base = self.config.api_base_url.rstrip("/")
        url = base + path[3:] if base.endswith("/v1") and path.startswith("/v1/") else base + path
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM HTTP {exc.code}: {body[:1200]}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMError("LLM request timed out") from exc
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON envelope: {raw[:800]}") from exc
        if not isinstance(result, dict):
            raise LLMError("LLM returned an unsupported response shape")
        return result

    @staticmethod
    def _parse_json(content: Any) -> dict[str, Any]:
        if isinstance(content, list):
            content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        text = str(content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise LLMError(f"LLM did not return a JSON object: {text[:800]}")
            try:
                result = json.loads(text[start:end + 1])
            except json.JSONDecodeError as exc:
                raise LLMError(f"LLM returned malformed JSON: {text[:800]}") from exc
        if not isinstance(result, dict):
            raise LLMError("LLM result must be a JSON object")
        return result
