"""Live Ollama generation backend and recorded request metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    temperature: float = 0.0
    seed: Optional[int] = None
    max_output_tokens: Optional[int] = 1024
    context_window: Optional[int] = 8192

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "context_window": self.context_window,
        }


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Mapping[str, int] = field(default_factory=dict)
    response_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordedCall:
    request_id: str
    system_prompt: str
    user_prompt: str
    config: GenerationConfig


class OllamaBackendError(RuntimeError):
    pass


class OllamaBackend:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.calls: List[RecordedCall] = []
        self._model_records: Dict[str, Dict[str, Any]] = {}

    def _request(
        self, path: str, payload: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaBackendError(
                "Ollama HTTP %s for %s: %s" % (exc.code, path, detail)
            )
        except (TimeoutError, URLError, OSError) as exc:
            raise OllamaBackendError(
                "cannot reach Ollama at %s%s: %s" % (self.base_url, path, exc)
            )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaBackendError("Ollama returned invalid JSON: %s" % exc.msg)
        if not isinstance(value, dict):
            raise OllamaBackendError("Ollama response must be a JSON object")
        if value.get("error"):
            raise OllamaBackendError("Ollama error: %s" % value["error"])
        return value

    def describe(self, model: str) -> Dict[str, Any]:
        if model in self._model_records:
            return dict(self._model_records[model])
        version = self._request("/api/version")
        tags = self._request("/api/tags")
        models = tags.get("models")
        if not isinstance(models, list):
            raise OllamaBackendError("Ollama /api/tags returned no models array")
        match = None
        for item in models:
            if isinstance(item, dict) and (
                item.get("name") == model or item.get("model") == model
            ):
                match = item
                break
        if match is None:
            raise OllamaBackendError("model %r is not installed in Ollama" % model)
        details = match.get("details") if isinstance(match.get("details"), dict) else {}
        record = {
            "provider": "ollama",
            "ollama_version": str(version.get("version") or "unknown"),
            "model": model,
            "model_digest": str(match.get("digest") or ""),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
        }
        self._model_records[model] = record
        return dict(record)

    def generate(
        self,
        *,
        request_id: str,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
        response_schema: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        snapshot = self.describe(config.model)
        options: Dict[str, Any] = {"temperature": config.temperature}
        if config.seed is not None:
            options["seed"] = config.seed
        if config.max_output_tokens is not None:
            options["num_predict"] = config.max_output_tokens
        if config.context_window is not None:
            options["num_ctx"] = config.context_window

        schema = dict(response_schema or {"type": "object"})
        payload = {
            "model": config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": schema,
            "options": options,
        }
        self.calls.append(
            RecordedCall(request_id, system_prompt, user_prompt, config)
        )
        raw = self._request("/api/chat", payload)
        message = raw.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaBackendError("Ollama response is missing message.content")
        text = message["content"]
        created_at = str(raw.get("created_at") or "")
        response_id = hashlib.sha256(
            (request_id + "\n" + created_at + "\n" + text).encode("utf-8")
        ).hexdigest()[:24]
        usage = {
            "input_tokens": int(raw.get("prompt_eval_count") or 0),
            "output_tokens": int(raw.get("eval_count") or 0),
        }
        metadata = {
            **snapshot,
            "created_at": created_at,
            "done_reason": raw.get("done_reason"),
        }
        return LLMResponse(
            text=text,
            model=str(raw.get("model") or config.model),
            usage=usage,
            response_id=response_id,
            metadata=metadata,
        )