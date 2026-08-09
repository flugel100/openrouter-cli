"""Thin, typed client for the OpenRouter chat-completions API.

Covers:
  - chat completion (non-streaming)
  - streaming (SSE) completion
  - tool / function calling
  - model listing (for the interactive picker)

The client does not require any third-party HTTP library: it uses only the
Python standard library (urllib). This keeps the dependency surface to zero.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


DEFAULT_BASE_URL = "https://openrouter.ai"
DEFAULT_SITE_URL = "http://localhost"
DEFAULT_APP_NAME = "openrouter-cli"


class OpenRouterError(Exception):
    """Base error raised by the client."""


class OpenRouterHTTPError(OpenRouterError):
    """Raised when the API responds with a non-2xx status."""

    def __init__(self, status: int, message: str, data: Optional[dict[str, Any]] = None):
        super().__init__(f"OpenRouter HTTP {status}: {message}")
        self.status = status
        self.data = data


class OpenRouterAIContentError(OpenRouterError):
    """Raised when completion content is blocked by content moderation."""


@dataclass
class OpenRouterClient:
    """Minimal HTTP wrapper around the OpenRouter chat-completions API.

    Never logs or prints the API key.
    """

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    site_url: str = DEFAULT_SITE_URL
    app_name: str = DEFAULT_APP_NAME
    timeout: float = 60.0
    headers: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #
    def _request(
        self,
        path: str,
        method: str = "POST",
        payload: Optional[dict[str, Any]] = None,
        stream: bool = False,
    ) -> urllib.request.Response:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:  # pragma: no cover - depends on network
            try:
                body = json.loads(exc.read().decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                body = {}
            raise OpenRouterHTTPError(exc.code, exc.reason or str(exc), body) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - depends on network
            raise OpenRouterError(f"Network error: {exc.reason}") from exc

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    def list_models(self) -> list[dict[str, Any]]:
        """Return the raw list of models from GET /api/v1/models."""
        resp = self._request("/api/v1/models", method="GET")
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", data) if isinstance(data, dict) else data

    # ------------------------------------------------------------------ #
    # Chat completion (non-streaming)
    # ------------------------------------------------------------------ #
    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Run one completion and return the full API response dict."""
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        payload.update(extra)

        resp = self._request("/api/v1/chat/completions", payload=payload)
        return json.loads(resp.read().decode("utf-8"))

    def chat_content(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper returning just the assistant text content."""
        data = self.chat(model, messages, **kwargs)
        return extract_text(data)

    # ------------------------------------------------------------------ #
    # Streaming completion (SSE)
    # ------------------------------------------------------------------ #
    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield each SSE event dict for a streaming completion.

        The last yielded object has ``stream: "end"`` and a ``_request`` field
        holding the raw urllib response so callers can close it if needed.
        """
        stream_payload: dict[str, Any] = dict(kwargs)
        stream_payload["model"] = model
        stream_payload["messages"] = messages
        stream_payload["stream"] = True
        resp = self._request("/api/v1/chat/completions", payload=stream_payload)

        for line in resp:
            text = line.decode("utf-8", errors="replace").strip()
            if not text.startswith("data:"):
                continue
            data = text[len("data:"):].strip()

            if data != "[DONE]":
                event = json.loads(data)
            else:
                event = {"stream": "end"}
            if not hasattr(event, "_request"):
                event["_request"] = resp
            if event.get("_request") is None:
                event["_request"] = resp
            yield event

    def chat_stream_content(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield only the text deltas from a streaming completion."""
        for event in self.chat_stream(model, messages, **kwargs):
            if event.get("stream") == "end":
                break
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if text:
                yield text


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant message text out of a completion response."""
    choices = response.get("choices") or []
    if not choices:
        raise OpenRouterError("No choices in response.")
    message = choices[0].get("message") or {}

    if message.get("content") is None:
        # Tool calls or empty content; surface the reason if any.
        refusal = message.get("refusal")
        if refusal:
            raise OpenRouterAIContentError(str(refusal))
        raise OpenRouterError("Assistant produced no text (content is None).")

    content = message["content"]
    if not isinstance(content, str):
        # Some providers return structured content arrays.
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block and block["text"]:
                    parts.append(str(block["text"]))
        return "\n".join(parts)
    return content


def resolve_api_key() -> str:
    """Resolve the API key from (in order): env, then ~/.openrouter-key.

    Raises OpenRouterError when none is found.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()

    home = os.path.expanduser("~")
    key_file = os.path.join(home, ".openrouter-key")
    if os.path.isfile(key_file):
        with open(key_file, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        if content:
            return content

    raise OpenRouterError(
        "No API key found. Set OPENROUTER_API_KEY or write the key to ~/.openrouter-key"
    )


def build_default_tools() -> list[dict[str, Any]]:
    """Return a small set of example tools demonstrating function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current UTC time as an ISO-8601 string.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two numbers and return the sum.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]
