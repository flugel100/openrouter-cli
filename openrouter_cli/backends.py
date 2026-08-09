"""Pluggable backend abstraction for chat completions.

Two backends share one small interface:

- ``OpenRouterBackend``: speaks to the OpenRouter HTTP API (the default).
- ``LlmRouterBackend``: calls the local ``llm_router`` package directly
  (routing / fallback / budget handled by llm-router), optionally imported so
  ``openrouter-cli`` still runs without llm-router installed.

The ``Backend.run`` method normalizes the result to a single message dict on
the OpenAI convention (``{"role": "assistant", "content": ...}``) so the CLI
does not care which backend produced it.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional, Protocol

from .client import OpenRouterClient, OpenRouterError, extract_text

__all__ = ["Backend", "OpenRouterBackend", "LlmRouterBackend"]


class Backend(Protocol):
    """The minimal interface both backends implement."""

    @property
    def model(self) -> str: ...

    @property
    def backend_name(self) -> str: ...

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Return one assistant message dict for the given transcript."""


class OpenRouterBackend:
    """Default backend: thin wrapper over the HTTP client."""

    def __init__(self, client: OpenRouterClient, model: str):
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def backend_name(self) -> str:
        return "openrouter"

    def set_model(self, model: str) -> None:
        self._model = model

    def complete(self, messages, **kwargs):
        data = self._client.chat(self._model, messages, **kwargs)
        message = data["choices"][0]["message"]
        # Normalize content (extract_text handles blocks/refusal-like None).
        if message.get("content") is None and not message.get("tool_calls"):
            content = extract_text(data)
            message = dict(message)
            message["content"] = content
        return message

    def stream(self, messages, **kwargs) -> Iterator[str]:
        return self._client.chat_stream_content(self._model, messages, **kwargs)

    def available_models(self) -> list[str]:
        try:
            return [m.get("id", "") for m in self._client.list_models()]
        except OpenRouterError:
            return []


class LlmRouterBackend:
    """Optional backend that drives the local ``llm_router`` package.

    Construction raises ``OpenRouterError`` if llm-router is not installed or
    has no provider credentials available.
    """

    def __init__(
        self,
        model: str,
        *,
        task: str = "default",
        budget_usd: Optional[float] = None,
        provider: Optional[str] = None,
        require_all: bool = False,
    ):
        try:
            from llm_router import build_default_router, available_providers  # type: ignore
            from llm_router.policy import Policy  # type: ignore
        except ImportError:
            # Optional integration: allow pointing at a llm-router checkout via
            # the LLM_ROUTER_PATH environment variable.
            import sys

            candidate = os.environ.get("LLM_ROUTER_PATH", "")
            if candidate and candidate not in sys.path:
                sys.path.insert(0, candidate)
            try:
                from llm_router import build_default_router, available_providers  # type: ignore
                from llm_router.policy import Policy  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on env
                raise OpenRouterError(
                    "Integrasi llm-router: paket tidak ditemukan. Install "
                    "llm-router atau set variabel lingkungan LLM_ROUTER_PATH."
                ) from exc

        if not any(available_providers().values()):
            raise OpenRouterError(
                "llm-router terpasang tapi tidak ada kredensial provider ditemukan."
            )

        policy = Policy(task_tiers={task: "standard"}) if task else Policy()
        self._router = build_default_router(
            policy=policy,
            budget_usd=budget_usd,
            require_all=require_all,
        )
        self._task = task
        self._model = model
        self._provider = provider

    @property
    def model(self) -> str:
        return self._model or "auto"

    @property
    def backend_name(self) -> str:
        return "llm-router"

    def set_model(self, model: str) -> None:
        self._model = model

    def complete(self, messages, **kwargs):
        from llm_router import ChatRequest, Message  # type: ignore

        system = None
        msgs = []
        for m in messages:
            if m.get("role") == "system":
                system = (system or "") + str(m.get("content", ""))
            elif m.get("role") in ("user", "assistant", "tool"):
                msgs.append(Message(role=m["role"], content=str(m.get("content", ""))))

        resp = self._router.chat(
            ChatRequest(messages=msgs, system=system, task=self._task, provider=self._provider)
        )
        return {"role": "assistant", "content": resp.text}
