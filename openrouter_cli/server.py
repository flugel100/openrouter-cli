"""Server HTTP OpenAI-compatible untuk openrouter-cli.

Jalankan: openrouter serve --port 9876
Endpoint: POST /v1/chat/completions
Format: kompatibel dengan OpenAI API (non-streaming + SSE streaming).
"""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from .backends import Backend, OpenRouterBackend


class _Handler(BaseHTTPRequestHandler):
    """Handler untuk endpoint /v1/chat/completions."""

    backend: Backend = None  # type: ignore[assignment]
    default_model: str = "openai/gpt-4o-mini"

    def log_message(self, fmt: str, *args: Any) -> None:
        """Log ke stderr (format ringkas)."""
        import sys
        print(f"[serve] {fmt % args}", file=sys.stderr, flush=True)

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": {"message": message, "code": status}}, status)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_error(404, "Not found. Only /v1/chat/completions is supported.")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._send_error(400, "Invalid JSON body.")
            return

        model = data.get("model", self.default_model)
        messages = data.get("messages", [])
        stream = data.get("stream", False)

        if not messages:
            self._send_error(400, "At least one message is required.")
            return

        # Gunakan model dari request jika ada
        b = self.backend
        if isinstance(b, OpenRouterBackend):
            b.set_model(model)

        if stream:
            self._handle_stream(model, messages)
        else:
            self._handle_completion(messages)

    def _handle_completion(self, messages: list[dict[str, Any]]) -> None:
        try:
            msg = self.backend.complete(messages)
        except Exception as exc:
            self._send_error(500, str(exc))
            return

        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )

        resp = {
            "id": "chatcmpl-openrouter-cli",
            "object": "chat.completion",
            "created": 0,
            "model": self.default_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": str(content)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send_json(resp)

    def _handle_stream(self, model: str, messages: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            if isinstance(self.backend, OpenRouterBackend):
                for delta in self.backend.stream(messages):
                    chunk = {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            else:
                # llm-router fallback: non-streaming dibungkus jadi satu chunk
                msg = self.backend.complete(messages)
                content = msg.get("content", "")
                chunk = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": str(content)}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except Exception as exc:
            error_chunk = {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
            return

        # Sinyal selesai
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def run_server(
    backend: Backend,
    host: str = "127.0.0.1",
    port: int = 9876,
    model: str = "openai/gpt-4o-mini",
) -> None:
    """Mulai server HTTP OpenAI-compatible."""
    _Handler.backend = backend
    _Handler.default_model = model
    server = HTTPServer((host, port), _Handler)
    print(f"\n🔌 openrouter-cli server — http://{host}:{port}")
    print(f"   Endpoint: POST /v1/chat/completions")
    print(f"   Format: OpenAI-compatible (response + SSE streaming)")
    print(f"   Tekan Ctrl+C untuk berhenti.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer berhenti.")
        server.server_close()
