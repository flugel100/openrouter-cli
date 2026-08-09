"""Tests for the OpenRouter client — no network required (mocked urllib)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from openrouter_cli.client import (
    OpenRouterClient,
    build_default_tools,
    extract_text,
    resolve_api_key,
    OpenRouterAIContentError,
    OpenRouterError,
    OpenRouterHTTPError,
)


def _fake_response(body, status=200):
    """Build a fake urllib.Response-like object returning JSON bytes."""
    resp = mock.MagicMock()
    if isinstance(body, (dict, list)):
        encoded = json.dumps(body).encode("utf-8")
    else:
        encoded = body
    if callable(encoded):
        encoded = encoded()
    resp.read.return_value = encoded
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _iterable_response(chunks):
    """Build a fake response that iterates over raw SSE bytes."""
    resp = mock.MagicMock()
    resp.__iter__.return_value = iter([c.encode("utf-8") if isinstance(c, str) else c for c in chunks])
    return resp


class ClientTestCase(unittest.TestCase):
    def setUp(self):
        self.client = OpenRouterClient(api_key="test-key", base_url="https://or.example")

    @mock.patch("openrouter_cli.client.urllib.request.urlopen")
    def test_chat_returns_json(self, urlopen_mock):
        urlopen_mock.return_value = _fake_response(
            {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
        )
        out = self.client.chat("model-x", [{"role": "user", "content": "hi"}])
        self.assertEqual(out["choices"][0]["message"]["content"], "hello")

        req = urlopen_mock.call_args.args[0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "model-x")
        self.assertEqual(body["messages"][0]["content"], "hi")

    @mock.patch("openrouter_cli.client.urllib.request.urlopen")
    def test_chat_passes_optional_params(self, urlopen_mock):
        urlopen_mock.return_value = _fake_response({"choices": [{"message": {"content": "x"}}]})
        self.client.chat(
            "m", [{"role": "user", "content": "hi"}],
            temperature=0.5, max_tokens=10, tools=[{"type": "function"}],
        )
        req = urlopen_mock.call_args.args[0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["max_tokens"], 10)
        self.assertEqual(body["tools"], [{"type": "function"}])

    @mock.patch("openrouter_cli.client.urllib.request.urlopen")
    def test_chat_stream_yields_events(self, urlopen_mock):
        urlopen_mock.return_value = _iterable_response(
            [
                'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
                'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
                "data: [DONE]\n",
            ]
        )
        collected = list(self.client.chat_stream("m", [{"role": "user", "content": "x"}]))
        texts = [e["choices"][0]["delta"]["content"] for e in collected if "choices" in e]
        self.assertEqual(texts, ["Hel", "lo"])
        self.assertTrue(any(e.get("stream") == "end" for e in collected))

    @mock.patch("openrouter_cli.client.urllib.request.urlopen")
    def test_chat_stream_sends_model_and_messages(self, urlopen_mock):
        urlopen_mock.return_value = _iterable_response(["data: [DONE]\n"])
        list(self.client.chat_stream("model-x", [{"role": "user", "content": "hi"}]))
        req = urlopen_mock.call_args.args[0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "model-x")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertTrue(body["stream"])

    @mock.patch("openrouter_cli.client.urllib.request.urlopen")
    def test_chat_stream_content_yields_text_deltas(self, urlopen_mock):
        urlopen_mock.return_value = _iterable_response(
            [
                'data: {"choices":[{"delta":{"content":"A"}}]}\n',
                'data: {"choices":[{"delta":{"content":"B"}}]}\n',
                "data: [DONE]\n",
            ]
        )
        deltas = list(self.client.chat_stream_content("m", [{"role": "user", "content": "x"}]))
        self.assertEqual(deltas, ["A", "B"])

    @mock.patch("openrouter_cli.client.urllib.request.urlopen")
    def test_http_error_raises_wrapped(self, urlopen_mock):
        err = mock.MagicMock()

        def _open(*args, **kwargs):
            raise err

        urlopen_mock.side_effect = _open
        exc = OpenRouterHTTPError(404, "Not Found", {})
        err.code = 404
        err.reason = "Not Found"
        err.read.return_value = b"{}"
        urlopen_mock.side_effect = None
        urlopen_mock.side_effect = lambda *a, **k: (_ for _ in ()).throw(exc)
        with self.assertRaises(OpenRouterHTTPError) as ctx:
            self.client.chat("m", [{"role": "user", "content": "x"}])
        self.assertEqual(ctx.exception.status, 404)

    def test_list_models(self):
        resp = _fake_response({"data": [{"id": "a/x"}, {"id": "b/y"}]})
        with mock.patch(
            "openrouter_cli.client.urllib.request.urlopen", return_value=resp
        ) as m:
            models = self.client.list_models()
        self.assertEqual([x["id"] for x in models], ["a/x", "b/y"])
        req = m.call_args.args[0]
        self.assertEqual(req.get_method(), "GET")

    def test_extract_text(self):
        self.assertEqual(
            extract_text({"choices": [{"message": {"content": "hi"}}]}), "hi"
        )

    def test_extract_text_structured_blocks(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
                    }
                }
            ]
        }
        self.assertEqual(extract_text(resp), "part1\npart2")

    def test_extract_text_refusal_raises(self):
        resp = {"choices": [{"message": {"content": None, "refusal": "blocked"}}]}
        with self.assertRaises(OpenRouterAIContentError):
            extract_text(resp)

    def test_tools_builder(self):
        tools = build_default_tools()
        names = {t["function"]["name"] for t in tools}
        self.assertEqual(names, {"get_current_time", "add_numbers"})


class ResolveApiKeyTestCase(unittest.TestCase):
    def test_env_var_wins(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            self.assertEqual(resolve_api_key(), "env-key")

    def test_falls_back_to_key_file(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as d:
                key_file = os.path.join(d, ".openrouter-key")
                with open(key_file, "w", encoding="utf-8") as fh:
                    fh.write("file-key\n")
                with mock.patch.object(
                    os.path, "expanduser", return_value=d
                ):
                    self.assertEqual(resolve_api_key(), "file-key")

    def test_missing_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as d:
                with mock.patch.object(
                    os.path, "expanduser", return_value=d
                ):
                    with self.assertRaises(OpenRouterError):
                        resolve_api_key()


if __name__ == "__main__":
    unittest.main()
