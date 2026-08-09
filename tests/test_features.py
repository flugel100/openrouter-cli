"""Unit tests for new features: session export/import, backends, and REPL.

Everything is mocked — no network calls and no real llm-router dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from openrouter_cli import backends
from openrouter_cli import cli
from openrouter_cli import sessions as s
from openrouter_cli.client import OpenRouterError


class XDGIsolationMixin:
    """Point XDG_DATA_HOME at a temporary dir and restore it on teardown."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._xdg = os.environ.get("XDG_DATA_HOME")
        os.environ["XDG_DATA_HOME"] = self._tmp.name

    def tearDown(self):
        if self._xdg is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = self._xdg
        self._tmp.cleanup()


def _simple_session():
    return {
        "name": "demo",
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "hai"},
            {"role": "assistant", "content": "halo"},
        ],
    }


class ExportSessionTestCase(unittest.TestCase):
    def test_export_markdown_contains_title_and_headers(self):
        md = s.export_markdown(_simple_session())
        self.assertIn("# Session: demo", md)
        self.assertIn("### 🤖 Assistant", md)
        self.assertIn("hai", md)

    def test_export_json_is_valid_and_has_messages(self):
        raw = s.export_json(_simple_session())
        data = json.loads(raw)
        self.assertIn("messages", data)
        self.assertEqual(data["messages"][0]["content"], "hai")

    def test_export_session_md_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.md")
            fmt = s.export_session(_simple_session(), path, fmt="md")
            self.assertEqual(fmt, "md")
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("hai", content)

    def test_export_session_auto_chooses_json_for_dot_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.json")
            fmt = s.export_session(_simple_session(), path, fmt="auto")
            self.assertEqual(fmt, "json")
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("messages", data)


class ImportSessionTestCase(XDGIsolationMixin, unittest.TestCase):
    def test_roundtrip_export_then_import(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "roundtrip.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(s.export_json(_simple_session()))
            imported = s.import_session(path)
            self.assertEqual(imported["messages"], _simple_session()["messages"])

    def test_import_rejects_invalid_json_session(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"a": 1}, fh)
            with self.assertRaises(OpenRouterError):
                s.import_session(path)

    def test_import_rejects_non_list_messages(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad2.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"messages": "nope"}, fh)
            with self.assertRaises(OpenRouterError):
                s.import_session(path)

    def test_save_imported_session_returns_name(self):
        sess = dict(_simple_session())
        sess["name"] = "imported-demo"
        name = s.save_imported_session(sess)
        self.assertEqual(name, "imported-demo")
        loaded = s.load_session("imported-demo")
        self.assertEqual(loaded["messages"], _simple_session()["messages"])


class OpenRouterBackendTestCase(unittest.TestCase):
    def _backend(self):
        return backends.OpenRouterBackend(mock.Mock(), "openai/gpt-4o-mini")

    def test_complete_normalizes_none_content_via_extract_text(self):
        b = self._backend()
        b._client.chat.return_value = {
            "choices": [{"message": {"role": "assistant", "content": None}}]
        }
        with mock.patch.object(backends, "extract_text", return_value="milik-pengekstrak") as ex:
            out = b.complete([{"role": "user", "content": "hai"}])
            ex.assert_called_once()
        self.assertEqual(out["content"], "milik-pengekstrak")
        b._client.chat.assert_called_once_with(
            "openai/gpt-4o-mini", [{"role": "user", "content": "hai"}]
        )

    def test_set_model_changes_model_attr_and_property(self):
        b = self._backend()
        b.set_model("deepseek/deepseek-chat")
        self.assertEqual(b._model, "deepseek/deepseek-chat")
        self.assertEqual(b.model, "deepseek/deepseek-chat")


class LlmRouterBackendTestCase(unittest.TestCase):
    def tearDown(self):
        for key in [
            "llm_router",
            "llm_router.policy",
            "llm_router.model",
            "llm_router._request",
            "llm_router.schema",
        ]:
            sys.modules.pop(key, None)

    def _install_fake_llm_router(self):
        class _FakePolicy:
            def __init__(self, *a, **k):
                pass

        fake_router = mock.Mock()
        fake_router.chat.return_value = mock.Mock(text="dari llm-router")

        mod = types.ModuleType("llm_router")
        mod.Policy = _FakePolicy
        mod.ChatRequest = mock.Mock()
        mod.Message = mock.Mock()
        mod.build_default_router = mock.Mock(return_value=fake_router)
        mod.available_providers = mock.Mock(
            return_value={"anthropic": "key", "deepseek": "key"}
        )
        mod.Router = fake_router

        policy_mod = types.ModuleType("llm_router.policy")
        policy_mod.Policy = _FakePolicy
        sys.modules["llm_router"] = mod
        sys.modules["llm_router.policy"] = policy_mod
        return mod

    def test_init_raises_when_llm_router_missing(self):
        real_import = builtins_import = __import__
        side = "LLM_ROUTER_PATH"

        def fake_import(name, *a, **k):
            stripped = getattr(name, "__name__", name)
            if str(stripped).startswith("llm_router"):
                raise ImportError("no llm_router")
            return real_import(name, *a, **k)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_ROUTER_PATH", None)
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(OpenRouterError):
                    backends.LlmRouterBackend("openai/gpt-4o-mini")

    def test_complete_returns_assistant_message(self):
        mod = self._install_fake_llm_router()
        b = backends.LlmRouterBackend("openai/gpt-4o-mini")
        out = b.complete([{"role": "user", "content": "hai"}])
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["content"], "dari llm-router")
        mod.build_default_router.assert_called_once()


class ReplCommandTestCase(unittest.TestCase):
    def _backend_mock(self, model="openai/gpt-4o-mini"):
        b = mock.Mock()
        b.set_model = mock.Mock()
        b.backend_name = "openrouter"
        b.model = model
        return b

    def _fn(self):
        return lambda: None

    def test_model_command_sets_backend_and_session(self):
        messages = []
        session = {"model": None, "messages": []}
        b = self._backend_mock()
        ret = cli._handle_repl_command(
            "/model openai/gpt-4o-mini", b, messages, session, "demo", self._fn()
        )
        self.assertEqual(ret, "")
        b.set_model.assert_called_once_with("openai/gpt-4o-mini")
        self.assertEqual(session["model"], "openai/gpt-4o-mini")

    def test_clear_command_empties_messages(self):
        messages = [{"role": "user", "content": "x"}]
        b = self._backend_mock()
        ret = cli._handle_repl_command(
            "/clear", b, messages, {}, "demo", self._fn()
        )
        self.assertEqual(ret, "")
        self.assertEqual(messages, [])

    def test_quit_command_returns_exit(self):
        b = self._backend_mock()
        ret = cli._handle_repl_command(
            "/quit", b, [], {}, "demo", self._fn()
        )
        self.assertEqual(ret, "exit")

    def test_unknown_command_returns_empty(self):
        b = self._backend_mock()
        with mock.patch.object(cli, "print"):
            ret = cli._handle_repl_command(
                "/unknown", b, [], {}, "demo", self._fn()
            )
        self.assertEqual(ret, "")


class CmdSessionsExportTestCase(XDGIsolationMixin, unittest.TestCase):
    def test_cmd_sessions_exports_demo_session(self):
        s.save_session(_simple_session())
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "demo.md")
            args = argparse.Namespace(
                delete=None,
                export="demo",
                out_path=out_path,
                format="auto",
                import_file=None,
            )
            with mock.patch.object(cli, "print") as pr:
                code = cli.cmd_sessions(args)
            self.assertEqual(code, 0)
            written = "".join(
                str(c.args[0] if c.args else "") for c in pr.call_args_list
            )
            self.assertIn("demo", written)
            self.assertIn(out_path, written)
            self.assertTrue(os.path.isfile(out_path))


if __name__ == "__main__":
    unittest.main()
