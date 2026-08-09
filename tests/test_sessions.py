"""Tests for persistent chat sessions (isolated to a temp data dir)."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from openrouter_cli import sessions as s


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Isolate the data dir so tests never touch the real home stash.
        self._xdg = os.environ.get("XDG_DATA_HOME")
        os.environ["XDG_DATA_HOME"] = self._tmp.name

    def tearDown(self):
        if self._xdg is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = self._xdg
        self._tmp.cleanup()

    def test_load_returns_empty_skeleton(self):
        sess = s.load_session("brand-new")
        self.assertEqual(sess["name"], "brand-new")
        self.assertEqual(sess["messages"], [])
        self.assertIsNone(sess["model"])

    def test_save_then_load_roundtrip(self):
        s.save_session({"name": "foo", "model": "m1", "messages": [{"role": "user", "content": "hi"}]})
        loaded = s.load_session("foo")
        self.assertEqual(loaded["model"], "m1")
        self.assertEqual(len(loaded["messages"]), 1)
        self.assertEqual(loaded["messages"][0]["content"], "hi")

    def test_slug_sanitizes_name(self):
        s.save_session({"name": "My Session / #1", "model": None, "messages": []})
        self.assertTrue(s.list_sessions())
        self.assertTrue(s.delete_session("My Session / #1"))

    def test_list_returns_metadata(self):
        s.save_session({"name": "a", "model": "modelA", "messages": [{"role": "user", "content": "x"}]})
        info = s.list_sessions()
        self.assertEqual(len(info), 1)
        self.assertEqual(info[0]["name"], "a")
        self.assertEqual(info[0]["model"], "modelA")
        self.assertEqual(info[0]["messages"], 1)

    def test_delete_removes_file(self):
        s.save_session({"name": "delme", "model": None, "messages": []})
        self.assertTrue(s.delete_session("delme"))
        self.assertFalse(s.delete_session("delme"))
        self.assertEqual(s.list_sessions(), [])

    def test_corrupt_file_is_treated_as_new(self):
        data_dir = os.path.join(self._tmp.name, "openrouter-cli", "sessions")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, "corrupt.json"), "w", encoding="utf-8") as fh:
            fh.write("{ this is not json ]")
        sess = s.load_session("corrupt")
        self.assertEqual(sess["messages"], [])
        # list_sessions should skip the corrupt file gracefully.
        self.assertEqual(s.list_sessions(), [])

    @mock.patch("openrouter_cli.sessions.input", return_value="1")
    def test_pick_session_interactive(self, _input):
        s.save_session({"name": "pickme", "model": None, "messages": []})
        self.assertEqual(s.pick_session_interactive(), "pickme")

    @mock.patch("openrouter_cli.sessions.input", return_value="")
    def test_pick_session_new(self, _input):
        s.save_session({"name": "unused", "model": None, "messages": []})
        self.assertIsNone(s.pick_session_interactive())


if __name__ == "__main__":
    unittest.main()
