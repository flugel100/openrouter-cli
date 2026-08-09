"""Persistent multi-turn chat sessions stored as JSON files.

Sessions live under ``~/.local/share/openrouter-cli/sessions/`` (XDG data
dir) and are simply JSON lists of chat messages plus tiny metadata. This
keeps the feature dependency-free and trivially inspectable/editable.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

__all__ = [
    "export_markdown",
    "export_json",
    "export_session",
    "import_session",
    "save_imported_session",
    "load_session",
    "save_session",
    "list_sessions",
    "delete_session",
    "pick_session_interactive",
]


def sessions_dir() -> str:
    """Return the directory where session files are stored, creating it."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share"
    )
    path = os.path.join(base, "openrouter-cli", "sessions")
    os.makedirs(path, exist_ok=True)
    return path


def _slug(name: str) -> str:
    """Turn a session name into a safe filename slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return slug or "session"


def _path(name: str) -> str:
    return os.path.join(sessions_dir(), f"{_slug(name)}.json")


def load_session(name: str) -> dict[str, Any]:
    """Load a saved session, returning an empty skeleton if none exists.

    Returns a dict: ``{"name", "model", "messages", "created", "updated"}``.
    """
    path = _path(name)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("messages"), list):
                data["name"] = name
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "name": name,
        "model": None,
        "messages": [],
        "created": _now(),
        "updated": _now(),
    }


def save_session(session: dict[str, Any]) -> str:
    """Persist a session dict to disk, returning the file path written."""
    data = dict(session)
    data["updated"] = _now()
    name = data.get("name", "session")
    data["name"] = name
    # Strip out any objects that aren't JSON-serializable by default.
    data = json.loads(json.dumps(data, default=str))
    path = _path(name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


def list_sessions() -> list[dict[str, Any]]:
    """Return metadata for every stored session, most-recently-updated first."""
    directory = sessions_dir()
    out: list[dict[str, Any]] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                continue
            out.append(
                {
                    "name": data.get("name") or filename[:-5],
                    "model": data.get("model"),
                    "messages": len(data.get("messages", []) or []),
                    "updated": data.get("updated", ""),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda s: s.get("updated", ""), reverse=True)
    return out


def delete_session(name: str) -> bool:
    """Remove a session file, returning True if it existed and was removed."""
    path = _path(name)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pick_session_interactive() -> Optional[str]:
    """Show stored sessions and let the user pick one to resume (or new).

    Returns the chosen session name, or ``None`` if the user wants a fresh
    session. Falls back to ``None`` on empty input / non-interactive EOF.
    """
    sessions = list_sessions()
    if not sessions:
        print("No saved sessions yet.")
        return None

    print("Saved sessions:")
    for i, s in enumerate(sessions, 1):
        preview = s.get("model") or "no-model"
        print(f"{i:>3}. [{preview}] {s['name']} ({s['messages']} msgs, {s.get('updated', '')})")

    try:
        raw = input("Pick a session number (Enter for new): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    try:
        idx = int(raw) - 1
    except ValueError:
        return None
    if 0 <= idx < len(sessions):
        return sessions[idx]["name"]
    return None


# ---------------------------------------------------------------------- #
# Export / import
# ---------------------------------------------------------------------- #
def message_to_text(message: dict[str, Any]) -> str:
    """Render a single message's content (content may be str or block list)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def export_markdown(session: dict[str, Any]) -> str:
    """Render a session to readable Markdown."""
    name = session.get("name", "session")
    model = session.get("model") or "unknown"
    lines = [f"# Session: {name}", "", f"- **Model:** {model}", ""]
    for msg in session.get("messages", []):
        role = msg.get("role", "user")
        label = {"user": "🧑 You", "assistant": "🤖 Assistant", "system": "⚙️ System"}.get(
            role, role
        )
        # Tool messages show the tool name; tool_calls on assistant are noted.
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            label += f" (tool `{tool_call_id}`)"
        text = message_to_text(msg)
        lines.append(f"### {label}\n")
        lines.append(text)
        lines.append("")
        if msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                lines.append(f"- ⚙️ tool_call: **{fn.get('name', '')}** args=`{fn.get('arguments')}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_json(session: dict[str, Any]) -> str:
    """Render a session to pretty JSON."""
    return json.dumps(session, ensure_ascii=False, indent=2, default=str) + "\n"


def export_session(session: dict[str, Any], path: str, fmt: str = "auto") -> str:
    """Write a session transcript to ``path``.

    ``fmt`` is one of ``"md"``, ``"json"`` or ``"auto"`` (match on extension).
    Returns the format actually written.
    """
    if fmt == "auto":
        fmt = "json" if path.rstrip().lower().endswith(".json") else "md"
    payload = export_json(session) if fmt == "json" else export_markdown(session)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    return fmt


def import_session(path: str) -> dict[str, Any]:
    """Load a session from a JSON file (from ``export_session`` or ``load_session``).

    Returns the session dict. Raises OpenRouterError if the file is not a
    session file.
    """
    from .client import OpenRouterError

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise OpenRouterError(f"{path} is not a valid session JSON file.")
    data.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return data


def save_imported_session(session: dict[str, Any]) -> str:
    """Persist an imported session under the sessions dir, returning its name."""
    name = session.get("name") or "imported"
    sess = load_session(name)
    sess["messages"] = session["messages"]
    sess["model"] = session.get("model") or sess["model"]
    save_session(sess)
    return name
