"""Tool registry: fungsi yang bisa dipanggil model + sistem persetujuan pengguna.

Model dapat meminta eksekusi tool (function calling). Registry ini:
- Mendaftarkan tool dengan kategori keamanan
- Menyediakan spesifikasi OpenAI-compatible
- Mengeksekusi tool call
- Meminta persetujuan pengguna untuk tool berbahaya
"""

from __future__ import annotations

import datetime
import fnmatch
import glob as glob_module
import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """Satu fungsi yang bisa dipanggil model."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    func: Callable[..., str]
    category: str = "safe"  # safe, network, file, system
    dangerous: bool = False

    def to_openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> str:
        return self.func(**kwargs)


# --------------------------------------------------------------------------- #
# Built-in tool implementations
# --------------------------------------------------------------------------- #

def _tool_time() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _tool_add(a: Any, b: Any) -> str:
    return str(float(a) + float(b))


def _tool_web_fetch(url: str) -> str:
    """Ambil konten halaman web (maks 4 KB)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "openrouter-cli/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(8192)[:4096]
            return body.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error mengambil {url}: {exc}"


def _tool_read_file(path: str) -> str:
    """Baca isi file, terbatas pada direktori kerja."""
    p = os.path.abspath(os.path.expanduser(path))
    cwd = os.path.abspath(os.getcwd())
    # Lindungi dari traversal keluar cwd
    if not p.startswith(cwd):
        return f"Akses ditolak: {path} berada di luar direktori kerja."
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return fh.read(8192)  # batasi 8 KB
    except Exception as exc:
        return f"Error membaca {path}: {exc}"


def _tool_write_file(path: str, content: str) -> str:
    """Tulis string ke file, terbatas pada direktori kerja."""
    p = os.path.abspath(os.path.expanduser(path))
    cwd = os.path.abspath(os.getcwd())
    if not p.startswith(cwd):
        return f"Akses ditolak: {path} berada di luar direktori kerja."
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"Berhasil menulis {len(content)} karakter ke {path}."
    except Exception as exc:
        return f"Error menulis {path}: {exc}"


def _tool_run_command(command: str) -> str:
    """Jalankan perintah shell (maks 5 detik, output dibatasi 4 KB)."""
    try:
        r = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.getcwd(),
        )
        out = (r.stdout + r.stderr)[:4096]
        return out or f"(exit kode {r.returncode}, tidak ada output)"
    except subprocess.TimeoutExpired:
        return "Perintah timeout (5 detik)."
    except Exception as exc:
        return f"Error menjalankan '{command}': {exc}"


def _tool_edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Ganti string dalam file (search & replace presisi)."""
    p = os.path.abspath(os.path.expanduser(path))
    cwd = os.path.abspath(os.getcwd())
    if not p.startswith(cwd):
        return f"Akses ditolak: {path} di luar direktori kerja."
    try:
        with open(p, "r", encoding="utf-8") as fh:
            content = fh.read()
        if replace_all:
            count = content.count(old_string)
            content = content.replace(old_string, new_string)
        else:
            count = 1 if old_string in content else 0
            content = content.replace(old_string, new_string, 1)
        if count == 0:
            return f"Error: string tidak ditemukan di {path}"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"Berhasil: {count} penggantian di {path}."
    except Exception as exc:
        return f"Error edit {path}: {exc}"


def _tool_glob(pattern: str) -> str:
    """Cari file dengan pola glob."""
    try:
        matches = sorted(glob_module.glob(pattern, recursive=True))[:50]
        if not matches:
            return f"Tidak ada file yang cocok dengan '{pattern}'."
        return "\n".join(matches)
    except Exception as exc:
        return f"Error glob '{pattern}': {exc}"


def _tool_grep(pattern: str, include: str = "") -> str:
    """Cari teks dengan grep di file."""
    import re
    cwd = os.getcwd()
    results: list[str] = []
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"Error regex: {exc}"

    for root, dirs, files in os.walk(cwd):
        # Skip hidden & venv
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "__pycache__", ".venv", "venv", "dist", "build"}]
        for fname in sorted(files):
            if include and not fnmatch.fnmatch(fname, include):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if compiled.search(line):
                            rel = os.path.relpath(path, cwd)
                            results.append(f"{rel}:{lineno}: {line.rstrip()[:120]}")
            except (OSError, UnicodeDecodeError):
                continue
            if len(results) >= 30:
                break
        if len(results) >= 30:
            break
    return "\n".join(results) if results else f"Tidak ada yang cocok dengan '{pattern}'."


# --------------------------------------------------------------------------- #
# Default tools
# --------------------------------------------------------------------------- #

BUILTIN_TOOLS: list[Tool] = [
    Tool(
        "get_current_time", "Dapatkan waktu UTC saat ini sebagai ISO-8601.",
        {"type": "object", "properties": {}}, _tool_time,
    ),
    Tool(
        "add_numbers", "Tambah dua angka.",
        {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Angka pertama"},
                "b": {"type": "number", "description": "Angka kedua"},
            },
            "required": ["a", "b"],
        },
        _tool_add,
    ),
    Tool(
        "web_fetch", "Ambil teks dari URL (HTTP GET).",
        {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL lengkap"}},
            "required": ["url"],
        },
        _tool_web_fetch,
        category="network",
        dangerous=True,
    ),
    Tool(
        "read_file", "Baca isi file (dibatasi 8 KB). Hanya file di dalam direktori kerja.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path file relatif"}},
            "required": ["path"],
        },
        _tool_read_file,
        category="file",
        dangerous=False,
    ),
    Tool(
        "write_file", "Tulis string ke file. Hanya di dalam direktori kerja.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path file relatif"},
                "content": {"type": "string", "description": "Konten yang akan ditulis"},
            },
            "required": ["path", "content"],
        },
        _tool_write_file,
        category="file",
        dangerous=True,
    ),
    Tool(
        "run_command", "Jalankan perintah shell (timeout 5 detik, output 4 KB).",
        {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Perintah shell"}},
            "required": ["command"],
        },
        _tool_run_command,
        category="system",
        dangerous=True,
    ),
    Tool(
        "edit_file", "Ganti string persis di file (search & replace).",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path file relatif"},
                "old_string": {"type": "string", "description": "Teks yang akan diganti (harus persis)"},
                "new_string": {"type": "string", "description": "Teks pengganti"},
                "replace_all": {"type": "boolean", "description": "Ganti semua kemunculan (default false)"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        _tool_edit_file,
        category="file",
        dangerous=True,
    ),
    Tool(
        "glob_files", "Cari file berdasarkan pola glob (contoh: **/*.py, src/**/*.ts).",
        {
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "Pola glob"}},
            "required": ["pattern"],
        },
        _tool_glob,
        category="file",
        dangerous=False,
    ),
    Tool(
        "grep_search", "Cari teks di file dengan regex.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pola regex"},
                "include": {"type": "string", "description": "Filter file (contoh: *.py, *.go) (opsional)"},
            },
            "required": ["pattern"],
        },
        _tool_grep,
        category="file",
        dangerous=False,
    ),
]


# --------------------------------------------------------------------------- #
# Registry & approval
# --------------------------------------------------------------------------- #

class ApprovalPolicy:
    """Kebijakan persetujuan tool.

    - ``auto``: semua tool langsung dijalankan
    - ``ask``: tool ``dangerous`` minta persetujuan pengguna
    - ``deny``: tolak semua tool berbahaya
    """

    def __init__(self, mode: str = "ask"):
        self.mode = mode  # auto | ask | deny

    def needs_approval(self, tool: Tool) -> bool:
        if self.mode == "auto":
            return False
        if self.mode == "deny":
            return tool.dangerous
        return tool.dangerous

    def request_approval(self, tool: Tool, arguments: dict[str, Any]) -> bool:
        """Tampilkan prompt dan minta persetujuan pengguna. Return True jika disetujui."""
        try:
            args_str = json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(arguments)
        cat = f"[{tool.category}]" if tool.category != "safe" else ""
        msg = f"\n⚙️ Tool: {tool.name} {cat}\n   Args: {args_str}\n   Jalankan? [Y/n] "
        try:
            ans = input(msg).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in {"", "y", "yes", "ya"}


class ToolRegistry:
    """Kumpulan tool yang bisa dipanggil model."""

    def __init__(self, tools: Optional[list[Tool]] = None, approval: str = "ask"):
        self._tools: dict[str, Tool] = {}
        self.approval = ApprovalPolicy(approval)
        for t in (tools or BUILTIN_TOOLS):
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def to_openai_specs(self) -> list[dict[str, Any]]:
        return [t.to_openai_spec() for t in self._tools.values()]

    def execute_call(self, call: dict[str, Any]) -> str:
        """Jalankan satu tool_call dari respons model.
        
        ``call`` adalah elemen dari ``message["tool_calls"]`` (format OpenAI).
        """
        fn = call.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            args = {}

        tool = self._tools.get(name)
        if tool is None:
            return f"Tool tidak dikenal: {name}"

        if self.approval.needs_approval(tool):
            if not self.approval.request_approval(tool, args):
                return f"Tool '{name}' ditolak oleh pengguna."

        try:
            return tool.execute(**args)
        except Exception as exc:
            return f"Error tool '{name}': {exc}"


def build_default_registry(approval: str = "ask") -> ToolRegistry:
    """Buat registry dengan semua tool bawaan."""
    return ToolRegistry(BUILTIN_TOOLS, approval=approval)
