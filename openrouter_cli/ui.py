"""Enhanced terminal UI with optional rich rendering and readline history.

Import ``ui`` and call its functions; fallback to plain stdout when ``rich``
is not installed.
"""

from __future__ import annotations

import atexit
import os
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False
    Console = None  # type: ignore

try:
    import readline

    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False
    readline = None  # type: ignore

# --------------------------------------------------------------------------- #
# Console
# --------------------------------------------------------------------------- #

_console: Any = Console(highlight=False) if HAS_RICH else None
_HISTORY_FILE: Optional[str] = None


def get_console() -> Any:
    return _console


# --------------------------------------------------------------------------- #
# Readline history
# --------------------------------------------------------------------------- #

def setup_history() -> None:
    """Enable up/down arrow command history across REPL sessions."""
    global _HISTORY_FILE
    if not HAS_READLINE or _HISTORY_FILE:
        return
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "openrouter-cli")
    os.makedirs(d, exist_ok=True)
    _HISTORY_FILE = os.path.join(d, "history")
    try:
        readline.read_history_file(_HISTORY_FILE)
    except OSError:
        pass
    readline.set_history_length(500)
    # Auto-save on exit
    atexit.register(_save_history)


def _save_history() -> None:
    if HAS_READLINE and _HISTORY_FILE:
        try:
            readline.write_history_file(_HISTORY_FILE)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

def show_status(
    model: str,
    backend_name: str,
    stream_mode: bool,
    session_name: Optional[str] = None,
    msg_count: int = 0,
) -> None:
    """Tampilkan banner status ringkas."""
    mode = "streaming" if stream_mode else "blok"
    lines = [
        f"Model  : {model}",
        f"Backend: {backend_name}",
        f"Mode   : {mode}",
    ]
    if session_name:
        lines.append(f"Sesi   : {session_name} ({msg_count} pesan)")
    text = "\n".join(lines)
    if HAS_RICH:
        _console.print(Panel(text, title="Status", border_style="dim blue"))
    else:
        print(f"\n{text}\n")


def show_help() -> None:
    """Tampilkan bantuan perintah slash."""
    lines = [
        "/model <id>      ganti model",
        "/models          daftar model (backend OpenRouter)",
        "/clear           hapus riwayat percakapan sesi ini",
        "/status          tampilkan model & sesi aktif",
        "/stream on|off   nyalakan/matikan streaming token",
        "/quit, exit      keluar (sesi otomatis tersimpan)",
    ]
    if HAS_RICH:
        _console.print(Panel("\n".join(lines), title="Bantuan", border_style="dim green"))
    else:
        print("\nPerintah:\n  " + "\n  ".join(lines))
    print()


def show_welcome(model: str, session_name: Optional[str] = None, stream_mode: bool = True) -> None:
    """Tampilkan banner selamat datang REPL."""
    mode = "streaming" if stream_mode else "blok"
    lines = [f"Model  : {model}", f"Mode   : {mode}"]
    if session_name:
        lines.append(f"Sesi   : {session_name}")
    body = "\n".join(lines)
    footer = "/help untuk bantuan  |  Ctrl+D untuk keluar"
    if HAS_RICH:
        _console.print(Panel(body, title="OpenRouter REPL", subtitle=footer, border_style="bold cyan"))
    else:
        print(f"\nOpenRouter REPL\n{body}\n{footer}\n-" * 40)


def print_response_block(model: str, content: str) -> None:
    """Tampilkan respons utuh (mode blok)."""
    if HAS_RICH:
        _console.print()
        _console.print(Panel(Markdown(content or "(kosong)"), title=model, border_style="yellow"))
        _console.print()
    else:
        print(f"\n{model} > {content}\n")


def print_response_stream_start(model: str) -> Any:
    """Mulai tampilan streaming. Kembalikan konteks ``Live`` jika ada, ``None`` selainnya.

    Panggil ``update`` dengan teks akumulatif selama streaming, lalu lepaskan.
    """
    if HAS_RICH:
        _console.print()
        md = Markdown("▌")
        live = Live(
            Panel(md, title=model, border_style="yellow"),
            refresh_per_second=10,
            console=_console,
            transient=False,
        )
        live.start()
        live._or_title = model  # simpan untuk update berikutnya
        return live
    else:
        print(f"\n{model} > ", end="", flush=True)
        return None


def print_response_stream_update(live: Any, full_text: str) -> None:
    """Perbarui tampilan streaming."""
    if HAS_RICH:
        if live is not None:
            # Simpan judul panel dari argumen yang kita kirim saat start.
            title = getattr(live, "_or_title", "Model")
            live.update(Panel(Markdown(full_text + "▌"), title=title, border_style="yellow"))
    else:
        pass


def print_response_stream_end(live: Any, full_text: str, cancelled: bool = False) -> None:
    """Akhiri tampilan streaming."""
    if HAS_RICH:
        if live is not None:
            title = getattr(live, "_or_title", "Model")
            if cancelled and full_text:
                full_text += "\n\n*[Dibatalkan]*"
            live.update(Panel(Markdown(full_text), title=title, border_style="yellow"))
            live.stop()
        _console.print()
    else:
        suffix = " [Dibatalkan]" if cancelled else ""
        print(suffix)


def user_prompt(session_name: Optional[str] = None) -> str:
    """Tampilkan prompt dan baca input."""
    prefix = f"[{session_name}] " if session_name else ""
    if HAS_RICH:
        try:
            return _console.input(f"[bold]{prefix}You > [/bold]")
        except (EOFError, KeyboardInterrupt):
            raise
    else:
        try:
            return input(f"{prefix}You > ")
        except (EOFError, KeyboardInterrupt):
            raise


def show_info(text: str) -> None:
    """Cetak info singkat (ganti model, dll.)."""
    if HAS_RICH:
        _console.print(f"[dim]→ {text}[/dim]")
    else:
        print(f"→ {text}")


def show_error(text: str) -> None:
    """Cetak pesan error."""
    if HAS_RICH:
        _console.print(f"[red]Galat: {text}[/red]")
    else:
        print(f"Galat: {text}", file=__import__("sys").stderr)
