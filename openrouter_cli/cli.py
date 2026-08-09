"""Command-line interface for the OpenRouter client.

Features:
  --model        single-shot chat completion
  --stream       stream a single completion
  --tools        enable tool calling (executes local tool functions)
  --models       list available models (fuzzy picker)
  --repl         interactive chat loop
  --key          override API key (falls back to env / ~/.openrouter-key)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Optional

from . import __version__
from . import ui
from .agents import UltracodeAgent, AgentReport, CoderAgent, run_coder
from .backends import Backend, LlmRouterBackend, OpenRouterBackend
from .client import (
    OpenRouterClient,
    OpenRouterError,
    build_default_tools,
    resolve_api_key,
)
from .server import run_server
from .sessions import (
    _slug,
    delete_session,
    export_session,
    import_session,
    list_sessions,
    load_session,
    pick_session_interactive,
    save_session,
    save_imported_session,
)
from .tools import ToolRegistry, build_default_registry

DEFAULT_MODEL = "openai/gpt-4o-mini"


# ---------------------------------------------------------------------- #
# Tool implementations (executed locally when the model requests a call)
# ---------------------------------------------------------------------- #
def _tool_get_current_time() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _tool_add_numbers(a: Any, b: Any) -> float:
    return float(a) + float(b)


TOOL_FUNCS: dict[str, object] = {
    "get_current_time": _tool_get_current_time,
    "add_numbers": _tool_add_numbers,
}


# ---------------------------------------------------------------------- #
# Model picker
# ---------------------------------------------------------------------- #
def pick_model(client: OpenRouterClient) -> str:
    """Let the user interactively choose a model from the catalogue."""
    try:
        models = client.list_models()
    except OpenRouterError as exc:
        print(f"Could not list models: {exc}", file=sys.stderr)
        return DEFAULT_MODEL

    if not models:
        print("No models returned by the API.", file=sys.stderr)
        return DEFAULT_MODEL

    # Quick fuzzy filter: only include ids containing any sort token.
    query = input("Search models (Enter to list all): ").strip().lower()
    shown = []
    for m in models:
        mid = m.get("id", "")
        if not query or query.split()[0] in mid.lower():
            shown.append(mid)
    shown = sorted(shown)

    if not shown:
        print("No models matched your query.", file=sys.stderr)
        return DEFAULT_MODEL

    for i, mid in enumerate(shown[:50]):
        print(f"{i + 1:>3}. {mid}")

    try:
        choice = input("Pick number (Enter for first): ").strip()
    except (EOFError, KeyboardInterrupt):
        return shown[0]

    if not choice:
        return shown[0]

    try:
        idx = int(choice) - 1
    except ValueError:
        print("Invalid input, using first model.", file=sys.stderr)
        return shown[0]

    if 0 <= idx < len(shown):
        return shown[idx]
    print("Index out of range, using first model.", file=sys.stderr)
    return shown[0]


# ---------------------------------------------------------------------- #
# Message handling with executable tools
# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------- #
def cmd_chat(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    model = args.model or DEFAULT_MODEL
    backend.set_model(model)
    if args.pick:
        # Only OpenRouter backend has a model catalogue.
        if isinstance(backend, OpenRouterBackend):
            backend.set_model(pick_model(backend._client))
    messages = [{"role": "user", "content": args.prompt}]
    try:
        if args.tools:
            if not isinstance(backend, OpenRouterBackend):
                print("--tools hanya didukung untuk backend OpenRouter (tanpa --router).", file=sys.stderr)
                return 1
            tools = build_default_tools()
            _step_with_backend_tools(backend, messages, tools)
            last = messages[-1]
            print(_blocks_to_text(last.get("content")))
        else:
            message = backend.complete(messages)
            print(_blocks_to_text(message.get("content")))
    except OpenRouterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    model = args.model or DEFAULT_MODEL
    backend.set_model(model)
    if args.pick:
        if isinstance(backend, OpenRouterBackend):
            backend.set_model(pick_model(backend._client))
    messages = [{"role": "user", "content": args.prompt}]
    try:
        if isinstance(backend, OpenRouterBackend):
            for delta in backend.stream(messages):
                print(delta, end="", flush=True)
        else:
            message = backend.complete(messages)
            print(_blocks_to_text(message.get("content")))
        print()
    except OpenRouterError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    client = _build_client(args)
    try:
        models = client.list_models()
    except OpenRouterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for m in models:
        print(m.get("id", "<no-id>"), m.get("name") or "")
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    """List/delete/export/import saved sessions."""
    if args.export:
        sess = load_session(args.export)
        fmt = args.format
        out = args.out_path
        # Default output path: <name>.<ext> if --out looks like a directory.
        if os.path.isdir(out):
            ext = "json" if fmt == "json" else "md"
            out = os.path.join(out, f"{_slug(args.export)}.{ext}")
        else:
            fmt = "auto" if fmt == "auto" else fmt
        used_fmt = export_session(sess, out, fmt)
        print(f"Berhasil mengekspor sesi '{sess.get('name')}' ke {out} ({used_fmt}).")
        return 0

    if args.import_file:
        try:
            imported = import_session(args.import_file)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Could not import {args.import_file}: {exc}", file=sys.stderr)
            return 1
        except OpenRouterError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        name = save_imported_session(imported)
        print(f"Imported '{name}' from {args.import_file} (name: {imported.get('name')}).")
        return 0

    if args.delete:
        removed = delete_session(args.delete)
        if removed:
            print(f"Deleted session '{args.delete}'.")
        else:
            print(f"No session named '{args.delete}'.")
            return 1
        return 0

    sessions = list_sessions()
    if not sessions:
        print("No saved sessions.")
        return 0
    print(f"{'NAME':<20} {'MODEL':<30} {'MSGS':<5} UPDATED")
    for s in sessions:
        print(
            f"{s['name']:<20} {(s['model'] or '-')[:29]:<30} {s['messages']:<5} {s.get('updated','')}"
        )
    return 0


def _save_session_quiet(session: dict[str, Any]) -> bool:
    """Simpan sesi tanpa error. Kembalikan True kalau berhasil."""
    try:
        save_session(session)
    except (OSError, TypeError, ValueError):
        return False
    return True


def cmd_repl(args: argparse.Namespace) -> int:
    ui.setup_history()
    backend = _build_backend(args)
    model = args.model or DEFAULT_MODEL
    backend.set_model(model)
    if args.pick and isinstance(backend, OpenRouterBackend):
        backend.set_model(pick_model(backend._client))
        model = backend.model

    tools = build_default_tools() if args.tools else None

    # --- session handling ------------------------------------------------ #
    session = None
    session_name: Optional[str] = None
    if args.session:
        session_name = args.session
    elif args.continue_session:
        session_name = pick_session_interactive()
        if session_name is None:
            ui.show_info("Memulai sesi baru.")
    if session_name:
        session = load_session(session_name)
        num_turns = len(session["messages"])
        if num_turns:
            ui.show_info(f"Melanjutkan sesi '{session_name}' ({num_turns} pesan).")
            if session.get("model"):
                model = session["model"]
                backend.set_model(model)
        else:
            ui.show_info(f"Memulai sesi baru '{session_name}'.")
        session["model"] = model

    messages: list[dict[str, Any]] = session["messages"] if session else []
    use_stream = not getattr(args, "no_stream", False)

    ui.show_welcome(backend.model, session_name, use_stream)
    if tools:
        ui.show_info("Tools aktif: " + ", ".join(sorted(TOOL_FUNCS)))

    while True:
        try:
            user = ui.user_prompt(session_name)
        except (EOFError, KeyboardInterrupt):
            ui.show_info("Sampai jumpa!")
            if session_name is not None:
                _save_session_quiet(session)
            return 0

        cmd = user.strip()
        low = cmd.lower()

        # --- perintah keluar ---
        if low in {"exit", "quit"}:
            if session_name is not None:
                ok = _save_session_quiet(session)
                ui.show_info("Sesi tersimpan." if ok else "Gagal menyimpan sesi.")
            return 0

        # --- perintah slash ---
        if cmd.startswith("/"):
            handled = _handle_repl_command(
                cmd, backend, messages, session, session_name, use_stream
            )
            if isinstance(handled, dict):
                use_stream = handled.get("use_stream", use_stream)
                update = handled.get("update")
                if update:
                    ui.show_info(update)
            if handled == "exit":
                return 0
            continue

        if not user.strip():
            continue

        messages.append({"role": "user", "content": user})
        try:
            if tools and isinstance(backend, OpenRouterBackend):
                message = _step_with_backend_tools(backend, messages, tools)
                content = message.get("content")
                ui.print_response_block(backend.model, _blocks_to_text(content))
            elif isinstance(backend, OpenRouterBackend) and use_stream:
                live = ui.print_response_stream_start(backend.model)
                chunks: list[str] = []
                cancelled = False
                try:
                    for delta in backend.stream(messages):
                        chunks.append(delta)
                        ui.print_response_stream_update(live, "".join(chunks))
                except KeyboardInterrupt:
                    cancelled = True
                full = "".join(chunks)
                ui.print_response_stream_end(live, full, cancelled)
                if full:
                    messages.append({"role": "assistant", "content": full})
            else:
                message = backend.complete(messages)
                content = message["content"]
                messages.append({"role": "assistant", "content": _blocks_to_text(content)})
                ui.print_response_block(backend.model, _blocks_to_text(content))
        except OpenRouterError as exc:
            ui.show_error(str(exc))


def _handle_repl_command(
    cmd: str,
    backend: Backend,
    messages: list[dict[str, Any]],
    session,
    session_name: Optional[str],
    use_stream: bool,
) -> str | dict[str, Any]:
    """Tangani perintah slash di REPL.

    Return ``"exit"`` untuk keluar, ``dict`` untuk pembaruan status
    (mis. toggle streaming), atau ``str`` kosong.
    """
    parts = cmd.split()
    name = parts[0].lower()

    if name == "/help":
        ui.show_help()
        return ""

    if name == "/model":
        if len(parts) < 2:
            ui.show_info(f"Model aktif: {backend.model}")
            return ""
        backend.set_model(parts[1])
        if session is not None:
            session["model"] = parts[1]
        ui.show_info(f"Model diganti: {parts[1]}")
        return ""

    if name == "/models":
        if isinstance(backend, OpenRouterBackend):
            models = backend.available_models()
            if models:
                ui.show_info("Model tersedia:")
                for m in models[:100]:
                    print(f"  {m}")
            else:
                ui.show_info("Tidak bisa ambil daftar model (jaringan/API).")
        else:
            ui.show_info("Backend llm-router tidak menyediakan daftar model.")
        return ""

    if name == "/clear":
        messages.clear()
        if session_msg := (session or {}).get("messages"):
            session_msg.clear()
        ui.show_info("Riwayat percakapan dihapus.")
        return ""

    if name == "/status":
        ui.show_status(
            backend.model,
            backend.backend_name,
            use_stream,
            session_name,
            len(messages),
        )
        return ""

    if name == "/stream":
        sub = parts[1].lower() if len(parts) > 1 else "on"
        new_val = sub in {"on", "1", "true", "aktif"}
        return {"use_stream": new_val, "update": f"Mode {'streaming' if new_val else 'blok'} diaktifkan."}

    if name == "/agent":
        ui.show_info("Menjalankan agent ultracode...")
        task = " ".join(parts[1:]) if len(parts) > 1 else "audit kode di direktori ini"
        registry = build_default_registry(approval="auto")
        # Gunakan backend pertama yang tersedia
        if isinstance(backend, OpenRouterBackend):
            agent = UltracodeAgent(backend, registry, tier="lite", verbose=True)
            report = agent.run(task, scope=".")
            print(report.summary())
        else:
            ui.show_error("Agent hanya untuk backend OpenRouter.")
        return ""

    if name == "/coder":
        task = " ".join(parts[1:]) if len(parts) > 1 else "perbaiki kode ini"
        ui.show_info(f"Coder: {task}")
        if isinstance(backend, OpenRouterBackend):
            result = run_coder(backend, task, verbose=True)
            print(f"\n[CODER] Selesai dalam {result['turns']} turn.\n{result['output']}")
        else:
            ui.show_error("Coder hanya untuk backend OpenRouter.")
        return ""

    if name in {"/quit", "/exit"}:
        return "exit"

    ui.show_info(f"Perintah tidak dikenal: {name}. Ketik /help untuk bantuan.")
    return ""


def _step_with_backend_tools(
    backend: OpenRouterBackend,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Jalankan satu exchange tool-calls, mutasi messages, kembalikan message terakhir."""
    return _tools_loop(backend, messages, tools)


def _tools_loop(
    backend: OpenRouterBackend,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Jalankan satu putaran chat + tool-calls lewat OpenRouter backend."""
    message = backend.complete(messages, tools=tools)
    messages.append(message)

    if message.get("tool_calls"):
        for call in message["tool_calls"]:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                args = {}
            if name in TOOL_FUNCS:
                try:
                    value = TOOL_FUNCS[name](**args)
                    result = str(value)
                except Exception as exc:  # noqa: BLE001
                    result = f"Error: {exc}"
            else:
                result = f"Unknown tool: {name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result,
                }
            )
        final = backend.complete(messages, tools=tools)
        messages.append(final)
        return final
    return message


# ---------------------------------------------------------------------- #
# Agent commands
# ---------------------------------------------------------------------- #
def cmd_agent(args: argparse.Namespace) -> int:
    """Jalankan agent audit (ultracode) pada scope direktori/task."""
    backend = _build_backend(args)
    if not isinstance(backend, OpenRouterBackend):
        ui.show_error("Agent hanya didukung pada backend OpenRouter (tanpa --router).")
        return 1

    tools = build_default_registry(approval=args.approval or "auto")
    agent = UltracodeAgent(backend, tools, tier=args.tier or "medium", verbose=not args.quiet)
    ui.show_info(f"Agent ultracode (tier={agent.tier}, scope={args.scope})")
    ui.show_info(f"Tugas: {args.task}\n")

    report = agent.run(args.task, scope=args.scope)
    print(report.summary())

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report.summary() + "\n\n")
            for f in report.findings:
                fh.write(f"- [{f.severity}] {f.id}: {f.claim} ({f.location})\n")
        ui.show_info(f"Laporan tersimpan ke {args.output}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Mulai server HTTP OpenAI-compatible."""
    backend = _build_backend(args)
    model = args.model or DEFAULT_MODEL
    backend.set_model(model)
    try:
        run_server(backend, host=args.host, port=args.port, model=model)
    except OSError as exc:
        ui.show_error(f"Gagal mulai server: {exc}")
        return 1
    return 0


def cmd_coder(args: argparse.Namespace) -> int:
    """Jalankan coding agent otonom (seperti OpenCode / Claude Code)."""
    backend = _build_backend(args)
    if not isinstance(backend, OpenRouterBackend):
        ui.show_error("Coder hanya didukung pada backend OpenRouter (tanpa --router).")
        return 1

    tools = build_default_registry(approval="auto")
    agent = CoderAgent(backend, tools, max_turns=args.max_turns, verbose=not args.quiet)
    ui.show_info(f"Coder agent — tugas: {args.task}")

    result = agent.run(args.task, scope=args.scope)
    print(f"\n[CODER] Selesai dalam {result['turns']} turn.\n")
    print(result["output"])
    return 0


def _blocks_to_text(content: Any) -> str:
    """Convert OpenRouter's possible structured content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _build_client(args: argparse.Namespace) -> OpenRouterClient:
    key = args.key or resolve_api_key()
    return OpenRouterClient(
        api_key=key,
        site_url=args.referer or "http://localhost",
        app_name=args.app_name or "openrouter-cli",
        timeout=args.timeout,
    )


def _build_backend(args: argparse.Namespace) -> Backend:
    """Return the selected backend based on CLI flags.

    - default: OpenRouterBackend (HTTP via OpenRouterClient)
    - --router: LlmRouterBackend (drives the local llm-router package)
    """
    if getattr(args, "router", False):
        return LlmRouterBackend(
            model=args.model or DEFAULT_MODEL,
            task=getattr(args, "router_task", None) or "default",
            budget_usd=getattr(args, "router_budget", None),
            provider=getattr(args, "router_provider", None),
        )
    return OpenRouterBackend(_build_client(args), args.model or DEFAULT_MODEL)


# ---------------------------------------------------------------------- #
# Argument parsing
# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openrouter",
        description="Talk to OpenRouter models from the terminal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--key", help="OpenRouter API key (defaults to env/~/.");
    parser.add_argument("--referer", help="HTTP-Referer header (default http://localhost)");
    parser.add_argument("--app-name", default="openrouter-cli", help="X-Title header");
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout (s)");
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id to use");
    parser.add_argument("--pick", action="store_true", help="Choose a model interactively");
    parser.add_argument(
        "--router", action="store_true",
        help="Route through the local llm-router package (Claude/DeepSeek/OpenRouter)",
    );
    parser.add_argument("--router-task", help="Task name for llm-router policy");
    parser.add_argument("--router-budget", type=float, help="Budget (USD) for llm-router");
    parser.add_argument("--router-provider", help="Force provider for llm-router (anthropic/deepseek/openrouter)");

    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", help="Single chat completion")
    p_chat.add_argument("prompt", help="User message")
    p_chat.add_argument("--stream", action="store_true", help="Stream tokens");
    p_chat.add_argument("--tools", action="store_true", help="Enable tool calling");
    p_chat.set_defaults(func=_route_chat)

    p_stream = sub.add_parser("stream", help="Stream a single completion")
    p_stream.add_argument("prompt", help="User message")
    p_stream.set_defaults(func=cmd_stream)

    p_models = sub.add_parser("models", help="List models")
    p_models.set_defaults(func=cmd_models)

    p_repl = sub.add_parser("repl", help="Interactive chat loop")
    p_repl.add_argument("--tools", action="store_true", help="Enable tool calling");
    p_repl.add_argument("--no-stream", action="store_true", help="Matikan live streaming (mode blok)");
    p_repl.add_argument("--session", help="Resume/create a named session");
    p_repl.add_argument(
        "--continue", dest="continue_session", action="store_true",
        help="Pick a saved session to resume interactively",
    );
    p_repl.set_defaults(func=cmd_repl)

    p_sessions = sub.add_parser("sessions", help="List/export/import saved chat sessions")
    p_sessions.add_argument("--delete", metavar="NAME", help="Delete a session by name");
    p_sessions.add_argument("--export", metavar="NAME", help="Export a session to a file");
    p_sessions.add_argument("--out", dest="out_path", default=".", help="Output file path (used with --export)");
    p_sessions.add_argument("--format", choices=["auto", "md", "json"], default="auto", help="Export format");
    p_sessions.add_argument("--import", dest="import_file", metavar="FILE", help="Import a session JSON file");
    p_sessions.set_defaults(func=cmd_sessions)

    p_agent = sub.add_parser("agent", help="Jalankan agent audit kode (ultracode)")
    p_agent.add_argument("task", help="Deskripsi tugas audit")
    p_agent.add_argument("--scope", default=".", help="Direktori/file target (default: .)")
    p_agent.add_argument("--tier", choices=["lite", "medium", "deep"], default="medium", help="Kedalaman audit")
    p_agent.add_argument("--approval", choices=["auto", "ask", "deny"], default="auto", help="Persetujuan tool")
    p_agent.add_argument("--output", "-o", help="Simpan laporan ke file")
    p_agent.add_argument("--quiet", "-q", action="store_true", help="Sembunyikan log agent")
    p_agent.set_defaults(func=cmd_agent)

    p_coder = sub.add_parser("coder", help="Jalankan coding agent otonom (seperti OpenCode/Claude Code)")
    p_coder.add_argument("task", help="Deskripsi tugas coding")
    p_coder.add_argument("--scope", default=".", help="Direktori kerja")
    p_coder.add_argument("--max-turns", type=int, default=10, help="Maksimum turn agent")
    p_coder.add_argument("--quiet", "-q", action="store_true", help="Sembunyikan log")
    p_coder.set_defaults(func=cmd_coder)

    p_serve = sub.add_parser("serve", help="Mulai server HTTP OpenAI-compatible")
    p_serve.add_argument("--host", default="127.0.0.1", help="Alamat bind (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=9876, help="Port (default: 9876)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def _route_chat(args: argparse.Namespace) -> int:
    if args.stream:
        return cmd_stream(args)
    return cmd_chat(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
