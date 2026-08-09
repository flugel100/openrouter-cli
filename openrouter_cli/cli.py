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
import sys
from typing import Any, Optional

from . import __version__
from .client import (
    OpenRouterClient,
    OpenRouterError,
    build_default_tools,
    resolve_api_key,
)
from .sessions import (
    delete_session,
    list_sessions,
    load_session,
    pick_session_interactive,
    save_session,
)

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
def step_with_tools(
    client: OpenRouterClient,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run one exchange, executing any tool calls the model requests.

    Mutates ``messages`` in place and returns the updated list.
    """
    response = client.chat(model, messages, tools=tools)
    message = response["choices"][0]["message"]
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
            tool_result: dict[str, Any]
            if name in TOOL_FUNCS:
                try:
                    value = TOOL_FUNCS[name](**args)
                    tool_result = {"content": str(value)}
                except Exception as exc:  # noqa: BLE001
                    tool_result = {"content": f"Error: {exc}"}
            else:
                tool_result = {"content": f"Unknown tool: {name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": tool_result["content"],
                }
            )
        # Ask the model to continue with the tool results.
        final = client.chat(model, messages, tools=tools)
        messages.append(final["choices"][0]["message"])
    return messages


def _print_text(text: str) -> None:
    print(text)


# ---------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------- #
def cmd_chat(args: argparse.Namespace) -> int:
    client = _build_client(args)
    model = args.model
    if args.pick:
        model = pick_model(client)
    messages = [{"role": "user", "content": args.prompt}]
    try:
        if args.tools:
            tools = build_default_tools()
            step_with_tools(client, model, messages, tools)
            # Print the final assistant text.
            last = messages[-1]
            content = last.get("content")
            if isinstance(content, str):
                print(content)
            else:
                print(_blocks_to_text(content))
        else:
            text = client.chat_content(model, messages)
            print(text)
    except OpenRouterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    client = _build_client(args)
    model = args.model or DEFAULT_MODEL
    if args.pick:
        model = pick_model(client)
    messages = [{"role": "user", "content": args.prompt}]
    try:
        for delta in client.chat_stream_content(model, messages):
            print(delta, end="", flush=True)
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
    """List/delete saved sessions."""
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


def cmd_repl(args: argparse.Namespace) -> int:
    client = _build_client(args)
    model = args.model or DEFAULT_MODEL
    if args.pick:
        model = pick_model(client)

    tools = build_default_tools() if args.tools else None

    # --- session handling ------------------------------------------------ #
    session = None
    session_name: Optional[str] = None
    if args.session:
        session_name = args.session
    elif args.continue_session:
        session_name = pick_session_interactive()
        if session_name is None:
            print("Starting a fresh session.")
    if session_name:
        session = load_session(session_name)
        num_turns = len(session["messages"])
        if num_turns:
            print(f"Resuming session '{session_name}' with {num_turns} messages.")
            model = session.get("model") or model
        else:
            print(f"Starting new session '{session_name}'.")
        session["model"] = model

    messages: list[dict[str, Any]] = session["messages"] if session else []
    print(f"OpenRouter REPL — model: {model}  (Ctrl+D/Ctrl+C to quit)")
    if session_name:
        print(f"Session: {session_name}")
    if tools:
        print("Tools enabled: " + ", ".join(sorted(TOOL_FUNCS)))
    print("-" * 50)

    while True:
        try:
            user = input("You > ")
        except (EOFError, KeyboardInterrupt):
            print()
            if session_name is not None:
                try:
                    save_session(session)
                except OSError as exc:
                    print(f"Could not save session: {exc}", file=sys.stderr)
                else:
                    print(f"Session saved to '{session_name}'.")
            return 0
        if user.strip().lower() in {"exit", "quit"}:
            if session_name is not None:
                try:
                    save_session(session)
                except OSError as exc:
                    print(f"Could not save session: {exc}", file=sys.stderr)
                else:
                    print(f"Session saved to '{session_name}'.")
            return 0
        if not user.strip():
            continue
        messages.append({"role": "user", "content": user})

        try:
            if tools:
                messages = step_with_tools(client, model, messages, tools)
                final = messages[-1]
                content = final.get("content")
            else:
                response = client.chat(model, messages)
                messages.append(response["choices"][0]["message"])
                content = messages[-1].get("content")

            if isinstance(content, str):
                print(f"\n{model} > {content}\n")
            else:
                print(f"\n{model} > {_blocks_to_text(content)}\n")
        except OpenRouterError as exc:
            print(f"Error: {exc}", file=sys.stderr)


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
    p_repl.add_argument("--session", help="Resume/create a named session");
    p_repl.add_argument(
        "--continue", dest="continue_session", action="store_true",
        help="Pick a saved session to resume interactively",
    );
    p_repl.set_defaults(func=cmd_repl)

    p_sessions = sub.add_parser("sessions", help="List saved chat sessions")
    p_sessions.add_argument("--delete", metavar="NAME", help="Delete a session by name");
    p_sessions.set_defaults(func=cmd_sessions)

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
