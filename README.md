# openrouter-cli

A minimal, **dependency-free** (stdlib only) OpenRouter API client and
interactive CLI, written in Python.

## Features

- Chat completions (non-streaming + SSE streaming)
- Tool / function calling with local execution
- Interactive REPL with slash commands (`/model`, `/models`, `/clear`, `/status`, `/help`, `/quit`)
- Interactive model picker / catalogue listing + multi-model switch mid-session
- Persistent multi-turn chat sessions (stored as JSON under `~/.local/share/openrouter-cli/sessions`)
- Session export/import (`--export` to Markdown/JSON, `--import`)
- Optional llm-router integration (`--router`) for policy-based routing

## Installation

```bash
pip install -e .
```

Optional llm-router support (needed for `--router`):

```bash
pip install -e ".[router]"
```

## Configuration

The API key is resolved from (in order):

1. `--key` CLI flag
2. `OPENROUTER_API_KEY` env var
3. `~/.openrouter-key` file

## Usage

```bash
# single completion
openrouter chat "What is the capital of France?"

# streamed completion
openrouter stream "Write a haiku about coding"

# enable tool calling (model can call get_current_time / add_numbers)
openrouter chat --tools "What's 1 + 2? and what time is it?"

# list models
openrouter models

# choose a model interactively
openrouter chat --pick "hello"

# interactive chat
openrouter repl
openrouter repl --tools

# persistent sessions (stored as JSON under ~/.local/share/openrouter-cli/sessions)
openrouter repl --session research       # save/resume a named session
openrouter repl --continue               # pick a saved session interactively
openrouter sessions                      # list saved sessions
openrouter sessions --delete research    # delete a session

# export a session to Markdown or JSON
openrouter sessions --export demo --out demo.md --format md
openrouter sessions --export demo --out demo.json --format json

# import a session back from a JSON file
openrouter sessions --import demo.json

# pick a model at the start of the REPL
openrouter repl --pick
```

Inside the REPL you can switch models, list models, clear the conversation,
see the active status, or quit (the session is saved automatically):

```
/model <id>     switch the active model
/models         list available models (OpenRouter backend)
/clear          clear the current conversation history
/status         show the active model and session
/help           show available commands
/quit           exit (session is saved automatically)
```

### llm-router routing

Route requests through the local `llm-router` package instead of calling
OpenRouter directly:

```bash
# route a single completion through llm-router
openrouter --router chat "What is the capital of France?"

# start a REPL backed by llm-router
openrouter --router repl
openrouter --router repl --session research

# optional router control flags
openrouter --router --router-task coding --router-budget 0.05 chat "..."
openrouter --router --router-provider anthropic chat "..."   # force a provider
```

This requires the extra install `pip install -e ".[router]"`, and the
`llm-router` package can be located via the `LLM_ROUTER_PATH` env var.

## Test

```bash
python -m unittest tests.test_client -v
```

## Library

```python
from openrouter_cli.client import OpenRouterClient, resolve_api_key

client = OpenRouterClient(api_key=resolve_api_key())
print(client.chat_content("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}]))
```
