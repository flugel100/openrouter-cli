# openrouter-cli

A minimal, **dependency-free** (stdlib only) OpenRouter API client and
interactive CLI, written in Python.

## Features

- Chat completions (non-streaming + SSE streaming)
- Tool / function calling with local execution
- Interactive REPL
- Interactive model picker / catalogue listing
- Persistent multi-turn chat sessions

## Installation

```bash
pip install -e .
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

# pick a model at the start of the REPL
openrouter repl --pick
```

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
