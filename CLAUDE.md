# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
source venv/bin/activate
python chat.py              # start chat (auto-selects first model)
python chat.py -m llama3    # specify model
python chat.py --url http://host:11434  # custom Ollama URL
```

Requires a running Ollama instance (`ollama serve`).

## Linting

```bash
venv/bin/ruff check .
venv/bin/ruff format --check .
```

## Architecture

Air-gapped terminal chat app talking to a local Ollama server. Five source files, no package structure:

- **`chat.py`** — Entry point. Parses args, runs the REPL loop, dispatches slash commands via `handle_command()`.
- **`config.py`** — Loads `~/.config/chat/config.toml` (stdlib `tomllib`), merges with `DEFAULTS` dict.
- **`ollama_client.py`** — `OllamaClient` wraps the Ollama HTTP API. `ChatStream` is an iterable that yields tokens and exposes `.stats` after iteration.
- **`conversation.py`** — `Conversation` holds message history with timestamps. Handles save/load to JSON files in `~/.local/share/chat/conversations/`, pair recall, and system prompt.
- **`ui.py`** — All terminal I/O via Rich. Streaming display writes raw tokens with word-wrap, then erases and re-renders as Markdown. Readline integration for input history and tab-completion of commands and conversation names.
- **`conv2txt.py`** — Standalone CLI utility to convert saved conversation JSON to plain text.

### Data flow

User input → `chat.py` REPL → `Conversation.add_user()` → `OllamaClient.chat()` returns `ChatStream` → `ui.display_assistant_stream()` consumes iterator, shows raw tokens, re-renders as Markdown → `Conversation.add_assistant()`.

### Key conventions

- Zero external dependencies beyond `requests` and `rich`. New features should use stdlib only.
- Config, conversations, and readline history all live under `~/.config/chat/` and `~/.local/share/chat/`.
- `state` dict in the REPL carries mutable session state (`model`, `config`, `show_stats`, `options`, `last_stats`).
- Model options (`seed`, `temperature`, `top_p`) use `None` to mean "use Ollama default"; `None` values are filtered out before sending to the API.
