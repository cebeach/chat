# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Running

```bash
source venv/bin/activate
python chat.py              # start chat with whatever model llama-server is serving
python chat.py --url http://host:8001  # custom llama-server URL
```

Requires a running llama-server (`llama-server --port 8001 -m <model>`).

## Linting

```bash
venv/bin/ruff check .
venv/bin/ruff format --check .
```

## Testing

```bash
pip install -r requirements-dev.txt   # pytest (needs >= 9)
venv/bin/python -m pytest
venv/bin/python -m pytest tests/test_load.py::TestLoadMergesRecordedThinkPairs   # one class
```

Tests are pytest style (plain `assert`, `tmp_path`, `subtests`); mocking stays on `unittest.mock`. Fixtures and fakes shared between files live in `tests/helpers.py`.

## Architecture

Air-gapped terminal chat app talking to a local llama.cpp server. Six source files, no package structure:

- **`chat.py`** — Entry point. Parses args, runs the REPL loop, dispatches slash commands via `handle_command()`.
- **`config.py`** — Loads `~/.config/chat/config.toml` (stdlib `tomllib`), merges with `DEFAULTS` dict.
- **`llama_client.py`** — `LlamaClient` wraps the llama-server native API (`/health`, `/props`, `/apply-template`, `/completion`). The model and its context length are read from `/props` before every turn (`refresh()`), never chosen by the client. `LlamaChatStream` is an iterable that yields tokens (SSE) and exposes `.stats` after iteration. Thinking tags are detected from the server per model (`find_think_tags()`, `LlamaClient.detect_think_tags()`); read `docs/thinking-tags.md` before changing that.
- **`conversation.py`** — `Conversation` holds message history with timestamps. Handles save/load to JSON files in `~/.local/share/chat/conversations/`, pair recall, and system prompt.
- **`ui.py`** — All terminal I/O via Rich. Streaming display writes raw tokens with word-wrap, then erases and re-renders as Markdown. Readline integration for input history and tab-completion of commands and conversation names.
- **`conv2txt.py`** — Standalone CLI utility to convert saved conversation JSON to plain text.

### Data flow

User input → `chat.py` REPL → `Conversation.add_user()` → `LlamaClient.chat()` returns `LlamaChatStream` → `ui.display_assistant_stream()` consumes iterator, shows raw tokens, re-renders as Markdown → `Conversation.add_assistant()`.

### Key conventions

- Zero external dependencies beyond `requests` and `rich`. New features should use stdlib only. `ruff` and `pytest` are development-only tools (`pytest` is in `requirements-dev.txt`).
- Config, conversations, and readline history all live under `~/.config/chat/` and `~/.local/share/chat/`.
- `state` dict in the REPL carries mutable session state (`model`, `config`, `show_stats`, `options`, `last_stats`).
- Model options (`seed`, `temperature`, `top_p`) use `None` to mean "use llama-server default"; `None` values are filtered out before sending to the API.
