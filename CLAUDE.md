# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## File access

Claude has permission to access only the current working directory (this repository). Every other path requires explicit permission from the user.

The one exception is the llama.cpp source. It is readable at `llama.cpp/`, a symlink the user maintains to their local checkout, which matches the `llama-server` binary they run. Use it for llama-server flags, endpoints and defaults instead of relying on memory. That directory alone is granted by the `Read(llama.cpp/**)` rule in `.claude/settings.local.json`. Treat it as read-only. If the link is missing, or a read through it is refused, ask the user; do not look for the source anywhere else. The link is not committed (see `.gitignore`).

## Running

```bash
source venv/bin/activate
python chat.py              # start chat with whatever model llama-server is serving
python chat.py --url http://host:8001  # custom llama-server URL
```

`chat.py` requires a running llama-server (`llama-server --port 8001 -m <model>`). The test suite does not: the offline tests fake the server, and the `integration` tests start their own (see Testing).

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

Tests marked `integration` start a real llama-server (profiles in `tests/llama-server.toml`) and are skipped unless a models directory is configured. They are serial and always use port 8001, so a plain `pytest` fails on a configured machine while your own llama-server is running there; use `-m "not integration"` then. A plain run exercises the first profile only; `--llama-model all` (or a comma-separated list) runs every profile, one server after another. See `docs/testing.md`.

## Architecture

Air-gapped terminal chat app talking to a local llama.cpp server. Six source files, no package structure:

- **`chat.py`** — Entry point. Parses args, runs the REPL loop, dispatches slash commands via `handle_command()`.
- **`config.py`** — Loads `~/.config/chat/config.toml` (stdlib `tomllib`), merges with `DEFAULTS` dict.
- **`llama_client.py`** — `LlamaClient` wraps the llama-server native API (`/health`, `/props`, `/apply-template`, `/completion`). The model and its context length are read from `/props` before every turn (`refresh()`), never chosen by the client. `LlamaChatStream` is an iterable that yields tokens (SSE) and exposes `.stats` after iteration. Thinking tags are detected from the server per model (`find_think_tags()`, `LlamaClient.detect_think_tags()`); read `docs/thinking-tags.md` before changing that.
- **`conversation.py`** — `Conversation` holds message history with timestamps. Handles save/load to JSON files in `~/.local/share/chat/conversations/`, pair recall, and system prompt.
- **`ui.py`** — All terminal I/O via Rich. Streaming display writes raw tokens with word-wrap, then erases and re-renders as Markdown. Readline integration for input history and tab-completion of commands and conversation names.
- **`conv2txt.py`** — Standalone CLI utility to convert saved conversation JSON to plain text.

### Documentation

`docs/user-guide.md` indexes the user docs. When a slash command, config key or CLI flag changes, update `docs/commands.md` / `docs/configuration.md` and the `README.md` feature list too. `docs/testing.md` is for developers (the real-server test fixtures), is not part of the user guide, and must be updated when the profile format, the local settings or the fixture behavior changes.

### Data flow

User input → `chat.py` REPL → `Conversation.add_user()` → `LlamaClient.chat()` returns `LlamaChatStream` → `ui.display_assistant_stream()` consumes iterator, shows raw tokens, re-renders as Markdown → `Conversation.add_assistant()`.

### Key conventions

- Zero external dependencies beyond `requests` and `rich`. New features should use stdlib only. `ruff` and `pytest` are development-only tools (`pytest` is in `requirements-dev.txt`).
- Config, conversations, and readline history all live under `~/.config/chat/` and `~/.local/share/chat/`.
- `state` dict in the REPL carries mutable session state (`model`, `config`, `show_stats`, `options`, `last_stats`).
- Model options (`seed`, `temperature`, `top_p`) use `None` to mean "use llama-server default"; `None` values are filtered out before sending to the API.
