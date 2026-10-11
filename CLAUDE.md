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

Tests marked `integration` start a real llama-server (profiles in `tests/llama-server.toml`) and are skipped unless a models directory is configured. They are serial and always use port 8001, so a plain `pytest` fails on a configured machine while your own llama-server is running there; use `-m "not integration"` then. A plain run exercises the first profile only; `--llama-model all` (or a comma-separated list) runs every profile, one server after another. A test marked `llama_parallel(2)` (the `/duel` integration tests) runs only on profiles whose effective `--parallel` is at least 2 and is otherwise skipped with a reason; run them with `--llama-model qwen35-2B-Q8_0-parallel2`. See `docs/testing.md`.

`tests/test_prompt_pty.py` and `tests/test_repl_integration.py` run the real prompt, and the real `chat.py`, on a pseudo-terminal (`tests/pty_helpers.py`; stdlib only, no new dependency). `tests/test_turn_echo.py` pins which part of a turn `echo_suppressed()` wraps. Its `run_session()` runs the real `main()` over a mock client with scripted input (also used by `tests/test_project_session.py` for projects and `/remember`, and by `tests/test_duel_session.py` for `/duel` with `SlotScript` from `tests/helpers.py`, which hands out scripted replies per `id_slot`). `tests/test_duel_phase.py` runs `_run_phase` directly over scripted slots.

## Architecture

Air-gapped terminal chat app talking to a local llama.cpp server. Twelve source files, no package structure:

- **`chat.py`** — Entry point. Parses args, runs the REPL loop, dispatches slash commands via `handle_command()`.
- **`commands.py`** — The slash-command table: a frozen `Command` per command (name, aliases, help rows, `handler_name` string, completion data). Stdlib, imports neither `chat` nor `ui`. `ui.py` derives `COMMANDS`, `print_help` and tab-completion from it; `chat.handle_command` looks `globals()[handler_name]` up on each call (a handler returning `False` stops the REPL) and `commands.check_handlers(globals())` runs at the end of `chat.py`. Adding a command: a table entry plus a `_handle_<name>(args, client, conversation, state)` in `chat.py`.
- **`config.py`** — Loads `~/.config/chat/config.toml` (stdlib `tomllib`), merges with `DEFAULTS` dict.
- **`llama_client.py`** — `LlamaClient` wraps the llama-server native API (`/health`, `/props`, `/apply-template`, `/tokenize`, `/completion`). The model and its context length are read from `/props` before every turn (`refresh()`), never chosen by the client. `LlamaChatStream` is an iterable that yields tokens (SSE) and exposes `.stats` after iteration. Thinking tags are detected from the server per model (`find_think_tags()`, `LlamaClient.detect_think_tags()`); read `docs/thinking-tags.md` before changing that.
- **`conversation.py`** — `Conversation` holds message history with timestamps. Handles save/load to JSON files in `~/.local/share/chat/conversations/`, pair recall, and system prompt.
- **`ui.py`** — All terminal I/O via Rich. Streaming display writes raw tokens with word-wrap (it does not re-render as Markdown). The prompt is a plain `>>> ` read by readline (`get_user_input()`); `echo_suppressed()` keeps the tty from echoing keys typed during a model turn and must never be held across `input()` (readline entered with ECHO off draws nothing). Readline integration for input history and tab-completion of commands and conversation names.
- **`includes.py`** — `@@<path>` file includes: `expand_includes` parses a typed message, `build_message` reads the files (`_read_file`, 8 MB guard) and splices them into one flat text with the include metadata. Stdlib, pure logic; `chat.py` keeps `_prepare_user_message` (it prints the errors).
- **`tokens.py`** — Token accounting against the context window: `fits`, `near_limit`, the `/info` `breakdown`. Pure logic over `LlamaClient.count_tokens` / `prompt_tokens` / `check_fit` (`/tokenize` and `/apply-template`); `chat.py` runs the check before every send and `/system`.
- **`projects.py`** — Named projects (`docs/projects.md`): a directory with `project.md` (notes), `system.md` (system prompt), `project.toml` (allowed overrides) and `conversations/`. Stdlib, file I/O and pure logic only: name checks, atomic `write_notes`, `diff_text`, the `/remember` prompt and `clean_reply`. `chat.py` holds the switching (`_switch_project`, `_reload_project`, `_sync_notes`) and `/remember`; `Conversation.project_notes` rides in the system message and is never saved.
- **`duel.py`** — `/duel` (`docs/duel.md`): stdlib, pure logic like `projects.py`. The fixed phase sequence and who takes part in each phase, the issue `Ledger` (its transition table is the only way a status changes), prompt builders, lenient parsers, side options, the `/duel` argument parser and the atomic record file. `chat.py` runs it: `_run_phase` (threads only exist inside it, one per participant, handing results to the main thread through queues; a phase that ends abnormally is discarded whole) and `_handle_duel`. Sides are built with `duel.new_conversation`, never `chat.Conversation`.
- **`server_log.py`** — Load-time VRAM for `/info` (`docs/server-log.md`): stdlib, file I/O and pure logic like `projects.py`. `last_startup_block` (streams a log, keeps the lines between `listening on` markers), `matches` (strict fingerprint against `/props`), `vram_summary`, `find_log`. The server logs no request, so the log is found by fingerprint, never by a marker. `chat.py:load_vram` runs it once at startup and `State.vram` holds the result; the log is never read again.
- **`conv2txt.py`** — Standalone CLI utility to convert saved conversation JSON to plain text.

### Documentation

`docs/user-guide.md` indexes the user docs. When a slash command, config key or CLI flag changes, update `docs/commands.md` / `docs/configuration.md` and the `README.md` feature list too; project behavior is described in `docs/projects.md`, `/duel` in `docs/duel.md`. `docs/testing.md` is for developers (the real-server test fixtures), is not part of the user guide, and must be updated when the profile format, the local settings or the fixture behavior changes.

### Data flow

User input → `chat.py` REPL → `_sync_notes()` (picks up an edit of `project.md`) → token check (`LlamaClient.check_fit()`, `tokens.fits()`; refuses a prompt that cannot fit) → `Conversation.add_user()` → `LlamaClient.chat()` returns `LlamaChatStream` → `ui.display_assistant_stream()` consumes iterator, shows raw tokens → `Conversation.add_assistant()`.

### Key conventions

- Zero external dependencies beyond `requests` and `rich`. New features should use stdlib only. `ruff` and `pytest` are development-only tools (`pytest` is in `requirements-dev.txt`).
- A project switch (`_switch_project`) reads every file first, force-saves the old conversation (cancelling the switch if that fails), then applies the scope and resets `/set`, `/config` and `/stats` from `State.global_config`. `_conversations_dir(state)` is the one place that picks the conversations directory. `/remember` calls `LlamaClient.complete()` with its own `{"temperature": 0}` options, never `State.options`.
- Config, conversations, and readline history all live under `~/.config/chat/` and `~/.local/share/chat/`.
- The `State` dataclass (`chat.py`) carries mutable session state (`model`, `config`, `context_length`, `options`, `show_stats`, `think_tags`, …).
- Model options (`seed`, `temperature`, `top_p`, `min_p`, `repeat_penalty`, `n_predict`): `State.options` is a sparse dict of the user's overrides (from `config.toml` via `_initial_options()`, then `/set`). `/props` reports only the server's launch defaults, never what a request sent, so the overrides must be kept; an option not in the dict is not sent and the server's own value applies. Server values are never stored: `/set` and `/config` read them from `LlamaClient.sampling_defaults()` on every call. `/props` does not report `n_predict` (always -1).
