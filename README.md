# Chat

A Python chat application that provides a terminal-based interface for chatting with local LLMs. Talks to a [llama.cpp server](https://github.com/ggml-org/llama.cpp) (`127.0.0.1:8001` by default).

## Architecture

- **`chat.py`** — Main REPL entry point with slash commands
- **`config.py`** — Loads `~/.config/chat/config.toml`, merges with defaults
- **`llama_client.py`** — HTTP client for the llama.cpp server native API (`/apply-template` + `/completion` for chat, `/props`, `/health`)
- **`conversation.py`** — Message history and system prompt management
- **`ui.py`** — Rich-based terminal display with streaming output
- **`conv2txt.py`** — Standalone utility to convert saved conversation JSON to plain text

## Dependencies

- `requests` — HTTP client for the llama.cpp server
- `rich` — Terminal formatting and streaming output
- `ruff` — linting and formatting (development only; listed in `requirements.txt`)
- `pytest` — test runner (development only; `pip install -r requirements-dev.txt`, then `python -m pytest`). Integration tests that start a real llama-server are described in [docs/testing.md](docs/testing.md)

## Features

- **Local llama.cpp server** — uses whichever model the server is serving; its context window is read from the server
- **Save/load conversations** — JSON files with tab-completed names; auto-saved after every reply and on exit
- **Thinking blocks** — kept in saved files by default, separate from the answer; `save_thinking = false` or `/config save_thinking [on|off]` omits them. Tags are detected from the server per model; details in [docs/thinking-tags.md](docs/thinking-tags.md)
- **Files** — `/read <path> ...` sends text files as a message; `/system <file>` loads a system prompt from a file in the current directory (both limited by `read_file_max_kb`, 32 KB by default)
- **Recall and retry** — `/recall <n>` re-injects an older exchange; `/retry` regenerates the last reply
- **Stats** — tokens/sec, prompt tokens and context-window usage after each reply (`/stats` toggles), an 80% context warning, and `/info`
- **Model options** — `/set` for `seed`, `temperature` and `top_p` (session only)
- **Input** — readline history (`~/.local/share/chat/history`), tab-completion of commands and conversation names, `"""` multiline input, Shift+Enter or Alt+Enter for a newline, bracketed paste
- **Plain-text export** — `conv2txt.py` converts a saved conversation to text

## Documentation

See the [user guide](docs/user-guide.md): [commands](docs/commands.md),
[entering text](docs/input.md), [conversations](docs/conversations.md),
[configuration](docs/configuration.md), [statistics](docs/statistics.md),
[troubleshooting](docs/troubleshooting.md).

---
Built with [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
