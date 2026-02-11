# Chat

A Python chat application that provides a terminal-based interface for chatting with local LLMs via [Ollama](https://ollama.com) (`localhost:11434`).

## Architecture

- **`chat.py`** — Main REPL entry point with slash commands
- **`ollama_client.py`** — HTTP client for Ollama API (streaming chat, list models, health check)
- **`conversation.py`** — Message history and system prompt management
- **`ui.py`** — Rich-based terminal display with streaming output

## Dependencies

- `requests` — HTTP client for the Ollama API
- `rich` — Terminal formatting and streaming output

## Phase 2 Roadmap

1. Config file (`~/.config/chat/config.toml`)
2. Save/load conversations (JSON persistence)
3. Input history (readline)
4. Markdown rendering
5. Token/context stats
6. Multiline input
