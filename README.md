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

## Features

- **Config file** — TOML configuration at `~/.config/chat/config.toml`
- **Save/load conversations** — JSON persistence with tab-completion of saved names
- **Input history** — Readline-based history persisted to disk
- **Markdown rendering** — Streamed responses re-rendered as Rich Markdown
- **Word-wrap streaming** — Streamed output wraps at word boundaries instead of breaking mid-word
- **Token/context stats** — Toggle display of tokens/sec and prompt token counts
- **Conversation info** — `/info` command showing message, word, character, and token counts
- **Tab-completion** — Slash commands and `/load` conversation names
- **Multiline input** — `"""` delimiters for multi-line prompts
