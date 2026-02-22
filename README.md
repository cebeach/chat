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
- **Save/load conversations** — JSON persistence with tab-completion of saved names; auto-saved on exit
- **View conversations** — `/cat <name>` prints a saved conversation; `/conversations` lists all saved conversations
- **Recall** — `/recall <n>` re-injects an older message pair into the active context window
- **Retry** — `/retry` regenerates the last response
- **Input history** — Readline-based history persisted to disk
- **Word-wrap streaming** — Streamed output wraps at word boundaries instead of breaking mid-word
- **Token/context stats** — Toggle display of tokens/sec and prompt token counts with `/stats`; warns when context window is nearly full
- **Model options** — `/set` to view or adjust seed, temperature, and top_p
- **Conversation info** — `/info` command showing message, word, character, and token counts
- **Tab-completion** — Slash commands and conversation names for `/load` and `/cat`
- **Multiline input** — `"""` delimiters for multi-line prompts; Shift+Enter or Alt+Enter inserts a newline without submitting; paste support via bracketed-paste mode
- **Multiline system prompts** — `/system """` opens the same multiline input mode for setting multi-paragraph system prompts

---
Built with [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
