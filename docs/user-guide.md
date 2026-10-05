# AI Chat user guide

AI Chat is a terminal chat program for a local [llama.cpp](https://github.com/ggml-org/llama.cpp)
server. It works offline and talks only to the server you point it at.

## Quick start

1. Install the dependencies (once):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start llama-server with a model:
   ```bash
   llama-server --port 8001 -m <model>
   ```
3. Start the chat:
   ```bash
   python chat.py                          # server at 127.0.0.1:8001
   python chat.py --url http://host:8001   # another server
   ```
4. Type a message and press Enter. Type `/?` to list commands and `/exit` to quit.

The model is always the one the server is running; the app never chooses one. It must be
an instruction-tuned model; see [Choosing a model](models.md).

## A first session

```
>>> Explain what a context window is in two sentences.
Assistant: ...
  58 tokens | 41.0 tok/s | 31 prompt tokens | ctx 89 / 8,192 (1.1%)
>>> /save context-notes
Conversation saved: /home/you/.local/share/chat/conversations/context-notes.json
>>> /exit
```

The conversation was also auto-saved, so even without `/save` it can be found with
`/conversations`.

## Where to go next

| I want to... | Read |
|---|---|
| look up a command | [Command reference](commands.md) |
| type multi-line text, use history and Tab, send files | [Entering text](input.md) |
| set a system prompt, save, load or convert conversations | [Conversations](conversations.md) |
| learn why only instruction-tuned models work | [Choosing a model](models.md) |
| change the server URL, defaults, or what gets saved | [Configuration](configuration.md) |
| understand the stats line and the context warning | [Statistics](statistics.md) |
| fix a problem | [Troubleshooting](troubleshooting.md) |
| know how reasoning ("thinking") tags are handled | [Thinking tags](thinking-tags.md) |
