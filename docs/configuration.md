# Configuration

Settings come from `~/.config/chat/config.toml`. The file is optional; a missing file
or key falls back to the default. Unknown keys are ignored. Order of precedence:

1. `--url` on the command line (only for the server URL)
2. `config.toml`
3. built-in default

Changes made inside the app with [`/set` and `/config`](commands.md) last for the
current session only.

## Keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `llama_url` | string | `http://127.0.0.1:8001` | Where llama-server listens. |
| `system_prompt` | string | `""` | System prompt for new sessions. |
| `conversations_dir` | string | `~/.local/share/chat/conversations` | Where `/save`, auto-save and `/load` look. |
| `auto_save` | bool | `true` | Save after each reply and on exit. |
| `save_thinking` | bool | `true` | Keep the model's reasoning in saved files. |
| `think_start`, `think_end` | string | `""` | Override the detected thinking tags. Both must be set. |
| `seed` | int | unset | Sampling seed. Unset means the server's default (random). |
| `temperature` | float | unset | Unset means the server's default (0.8). |
| `top_p` | float | unset | Unset means the server's default (0.95). |
| `read_file_max_kb` | int | `32` | Largest file `/read` and `/system <file>` accept. |

TOML has no "unset": leave a key out to use the server's default.

The model is never a setting. The app uses whichever model llama-server is serving,
and its context length is read from the server. If you restart the server with
another model during a session, the app notices before the next reply and prints one
line saying so.

## Example

```toml
llama_url = "http://127.0.0.1:8001"
system_prompt = "You are a concise assistant."
auto_save = true
save_thinking = false
temperature = 0.3
read_file_max_kb = 64
```

## Command line

```bash
python chat.py                          # use config.toml or defaults
python chat.py --url http://host:8001   # another llama-server
```

## Files the app uses

| Path | Content |
|---|---|
| `~/.config/chat/config.toml` | settings (you create it) |
| `~/.local/share/chat/conversations/` | saved conversations (`*.json`) |
| `~/.local/share/chat/history` | input history, last 1000 lines |
