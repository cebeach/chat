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
| `save_thinking` | bool | `true` | Keep the model's reasoning (the `thinking` field) in saved files. |
| `think_start`, `think_end` | string | `""` | Override the detected thinking tags. Both must be set. |
| `seed` | int | unset | Sampling seed. |
| `temperature` | float | unset | Randomness of the reply. |
| `top_p` | float | unset | Nucleus sampling cutoff. |
| `min_p` | float | unset | Drop tokens much less likely than the best one. |
| `repeat_penalty` | float | unset | Discourage repeating recent tokens. |
| `n_predict` | int | unset | Longest reply, in tokens (1 or more). Overrides a server `--predict`; it cannot raise it to unlimited. |
| `context_check` | bool | `true` | Count every prompt with the server before sending and refuse one that cannot fit the context window; warn above 80%. `/config context_check on\|off` changes it for the session. Off means no counting, no refusal and no warning. See [statistics](statistics.md#the-pre-send-check). |
| `reserve_output_tokens` | int | `0` | Tokens kept free for the reply when deciding whether a prompt fits. It guarantees nothing: the reply length is unbounded unless you set `n_predict` (here, with `/set`, or `--predict` on the server), and a thinking model's is hard to predict. Try a few hundred to a few thousand for a thinking model. An `n_predict` you set is kept free too (the larger of the two); a limit from the server's `--predict` is not, because `/props` does not report it. |

TOML has no "unset": leave a key out and nothing is sent for it, so the server's own value
applies. A model option you do set starts every session as an override that `/set` shows
and can change or remove. Values are checked on start-up with the same rules as `/set`
(see [commands](commands.md#set)); an invalid one is reported by name and ignored.

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
reserve_output_tokens = 1000
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
