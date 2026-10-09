# Troubleshooting

## "Cannot connect to llama-server"
The app could not reach the server's `/health` endpoint at startup and exited.
Start the server first, then the app:

```bash
llama-server --port 8001 -m <model>
python chat.py
```

If the server is elsewhere or on another port, use `--url http://host:port` or set
`llama_url` in [the config file](configuration.md).

## "Failed to read the server's properties (GET /props)"
The server answers but has not finished loading the model. Wait for it to report ready
and start the app again.

## "Lost connection to llama-server" / "llama-server error"
The server stopped or rejected the request during a reply. Your last message is removed
from the conversation, so you can send it again once the server is back.

## The model changed
Each turn the app asks the server which model it is serving. If you restarted the
server with a different model, one line (`model: old → new (context N tokens)`) is
printed. The conversation continues with the new model. Replies keep a record of which
model wrote them.

## Replies stop making sense in a long chat
The conversation may have outgrown the context window. See
[Statistics](statistics.md). Watch the `ctx` figure and the 80% warning.

## Thinking blocks are still in my saved files
`save_thinking = false` leaves out the `thinking` field, but a reply is only split into
`thinking` and `content` when tags were detected for that turn. `/config` shows them in the
`think_tags` row. If it says `none detected`, set `think_start` and `think_end` in the config
file. Files saved by earlier versions keep their reasoning inline in `content` and cannot be
split. See [thinking-tags.md](thinking-tags.md).

## "Not sent: the prompt would be N tokens but the context window is M"
The prompt (system prompt, the conversation so far and the new message) does not fit the
server's context window, so nothing was sent and the conversation is unchanged. Shorten the
message or the included file, free space with `/clear` (`/save` first), or restart
llama-server with a larger `-c`. See [statistics](statistics.md#the-pre-send-check). A file
over 8 MB is refused earlier with "File too large to read."

## "File not found."
Paths with spaces need quotes for
`/read`; inside `@@<...>` they need none. When an `@@<path>` or `/read` file cannot be
read, the whole message is not sent and the error names the file.

## `/system file.txt` set the prompt to the file name
A path is read as a file only if it exists and is inside the directory you started the
app from. Otherwise the text itself becomes the prompt. Start the app from the right
directory, or use a path below it.

## "Unknown command"
Check the name with `/?`. `/help` is shown in that table but is not implemented.

## Shift+Enter submits instead of adding a line
Not every terminal can send Shift+Enter distinctly. Try Alt+Enter, or use `"""`
multiline mode ([Input](input.md)).

## Where did my conversation go?
`ls ~/.local/share/chat/conversations/` (or your `conversations_dir`). Nothing is
saved while a conversation is empty, and `auto_save = false` disables the automatic
saves.
