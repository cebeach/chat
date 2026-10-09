# Command reference

Type a line starting with `/` to run a command. Commands are case-insensitive
(`/Save` works), are never sent to the model, and are tab-completed (see
[Input](input.md)). Anything else you type is sent to the model as a message.

`/?` prints the in-app command table. The table also lists `/help`, but `/help` is
not implemented and answers `Unknown command`; use `/?`.

| Command | What it does |
|---|---|
| `/?` | Show the command table |
| [`/cat <name>`](#cat-name) | Print a saved conversation |
| [`/clear`](#clear) | Empty the current conversation |
| [`/config`](#config) | Show settings; toggle `save_thinking` and `context_check` |
| [`/conversations`](#conversations) | List saved conversations |
| [`/exit`](#exit) | Save and quit |
| [`/info`](#info) | Conversation and context-window statistics |
| [`/load <name>`](#load-name) | Replace the current conversation with a saved one |
| [`/read <path> ...`](#read-path-) | Send one or more text files as a message |
| [`/recall <n>`](#recall-n) | Re-inject an earlier exchange |
| [`/retry`](#retry) | Regenerate the last reply |
| [`/save [name]`](#save-name) | Save the conversation |
| [`/set [key [value]]`](#set) | View or change model options |
| [`/stats`](#stats) | Toggle the per-reply stats line |
| [`/system [text]`](#system) | View or set the system prompt |
| `"""` | Start multiline input (see [Input](input.md)) |

## /?
Prints the command table.

## /cat <name>
Prints the saved conversation `<name>` (see [`/conversations`](#conversations))
without loading it. Thinking is left out: only each reply's answer is printed, and a
reply that was only thinking is skipped. To read the thinking of a saved conversation use
`conv2txt.py --keep-thinking` (see [Conversations](conversations.md)).
Errors: `No saved conversation named '<name>'.`

## /clear
Removes all messages from the current conversation. The system prompt stays.
The next auto-save is skipped until there is at least one message again.

## /config
`/config` shows the model, `system_prompt`, `llama_url`, `conversations_dir`,
`save_thinking`, the detected `think_tags`, and the model options. It re-reads the
server first, so the model name is current.

`/config save_thinking on|off` sets whether saved files keep the model's reasoning (the `thinking` field of each reply).
`/config save_thinking` with no argument toggles it. `/config context_check on|off` does the same for the
[pre-send token check](statistics.md#the-pre-send-check). These are the only settings
`/config` can change, and a change lasts for the current session only. Put
`save_thinking = false` or `context_check = false` in [the config file](configuration.md) to make it permanent.
See [thinking-tags.md](thinking-tags.md) for how tags are detected.

## /conversations
Lists saved conversations, newest first, with their file paths.

## /exit
Auto-saves (if `auto_save` is on) and quits. Ctrl-D does the same.

## /info
Shows message count (you and AI), words, characters, the context window size and, once
there is a message, the tokens the next prompt would carry, split into system prompt,
your messages, the AI's replies and template overhead, with the share of the window.
The counts come from the server each time, so `/info` needs it to be running. See
[Statistics](statistics.md#info).

## /load <name>
Replaces the current messages and system prompt with the saved conversation
`<name>`. The model recorded in the file is shown for information only. Your
session keeps talking to whatever model the server is running.

Note that auto-save keeps writing to this session's own `auto_<timestamp>` file,
not to the file you loaded. Use `/save <name>` to write changes back.

## /read <path> ...
Reads one or more UTF-8 text files and sends their contents to the model at once as
your next message. Quote paths that contain spaces: `/read "my notes.txt" other.txt`.
`~` is expanded and relative paths are relative to where you started the app.
Files are joined the way `@@<path>` includes are: a blank line between them, and each
file's leading and trailing newlines dropped. A file that is missing, unreadable, or
over 8 MB is reported by name and nothing is sent, and so is a message whose
prompt would not fit the context window (see [statistics](statistics.md#the-pre-send-check)). The
files are recorded in the saved conversation as `includes`, shown by `/cat` and `/info`.
To put a file in the middle of a longer message, use `@@<path>` instead; see
[input](input.md#sending-files).

## /recall <n>
Copies the n-th question-and-answer pair (counting from 1) to the end of the
conversation, with a note that it is recalled. Use it to bring an old exchange back
into the model's context window. `Pair n out of range (1-N)` if there is no such pair.

## /retry
Removes the last reply (including one you interrupted with Ctrl-C) and sends your
last message again. Needs a conversation that ends with an assistant reply, otherwise
`Nothing to retry` or `Last message is not an assistant response.` If the prompt would
not fit the context window, nothing is sent and the reply it would have replaced is kept
(see [statistics](statistics.md#the-pre-send-check)).

## /save [name]
Writes the conversation to `<conversations_dir>/<name>.json`. With no name the file
is named with the current timestamp (`YYYYmmdd_HHMMSS`). Characters other than
letters, digits, `-` and `_` in the name become `_`. Saving to an existing name
overwrites it.

## /set
- `/set` shows `seed`, `temperature` and `top_p`.
- `/set <key>` shows one option.
- `/set <key> <value>` changes it; `/set <key> default` returns it to the server's
  default.

`seed` is an integer; `temperature` and `top_p` are numbers. Changes last for the
session; use [the config file](configuration.md) for permanent values.

## /stats
Turns the dim stats line printed after each reply on or off. It starts on.

## /system
- `/system` shows the current system prompt.
- `/system <text>` sets it to the text.
- `/system """` opens multiline input for a long prompt.
- `/system <path>` reads the prompt from a file, but only if the file is inside the
  directory you started the app from (or below it). Any other value, including a path
  outside that directory, is used as the literal prompt text.

A new system prompt that, with the conversation so far, would not fit the context window
is refused and the old one stays (see [statistics](statistics.md#the-pre-send-check)).

The system prompt is saved with the conversation.
