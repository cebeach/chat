# Conversations

## System prompt
Set it for a session with `/system` (inline text, `"""` for several lines, or a path
to a file in the current directory; see [commands](commands.md#system)), or for every
session with `system_prompt` in [the config file](configuration.md). It is sent with
every request and saved with the conversation.

## Saving
- **Automatic.** With `auto_save` on (the default), the conversation is saved to
  `auto_<YYYYmmdd_HHMMSS>.json` (the time the app started) after every completed
  reply and when the app exits, whether by `/exit`, Ctrl-D or an error. Nothing is
  saved while the conversation is empty.
- **Manual.** `/save <name>` writes `<name>.json`. With no name it uses the current
  timestamp.

Files go to `conversations_dir` (default `~/.local/share/chat/conversations/`).

## Finding and reading
- `/conversations` lists saved conversations, newest first.
- `/cat <name>` prints one without changing the current conversation.
- `/load <name>` replaces the current conversation with it. Tab completes names after
  `/load ` and `/cat `.

After `/load`, auto-save still writes to this session's `auto_...` file. Use
`/save <name>` if you want the continued conversation under the loaded name.

## Editing the history
- `/clear` empties the conversation (the system prompt stays).
- `/retry` regenerates the last reply.
- `/recall <n>` copies exchange `n` to the end so it is fresh in the model's context
  window; useful in a long conversation (see [Statistics](statistics.md)).

## Thinking blocks
Models that "think" write reasoning before the answer. It is shown while you chat, and
by default it is also kept in saved files, in a `thinking` field separate from the answer's
`content`. Set `save_thinking = false`, or use `/config save_thinking off`, to leave the
`thinking` field out of saved files. The reasoning is never sent back to the model. Details:
[thinking-tags.md](thinking-tags.md).

## File format
A saved conversation is a JSON file:

```json
{
  "model": "name of the model that was serving",
  "source_file": "/path/read/with/system",
  "system_prompt": "...",
  "messages": [
    {"role": "user", "timestamp": "2026-10-02T09:15:00.123456", "content": "..."},
    {"role": "assistant", "timestamp": "...", "model": "...", "thinking": "...", "content": "..."}
  ]
}
```

`source_file` appears only when it applies. A user message from `/read` also has
`source_file`. An assistant message has `thinking` only when the reply had a thinking block
and `save_thinking` is on; `content` is then the answer alone, without the thinking or its tags.
A reply that was only thinking (cut off by the token limit) has an empty `content`; it and the
question before it are left out of the history sent to the model.

The reply is split when it is generated, using the thinking tags detected for that turn, so a
file loads the same whichever model is served later. If no tags were detected for a turn, the
reply is stored whole in `content` and has no `thinking`.

### Files saved by earlier versions
This format is a breaking change. A file saved with a top-level `think_pairs` list and the
reasoning inline in `content` still loads, but nothing interprets it: `think_pairs` is ignored
and the tags stay in `content`, so they are shown by `/cat` and sent back to the model as plain
text. Saving such a file again does not separate them. Discard these files or edit them by hand.

## Converting to plain text
`conv2txt.py` turns a saved file into readable text:

```bash
python conv2txt.py ~/.local/share/chat/conversations/notes.json
python conv2txt.py notes.json -o notes.txt   # write to a file
python conv2txt.py notes.json --no-header    # omit the model and system prompt header
python conv2txt.py notes.json -l 80          # wrap at 80 columns (default 110)
```
