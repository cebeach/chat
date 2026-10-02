# Entering text

## Sending a message
Type and press Enter. The prompt is `>>> ` with a grey hint that disappears as soon
as you start typing. An empty line does nothing.

| Key | Effect |
|---|---|
| Enter | send |
| Shift+Enter or Alt+Enter | new line without sending (terminal support varies) |
| Ctrl-C | at the prompt, clear the line; during a reply, stop it |
| Ctrl-D | quit (the conversation is auto-saved) |
| Up / Down | input history |
| Tab | complete a command or conversation name |

## Multiline input
Type `"""` on a line by itself, then your text, then `"""` again:

```
>>> """
  ... entering multiline mode (type """ to finish)
... Summarise this:
... first line
... second line
... """
```

The same works for a system prompt: `/system """`. Pasting several lines also
works, because bracketed paste is enabled; the paste arrives as one message.

## History
Everything you type is kept in `~/.local/share/chat/history` (the last 1000 lines)
and is available with the arrow keys in later sessions.

## Tab-completion
- At the start of a line, `/` and Tab complete command names (`/?`, `/cat`, `/clear`,
  `/config`, `/conversations`, `/exit`, `/info`, `/load`, `/read`, `/recall`,
  `/retry`, `/save`, `/set`, `/stats`, `/system`).
- After `/load ` or `/cat `, Tab completes saved conversation names.

## Sending files
`/read <path> [<path> ...]` sends the contents of text files as your message. See
[commands](commands.md#read-path-).

## Stopping a reply
Ctrl-C while a reply is streaming stops it and returns to the prompt. The part
received so far is kept in the conversation, with ` [interrupted]` appended, and no
stats line is shown. Use `/retry` to ask again.
