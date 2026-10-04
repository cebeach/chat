# Entering text

## Sending a message
Type and press Enter. The prompt is a plain `>>> `; type `/?` for the list of commands.
An empty line does nothing.

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
stats line is shown. Use `/retry` to ask again. Ctrl-C during the short wait before the
reply starts prints "Response interrupted." instead.

## Typing while a reply is on its way
Keys you type while the model is answering are not shown and are not sent: they are
discarded when the next prompt appears (so a stray Enter or a paste during a reply
never sends a message). This applies to model replies only; while a command such as
`/cat` prints, typed keys are still echoed. For the same reason Ctrl-C during a reply
does not print `^C`.

The terminal's echo is switched off for the length of a reply and back on before the
next prompt. If the app is killed by a signal during a reply (`kill`, closing the
window), the terminal can be left with echo off: type `stty sane` (or `reset`) and
press Enter to get it back.

## Other notes
- Input piped into the app (`echo hi | python chat.py`) is read line by line.
- Ctrl-Z suspends the app, and `fg` resumes it. If you suspended it at the prompt, the
  prompt and the text typed so far are drawn again on `fg`. The cursor goes to the end of
  that text, so if it was in the middle of the line, further edits are drawn a few columns
  off until you press Enter.
