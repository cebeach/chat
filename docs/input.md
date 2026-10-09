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
Write `@@<path>` anywhere in a message to put a text file's contents there:

```
The story is set in a harbor town in winter. @@<characters.txt> Now Mara confronts Eli
about the missing cargo.
```

The model receives one flat piece of text. It never sees the `@@<...>` marker or the
path, only the file's text with a blank line between it and your own words:

```
The story is set in a harbor town in winter.

<the contents of characters.txt>

Now Mara confronts Eli about the missing cargo.
```

- The path ends at the first `>`, so it can contain spaces: `@@<my notes.txt>`. `~` is
  expanded and relative paths are relative to where you started the app. A name that
  contains `>` or starts or ends with a space cannot be written this way; use `/read`.
- It works inside a `"""` block too, and as many files as you like.
- The file's leading and trailing newlines are dropped. Text right after the closing `>`
  (a comma, say) stays as you typed it, after the blank line. Newlines you typed around
  the marker are kept. A file that is empty or only whitespace adds nothing.
- Text inside an included file is never expanded, so a `@@<...>` in it stays as it is.
- If a file is missing, unreadable or over 8 MB, an error names it,
  the message is **not sent** and nothing is added to the conversation. Retype the message
  (the up arrow recalls a one-line message, not a `"""` block). A `@@<` with no closing `>` on
  its line is an error too; a message that ends up empty is not sent.
- `@@count`, `@property` and similar text without `@@<` are untouched. To write a literal
  `@@<`, put a backslash before it: `\@@<`.
- The files are listed under the message by `/cat` and in `/info`, and saved with it in
  the conversation file as `includes`. They are never sent to the model.

`/read <path> [<path> ...]` sends the contents of text files as your message, joined the
same way. See [commands](commands.md#read-path-).

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
