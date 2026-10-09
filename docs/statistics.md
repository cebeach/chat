# Statistics and the context window

A model can only consider a limited number of tokens at once: its *context window*.
The app reads the window size from llama-server (`/props`), so it is whatever the
server was started with (`llama-server -c ...`). Your whole conversation, including the
system prompt, is sent with each message, so it grows with every turn.

Two token figures appear, and they are different things:

- **Prompt tokens** are what a request carries: the system prompt, the conversation so
  far, the new message and the chat template's own markup. The model's reasoning is kept
  apart from a reply and is **not** sent back, so it is not part of this figure.
- **Generated tokens** are what the model produced for one reply, thinking and answer
  together. They use the window while the reply is being written, but they are shown only
  on the stats line and are never added to the prompt figure.

## The pre-send check
Before each message is sent, the app asks the server to render the exact prompt and count
it. The count is exact, not an estimate, and it is compared with the window the server
reports at that moment:

- A prompt that cannot fit is **refused**. Nothing is sent and the conversation is left
  as it was (a refused `/retry` keeps the reply it would have replaced):

  ```
  Not sent: the prompt would be 5,000 tokens but the context window is 4,096. Shorten the text or file, or free space with /clear.
  ```

- A prompt above 80% of the window is sent, with a yellow warning first:

  ```
  Warning: context window 85% full (3,482 / 4,096 tokens)
  ```

- When a file was included, the dim line `Prompt: 1,234 tokens (30% of the 4,096-token window)`
  shows its effect.

`/system` is checked the same way: a system prompt that, with the conversation so far,
would not fit is refused and the old one stays.

What the check cannot do: it only knows the prompt. A reply that outgrows the window
while it is being written ends early, as if it had finished (llama-server stops it when
no room is left; it is not an error). That is what the 80% warning is for, and the
optional `reserve_output_tokens` setting, which keeps that many tokens free when deciding
whether a prompt fits. It is a margin, not a guarantee, because the length of a reply
(especially a thinking model's) cannot be known in advance.

Settings (see [configuration](configuration.md)):

| Setting | Effect |
|---|---|
| `context_check` | `false` turns off all of the above: no counting, no refusal, no warning, no prompt line. A prompt that is too large then reaches the server and comes back as its own error. Also `/config context_check on\|off`. Use it if the reported window is wrong |
| `reserve_output_tokens` | Tokens kept free for the reply (default 0) |

The check is skipped, without an error, when the server cannot be reached or reports no
window; the send itself then reports the problem as usual. Files over 8 MB are refused
before any counting ("File too large to read.").

## After each reply
A dim line is printed under the reply, for example:

```
  212 generated (thinking + answer) | 38.4 tok/s | 1530 prompt tokens
```

| Part | Meaning |
|---|---|
| `212 generated (thinking + answer)` | tokens the model generated for this reply, reasoning included |
| `38.4 tok/s` | generation speed (shown when the server reports it) |
| `1530 prompt tokens` | tokens in the request you sent (the whole conversation, as the server counted it) |

`/stats` turns this line off and on. It starts on. No line is shown for a reply you
interrupted with Ctrl-C.

When the window fills, start fresh with `/clear` (or `/save` first), or restart the
server with a larger `-c`. `/recall` is the opposite tool: it adds an old exchange back,
which uses more of the window.

## /info
```
Messages               8 (4 you, 4 AI)
Words                  1,204
Characters             7,318
Tokens: system prompt  40
Tokens: your messages  610
Tokens: AI replies     820
Tokens: template ≈     60
Prompt tokens          1,530
Context window         8,192 tokens
Window used            18.7%
```

`Messages`, `Words` and `Characters` cover the current conversation. The token rows
are what the next prompt would carry (before you type anything), counted by the server
when you run `/info`; thinking is not included. The `template` row is the rest, an
approximation: the chat template's markup and separators. The rows are left out for an
empty conversation or when the server cannot be reached. If the server did not report a
window size, `Context window` says `unknown` and the percentage is omitted. `/info` counts
even when `context_check` is off.
