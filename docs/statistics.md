# Statistics and the context window

A model can only consider a limited number of tokens at once: its *context window*.
The app reads the window size from llama-server (`/props`), so it is whatever the
server was started with (`llama-server -c ...`). Your whole conversation, including the
system prompt, is sent with each message, so it grows with every turn.

## After each reply
A dim line is printed under the reply, for example:

```
  212 tokens | 38.4 tok/s | 1530 prompt tokens | ctx 1,742 / 8,192 (21.3%)
```

| Part | Meaning |
|---|---|
| `212 tokens` | tokens the model generated |
| `38.4 tok/s` | generation speed (shown when the server reports it) |
| `1530 prompt tokens` | tokens in the request you sent (the whole conversation) |
| `ctx 1,742 / 8,192 (21.3%)` | prompt + generated tokens against the window (shown only when the window size is known) |

`/stats` turns this line off and on. It starts on. No line is shown for a reply you
interrupted with Ctrl-C.

## Context warning
When prompt + generated tokens exceed 80% of the window, a yellow warning follows the
reply:

```
Warning: context window 85% full (6,963 / 8,192 tokens)
```

This appears even when `/stats` is off. When the window fills, start fresh with
`/clear` (or `/save` first), or restart the server with a larger `-c`. `/recall` is
the opposite tool: it adds an old exchange back, which uses more of the window.

## /info
```
Messages        8 (4 you, 4 AI)
Words           1,204
Characters      7,318
Prompt tokens   1,530
Context window  8,192 tokens
Context used    1,742 tokens
Context usage   21.3%
```

`Messages`, `Words` and `Characters` cover the current conversation. The token rows
describe the **last reply only** and appear after the first one; they are cleared by
`/clear`, `/load` and `/retry`. If the server did not report a window size, `Context
window` says `unknown` and the percentages are omitted.
