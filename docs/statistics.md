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
while it is being written is stopped early by llama-server (when no room is left; it is not
an error). The 80% warning and the optional `reserve_output_tokens` setting, which keeps
that many tokens free when deciding whether a prompt fits, make that less likely. They are a
margin, not a guarantee, because the length of a reply (especially a thinking model's)
cannot be known in advance.

### A reply that was cut off
When it happens anyway, the server says so in its last message and the app prints a warning
after the reply:

```
Warning: the reply was cut off because the context window is full. Free space with /clear, or restart llama-server with a larger -c.
```

The partial reply is kept and saved like any other. A thinking model can be cut off while it
is still thinking; the warning then adds "The model was still thinking, so there is no
answer." Such a reply is stored with thinking only, and a reply with no answer is left out,
together with your message before it, when the conversation is sent again, so the model
will not see that exchange. After restarting the server with a larger `-c`, `/retry` asks
again; after `/clear` the conversation is empty, so send the message again.

A reply that stops because the server was started with a token limit (`--n-predict`) is not
this: the window did not fill, and no warning is shown.

Settings (see [configuration](configuration.md)):

| Setting | Effect |
|---|---|
| `context_check` | `false` turns off all of the above: no counting, no refusal, no warning, no prompt line. A prompt that is too large then reaches the server and comes back as its own error. Also `/config context_check on\|off`. Use it if the reported window is wrong |
| `reserve_output_tokens` | Tokens kept free for the reply (default 0) |

Files over 8 MB are refused before any counting ("File too large to read.").

### When the count cannot be made
If the server cannot be reached, answers the counting calls with an error, or reports no
window size, no count can be made, so there is nothing to refuse or warn about: no message
is shown and the send goes ahead. The send then
meets the same problem and reports it as it always has: "Lost connection to
llama-server" or `llama-server error: ...`, and the unanswered message is taken back out of
the conversation. A failed count therefore never blocks a send that would have worked. The
cost is that if the prompt really was too large, you get the server's own error (an HTTP
400) instead of the "Not sent" message above, and that error shows only the HTTP status,
not the server's explanation.

Ctrl-C while the count is being made cancels that message: "Cancelled." is printed, nothing
is sent or stored, and the app returns to the prompt.

## How the count works
The app does not estimate: it asks llama-server, which has the model's own chat template and
tokenizer. For each message it makes these calls:

1. **Window.** `GET /props` re-reads the context window the server is using right now, so
   a server restarted with another `-c` is noticed. This is the per-slot size, the limit one
   request must fit.
2. **Candidate.** The prompt is the system prompt, the stored messages and the new message,
   exactly as they will be sent: role and text only. A reply's stored reasoning is not
   included, and neither is an empty reply with the question before it, because those are
   never sent.
3. **Render.** `POST /apply-template` returns that candidate as the server will give it to
   the model: the template's role markers, special tokens and the opening of the reply.
   This is the same call that builds the real request.
4. **Tokenize.** `POST /tokenize` returns the tokens of the rendered text, and the count is
   their number. It asks for the same special-token handling the real request uses
   (`add_special` and `parse_special` both true), so a start-of-sequence token is counted
   and markers such as `<|im_start|>` count as the single tokens they are. Without
   `add_special` the count would be short on models that add one.

The count therefore equals what the server reports as the request's prompt tokens after
the reply. A test against a real server checks that equality.

A prompt fits when `prompt + reserve_output_tokens + 1 <= window`. The `+ 1` is because
llama-server rejects a request whose prompt alone is as long as the window, even with
nothing to generate. The warning is for a prompt above 80% of the window.

Counting costs three small requests per message, and for a large included file the
tokenizing can take seconds. The keyboard echo is off during it, as during a reply, so keys
typed meanwhile do not show up in the prompt line.

`/info` uses the same calls. Its total is the exact count for the stored conversation (the
next prompt before you type anything). The system, your-messages and AI-replies rows are
counted one group at a time without the start-of-sequence token, and `template` is the total
minus those, never below 0. Counted apart, the pieces can differ from the whole by a token
or two, which is why that row is marked as an approximation.

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
