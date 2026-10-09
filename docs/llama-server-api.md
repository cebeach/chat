# llama-server and the API AI Chat uses

AI Chat does not run a model itself. It is a terminal front end for **llama-server**, the
HTTP server that ships with [llama.cpp](https://github.com/ggml-org/llama.cpp). This page
explains what llama-server is, what the parts of its API that AI Chat uses do, and exactly
when and why the app calls each one. You do not need to read it to use the app. It helps
when you are choosing server flags, debugging a connection problem, or want to check by hand
what the app sees.

The authoritative reference is the [llama-server README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
(`tools/server/README.md` in the llama.cpp source). Where this page and that file disagree,
trust the README for the version of llama-server you run.

## What llama-server is

llama-server loads **one model** from a `.gguf` file into memory and then answers HTTP
requests about it. Nothing leaves your machine unless you point the app at another host.
You start it yourself, before the app:

```bash
llama-server --port 8001 -m <model.gguf>
```

The app connects to `http://127.0.0.1:8001` by default (`--url` or `llama_url` changes
that; see [configuration](configuration.md)). llama-server's own default port is 8080, so
the examples in these docs pass `--port 8001`.

The app never picks or loads a model. Whatever the server is running is the model, and the
app asks the server which one it is.

## Four ideas you need

**Tokens.** Models do not read characters; they read *tokens*, pieces of words (an English
word is usually one to three tokens). Everything about size is counted in tokens, and only
the server's own tokenizer can count them exactly.

**The context window** (`n_ctx`). The most tokens the model can consider at once: your
system prompt, the whole conversation so far, the new message, and the reply being written
all share it. You set it when starting the server with `-c` / `--ctx-size`. With `-c 0`
(the default) the server uses the size the model was trained for, which can be very large
and take a lot of memory. With several parallel *slots* (`-np` / `--parallel`; a slot is
one request being worked on), the window one request can use is the per-slot size. That
per-slot size is the figure the server reports and the one the app uses.

**The chat template.** A chat model was trained on conversations written in a specific
format: role markers, special tokens, a marker that says "the assistant speaks now". That
format is the model's *chat template*, and it differs from model to model. A list of
messages must be rendered into that format before the model can use it. The server has the
template (it is stored in the `.gguf` file), so AI Chat asks the server to do the rendering
rather than knowing any format itself.

**The prompt cache.** AI Chat sends the whole conversation with every message. The server
remembers what it processed last time and, when the new prompt starts with the same tokens,
only processes the new part. That is why a long conversation stays fast. The prompt-token
figure the server reports counts the whole prompt, cached part included.

## Two APIs; AI Chat uses the native one

llama-server offers its own *native* endpoints (`/completion`, `/tokenize`, `/props`, ...)
and, in addition, OpenAI-compatible and Anthropic-compatible ones (`/v1/chat/completions`
and others) that let tools written for those services talk to it.

AI Chat uses **only native endpoints**. It never calls anything under `/v1/`. The reason is
control over the prompt text: with the native API the app can ask for the rendered prompt as
plain text, count its tokens exactly before sending, and look at where the template opens a
thinking block ([thinking tags](thinking-tags.md)). The chat-style endpoints take messages
and render them internally, which hides all of that.

## Endpoints the app uses

| Endpoint | What it does | When AI Chat calls it | Why |
|---|---|---|---|
| `GET /health` | Says whether the server is ready | Once, at startup | To fail early with a clear message when no server is running |
| `GET /props` | Server properties: the model, the context window, the chat template | At startup, before every message is sent, and for `/config` and `/info` | To follow the server: a restart with another model or `-c` is noticed. The window size feeds the [token check](statistics.md#the-pre-send-check) |
| `POST /apply-template` | Renders messages with the model's chat template into prompt text, without generating anything | Twice per message (once to count, once for the real request); on `/info` and `/system`; and when a new model is first seen | To build the exact prompt the model will see, and to detect the model's thinking tags |
| `POST /tokenize` | Turns text into tokens | Once per message to count the prompt; several times for `/info` | To measure a prompt exactly before it is sent |
| `POST /completion` | Generates the reply, streamed token by token | Once per message, after the checks pass | The model's answer |

Nothing else is called: no `/slots`, `/metrics`, `/detokenize`, `/embedding`, `/models` or
`/v1/*` endpoint, and the app never changes a server setting (`POST /props` is not used).

Each call has its own timeout so a stuck server does not hang the app forever:
`/health` 5 seconds, `/props` 10, `/apply-template` and `/tokenize` 30 seconds plus a
little more for very large text, `/completion` 120 seconds of silence from the server.

### `GET /health`

Returns `200` with `{"status": "ok"}` when the model is loaded and the server is ready. While
the model is still loading it returns `503`. The endpoint needs no API key.

```bash
curl -s http://127.0.0.1:8001/health
```

The app treats anything other than `200` as "server not available" and exits with
"Cannot connect to llama-server" (see [troubleshooting](troubleshooting.md)). If you start
the server and the app together, a `503` just means the model is still loading: wait and
start the app again.

### `GET /props`

Returns a JSON object describing the running server. AI Chat reads three things from it:

| Field | Used for |
|---|---|
| `model_alias` (else `model_path`) | The model name shown in the welcome line, `/config`, and the "model: old → new" notice, and recorded with each reply |
| `default_generation_settings.n_ctx` | The context window (per slot), for the token check and `/info` |
| `chat_template` | The template's source text, used together with the model path to recognize which model is running (a change means a different model) and to help find its thinking tags |

```bash
curl -s http://127.0.0.1:8001/props | python3 -m json.tool | less
```

The app re-reads it before every message instead of remembering it, so you can stop the
server and start it again with another model or another `-c` in the middle of a session.
If `/props` cannot be read, the app keeps the values it had; it never invents new ones.

### `POST /apply-template`

Takes a list of chat messages and returns the single piece of text the model should be
given, formatted with its chat template. It does not run the model.

```bash
curl -s http://127.0.0.1:8001/apply-template -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]}'
```

The reply is `{"prompt": "..."}`. With a typical template the text contains the role
markers, the messages, and an opening for the assistant's turn at the end; the exact
markup differs per model, which is the point.

The app sends the messages from the conversation (role and content only; a reply's stored
reasoning is not sent) and uses the rendered text in two ways:

1. **To count.** The text is passed to `/tokenize` to find the prompt's size before
   anything is sent.
2. **To send.** The very same kind of call produces the prompt that goes to `/completion`.
   That makes two renders per message when the check is on.

It is also used, with two throw-away probe conversations, the first time a model is seen,
to work out where that model's template puts its thinking tags. The result is remembered
per model, so the probes are not repeated. See [thinking tags](thinking-tags.md).

### `POST /tokenize`

Takes text and returns the tokens the model would make of it. The count is the length of the
list.

```bash
curl -s http://127.0.0.1:8001/tokenize -H 'Content-Type: application/json' \
  -d '{"content":"Hello there","add_special":true}'
```

Two flags matter, and `/tokenize` defaults differ from what a real request does:

| Flag | Default | What AI Chat sends | Why |
|---|---|---|---|
| `add_special` | `false` | `true` for a whole prompt | `/completion` adds the model's start-of-sequence token to a text prompt; the count must include it |
| `parse_special` | `true` | `true` | The template's role markers are counted as the single special tokens they are, not as plain letters |

How the count is used is described in [statistics](statistics.md#how-the-count-works).

### `POST /completion`

Generates text from a prompt. This is the call that produces the reply.

What AI Chat sends:

```json
{
  "model": "<name from /props>",
  "prompt": "<rendered by /apply-template>",
  "stream": true,
  "temperature": 0.3
}
```

`seed`, `temperature` and `top_p` are included only when you set them (with `/set` or in
the config file); otherwise the server's defaults apply. `n_predict` (maximum reply length)
is not sent, so the server's own limit is used, which by default is "until the model
finishes". See the llama.cpp README for the dozens of other options this endpoint takes;
the app uses none of them.

`"stream": true` makes the server answer as it generates, in the *server-sent events*
format: a series of lines, each `data: ` followed by a JSON object.

```
data: {"content": "Hel", "stop": false, ...}
data: {"content": "lo", "stop": false, ...}
data: {"content": "", "stop": true, "model": "...", "tokens_predicted": 2, "tokens_evaluated": 31, "timings": {...}, ...}
```

The app prints each `content` piece as it arrives, which is why replies appear word by
word. The last object (`"stop": true`) carries the totals. AI Chat reads:

| Field | Shown as |
|---|---|
| `model` | Which model wrote the reply (stored with it) |
| `tokens_predicted` | "N generated (thinking + answer)" on the [stats line](statistics.md#after-each-reply) |
| `tokens_evaluated` | "N prompt tokens": the whole prompt, cached part included |
| `timings.predicted_per_second` | "N tok/s" |

The same last object also has `truncated` and `stop_type`, which say whether the reply was
cut short because the context window filled up. The app does not read them, so a reply that
ran out of room looks like a finished one. See
[statistics](statistics.md#the-pre-send-check).

Ctrl-C while a reply is streaming makes the app stop reading the stream; the part received
so far is kept in the conversation and the app returns to the prompt.

## What the app does, step by step

**At startup**
1. `GET /health`: is the server up? If not, the app exits.
2. `GET /props`: which model, which window, which template? If this fails, the app exits.
   Because the model has not been seen yet, two `POST /apply-template` probes then find
   its thinking tags (skipped when `think_start` and `think_end` are set in the config).

**For each message you send** (the numbers are in the order they happen)
1. `GET /props`: re-read the model and the window. The first time a model is seen, the two
   thinking-tag probes (`POST /apply-template`) run here.
2. `POST /apply-template`, then `POST /tokenize`: render and count the prompt. If it cannot
   fit, the app stops here and nothing is sent ([details](statistics.md#the-pre-send-check)).
3. `POST /apply-template`: render the prompt for the real request.
4. `POST /completion`: stream the reply.

With `context_check = false`, step 2 and the second `/props` read are skipped.

**For commands**
- `/config` reads `/props` so the model, window and tags it shows are current.
- `/info` reads `/props` and counts the conversation: one `/apply-template` and one
  `/tokenize` for the total, plus one `/tokenize` for each of the system prompt, your
  messages and the replies that exist.
- `/system` (when setting a prompt) renders and counts the conversation with the new prompt
  to check that it fits.
- `/retry` and `/read` send a message like any other, so they follow the same steps.
- `/save`, `/load`, `/clear`, `/set`, `/stats`, `/recall`, `/cat`, `/conversations` and
  `/exit` make no requests at all.

## Server options that matter to the app

| Option | Effect on AI Chat |
|---|---|
| `-m <file>` | Which model. The app follows it |
| `--port`, `--host` | Where to find the server (`--url` in the app). The default host, `127.0.0.1`, accepts only connections from the same machine |
| `-c` / `--ctx-size` | The context window. A bigger window holds a longer conversation and uses more memory |
| `-np` / `--parallel` | Number of slots. The window one request can use is the per-slot size |
| `--context-shift` | Off by default. When on, the server may drop old tokens from a long reply to keep going. The app's token check assumes it is off |
| `--api-key` | The app sends no key, so a server started with one refuses its requests. Do not use it with AI Chat |

Not supported: llama-server's *router mode* (several models behind one server), because
`/props` has a different shape there.

## Errors

llama-server reports errors as HTTP status codes with a JSON body in the OpenAI error
format, for example `{"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}}`.

| Situation | Status | What the app shows |
|---|---|---|
| Model still loading | 503 from `/health` | "Cannot connect to llama-server" at startup |
| Prompt longer than the window (only reachable with `context_check` off, or when counting failed) | 400 from `/completion` | `llama-server error: 400 ...`; the status line only, not the server's explanation; your message is taken back out of the conversation |
| Server stopped mid-session | connection refused or dropped | "Lost connection to llama-server. Is it still running?" |

To see the server's own explanation of an error, repeat the request with `curl` and read the
body, or look at the server's console output.

## Checking by hand

These are the same calls the app makes, so they are the quickest way to see what it sees
(replace the URL if yours differs):

```bash
URL=http://127.0.0.1:8001
curl -s $URL/health
curl -s $URL/props | python3 -c 'import json,sys; p=json.load(sys.stdin); print(p.get("model_alias") or p["model_path"], p["default_generation_settings"]["n_ctx"])'
curl -s $URL/apply-template -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hi"}]}'
curl -s $URL/tokenize -H 'Content-Type: application/json' -d '{"content":"Hi","add_special":true}'
curl -sN $URL/completion -H 'Content-Type: application/json' -d '{"prompt":"Hi","n_predict":16,"stream":true}'
```

Every other endpoint and option is described in the
[llama-server README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

For the thinking-tag probes in particular, see
[How to check a new model](thinking-tags.md#how-to-check-a-new-model).
