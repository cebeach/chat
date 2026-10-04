# Thinking tags

Thinking models write their reasoning inside tags, and the tags differ per model.
This app needs to know them for two things:

- **Saving:** with `save_thinking` off, `/save` and the automatic saves drop the reasoning
  from the saved JSON (the live session always keeps it).
- **Display:** some chat templates end the prompt with the *opening* tag, so the model only
  writes the closing one. The app re-emits the opening tag so the reply shows a balanced block.

Nothing is hard-coded. The tags are discovered from the llama-server for whichever model is
loaded (`find_think_tags()` and `LlamaClient.detect_think_tags()` in `llama_client.py`).

## Examples

| Model | Reasoning starts with | Reasoning ends with |
|---|---|---|
| Qwen (forced-open template) | `<think>` | `</think>` |
| DeepSeek-R1 distill | `<think>` | `</think>` |
| Gemma 4 | `<|channel>thought` | `<channel|>` |
| gpt-oss (OpenAI "harmony") | `<|channel|>analysis<|message|>` | `<|end|><|start|>assistant<|channel|>final<|message|>` |
| Devstral (Mistral, does not think) | none | none |

Two details worth knowing:

- **Qwen's template forces the tag open.** Its generation prompt already ends in `<think>`, so
  the model never writes it. The app re-emits it.
- **gpt-oss's end tag is several markers.** The end of the reasoning is also the start of the
  answer channel, so the "end tag" covers `<|end|>` plus the final-channel header. Stripping a
  pair therefore leaves just the answer (`Hi`), not a stray header.

## The server does not publish them

llama.cpp knows the tags internally (`thinking_start_tag` and `thinking_end_tags` in
`common/chat.h`, set per template by parsers such as `common/parsers/gemma4.cpp`), but only uses
them to configure sampling (`tools/server/server-common.cpp`). `GET /props` returns the template
source and capability flags (`chat_template_caps` has no tags) and `POST /apply-template`
returns only the rendered prompt. There is nothing to read, so the tags are inferred.

## How detection works

Two renders and the template source, all from native endpoints, no generation:

1. **Continuation render.** `POST /apply-template` with
   `[user "QZUSERQZ", assistant {reasoning_content "QZREASONQZ", content "QZANSWERQZ"}]`,
   `add_generation_prompt: false` and `continue_final_message: true`. When the last message is
   from the assistant, llama.cpp writes its own derived start tag, the reasoning, its end tag and
   the content after the prompt (`common/chat-auto-parser-generator.cpp`). The render therefore
   exposes the server's tags even for templates that never mention `reasoning_content`.
2. **Generation prompt.** `POST /apply-template` with just `[user "QZUSERQZ"]`
   (same user message, so step 1's render begins with it).
3. **Template source.** `GET /props` -> `chat_template`.

**Layer 1 (primary).**

- `end` is the text between `QZREASONQZ` and `QZANSWERQZ`.
- `start` is what follows the generation prompt before `QZREASONQZ`. That is exactly what the
  model has to write to open its reasoning. If nothing follows (a forced-open template such as
  Qwen's), `start` is the generation prompt's own trailing marker.
- Both must be a contiguous run of markers (`<think>`, `[THINK]`, `<|channel|>`, ...) and plain
  words, starting with a marker, no whitespace inside, at most about 60 (start) and 80 (end)
  characters. Prose such as `so the answer is` or labels such as `Answer:` are rejected.
- **Thinking-word guard.** The template source must contain one of `think`, `reason`,
  `thought`, `analysis`, `channel` (case-insensitive). Without it the pair is rejected. This is
  what keeps a family-level guess out: for Devstral the server's Mistral parser writes
  `[THINK]`/`[/THINK]` even though the template never mentions thinking and the model never
  emits them, so detection would otherwise report a pair that does not exist.

**Layer 2 (fallback).** Used only if Layer 1 finds nothing. If the generation prompt ends in an
opened marker (`<X>` or `[X]`), the pair is that marker and its mirror (`</X>` or `[/X]`), but
only if the mirror literally appears in the template source. That check rejects templates whose
prompt merely ends in a role marker (`<|assistant|>`, `<|end_header_id|>`).

**Config override.** `think_start` and `think_end` in `~/.config/chat/config.toml` (both
non-empty) win over detection and make no server call.

## Outcomes

Detection never raises, and has three outcomes:

- **Accepted pair.** Becomes the active tags. Only accepted pairs are memoized (by model path
  and a hash of the template), so a restart with another model is noticed.
- **Conclusive none.** The server wrote no reasoning block, or shape validation / the guard
  rejected it, and Layer 2 found nothing. The active tags become `None` (a non-thinking model).
- **Inconclusive.** Server error, unparseable response, or an anchor mismatch (for example a
  template that embeds a timestamp that changed between the two renders). The previous value
  is kept, so a transient problem cannot make the tags flip.

It **fails closed**: when no pair is accepted, nothing is stripped. The app makes the no-op
visible instead of silent:

- `/config` has a `think_tags` row (`<think> … </think> (detected)`, `… (config)` or
  `none detected`).
- `/config save_thinking off` warns when no tags are known for the current model.
- When the active pair changes between turns (for example you restarted the server with
  another model) one line says so.

Detection runs at the top of every `chat()` call, whatever `save_thinking` is, because the
prefix needs it.

## Conversations that span models

If you restart the server with another model and keep chatting, earlier replies carry the
previous model's tags. The app remembers every pair it knows this session and strips all of them
when saving.

Saved files also record the pairs the saving session knew (an optional `think_pairs` list, at
most the 16 most recent), and `/load` merges them into the session. So loading a conversation
written by a model this session never ran still lets `save_thinking = false` strip its replies
when you save again. Pairs read from a file are untrusted: each must have the shape of a tag pair
(non-empty, a run of markers and plain words, within the length limits) or it is ignored, and the
session never keeps more than 16.

`/load` applies the whole saved conversation (messages, system prompt and where that prompt came
from) against the model being served now. The model names recorded in the file, both the
file-level one and the one on each reply, are information only: they never select a model and are
never sent to the server, and new replies are attributed to the model served now.

Files saved before the `think_pairs` key existed carry no pairs, so their replies from a model
this session never ran keep their reasoning when saved again; set the override to cover them.

## Display

Some models (Gemma, gpt-oss) run straight from the closing tag into the answer
(`…The answer is 144.<channel|>144`); others (Qwen, DeepSeek) already write a blank line after
it. So the REPL draws one newline after the active model's closing tag, unless a newline already
follows it or the reply ends there. `/cat` and `conv2txt` do the same for assistant replies, using
the pairs recorded in the saved file (a file saved without `think_pairs` is shown as it is).

This is display only. The stored reply, the saved JSON, the history sent back to the model and
strip-on-save never contain the added newline. `/cat` also escapes the text it prints, so a saved
reply containing something like `[/THINK]` or `[/path]` is shown literally instead of failing.

## Known limits

- **Asymmetric tags** such as Gemma's work only through Layer 1 (Layer 2 needs a mirror) or the
  override.
- **Delimiter-style templates**, where the server derives no start tag, make the server write
  neither the reasoning nor the end tag, so Layer 1 yields nothing. Layer 2 or the override has
  to cover them, otherwise `/config` says `none detected`.
- **The tags are only as good as llama.cpp's own derivation plus the guard.** The guard is a
  word list: a template that implements thinking without any of those words fails closed.
- **A wrong detection** is usually harmless, because the bogus pair never occurs in real
  replies and nothing is stripped. The worst case is a reply that quotes the pair: the text
  between the tags is removed from the *saved file* (never from memory), and only while
  `save_thinking` is off.
- **Router mode** is not supported (`/props` has a different shape there).
- **Outgoing history is not stripped.** The app sends each earlier reply back to the model
  unchanged, including its reasoning. Qwen's template removes it itself, but for gpt-oss the
  history then contains the model's earlier reasoning and its control tokens. Verified, and left
  to a separate follow-up.

## How to check a new model

Restart the server with the model, then run `/config` and read the `think_tags` row. If it says
`none detected` but the model does think, look at the evidence. These are the three calls
detection makes (replace the URL if needed):

```bash
URL=http://127.0.0.1:8001

# 1. continuation render: the server writes its own tags around the reasoning
curl -s $URL/apply-template -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"QZUSERQZ"},{"role":"assistant","reasoning_content":"QZREASONQZ","content":"QZANSWERQZ"}],"add_generation_prompt":false,"continue_final_message":true}'

# 2. generation prompt: the anchor for the start tag
curl -s $URL/apply-template -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"QZUSERQZ"}]}'

# 3. template source: needed for the guard and for Layer 2's mirror check
curl -s $URL/props | python3 -c 'import json,sys; print(json.load(sys.stdin)["chat_template"])'
```

Read the results like this:

- `end` is the text between `QZREASONQZ` and `QZANSWERQZ` in call 1.
- `start` is the text of call 1 before `QZREASONQZ`, minus the whole of call 2.
- No `QZREASONQZ` in call 1 means the server wrote no reasoning block (non-thinking, or a
  delimiter-style template).
- Call 3 must contain one of the guard words.

Sample output (DeepSeek-R1-distill, as run when this was written; other models differ):

```text
1: {"prompt":"<｜User｜>QZUSERQZ<｜Assistant｜><think>QZREASONQZ</think>QZANSWERQZ"}
2: {"prompt":"<｜User｜>QZUSERQZ<｜Assistant｜>"}
```

Here `start` = `<think>` and `end` = `</think>`. For gpt-oss, call 1 ends
`...<|start|>assistant<|channel|>analysis<|message|>QZREASONQZ<|end|><|start|>assistant<|channel|>final<|message|>QZANSWERQZ`
and call 2 ends `...<|start|>assistant`, giving the pair in the table above. If the pair cannot
be detected, set it by hand:

```toml
# ~/.config/chat/config.toml
think_start = "<|channel|>analysis<|message|>"
think_end = "<|end|><|start|>assistant<|channel|>final<|message|>"
```

For the developer test suite, record what the model really emits as `think_tags` in its profile in
`tests/llama-server.toml`. The integration tests then check the detection and a real reply against
it; see [Testing with a real llama-server](testing.md#thinking-tag-tests).
