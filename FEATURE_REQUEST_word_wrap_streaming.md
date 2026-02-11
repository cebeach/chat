# Feature Request: Prevent word breaking in streamed chat output

## Problem

During streaming, assistant responses are written character-by-character via
`sys.stdout.write(token)` in `ui.py:display_assistant_stream`. The terminal
performs character-level wrapping at the right edge, which splits words
mid-character. This makes the raw streaming output harder to read, especially
with long words, URLs, or code identifiers.

After streaming completes, the text is re-rendered as Rich Markdown which
word-wraps correctly — but the visual jarring during streaming remains.

## Proposed Solution

Wrap streamed output so that line breaks respect word boundaries instead of
splitting at the terminal column boundary. Two possible approaches:

### Approach 1: Buffered word wrapping (recommended)

Accumulate tokens into a line buffer and flush whole words. When appending a
token would exceed the terminal width, emit a newline before the current word.
This preserves the streaming feel while avoiding mid-word breaks.

Sketch:

```python
term_width = console.width or 80
col = 0
word_buf = ""

for token in token_generator:
    word_buf += token
    while "\n" in word_buf:
        before, _, word_buf = word_buf.partition("\n")
        sys.stdout.write(before + "\n")
        col = 0
    # Flush on whitespace boundaries
    while " " in word_buf:
        word, _, word_buf = word_buf.partition(" ")
        if col + len(word) + 1 > term_width and col > 0:
            sys.stdout.write("\n")
            col = 0
        if col > 0:
            sys.stdout.write(" ")
            col += 1
        sys.stdout.write(word)
        col += len(word)
    sys.stdout.flush()

# Flush remaining buffer
if word_buf:
    if col + len(word_buf) > term_width and col > 0:
        sys.stdout.write("\n")
    sys.stdout.write(word_buf)
```

### Approach 2: Rich Live re-render

Use `rich.live.Live` to continuously re-render the in-progress text with
proper word wrapping, overwriting the previous output each update. Gives
richer formatting during streaming but adds complexity and may flicker.

## Additional Context

- The line-counting logic that erases raw output before markdown re-render
  (`visual_lines` calculation in `display_assistant_stream`) already assumes
  character-level wrapping via `len(line) / term_width`. A word-wrap buffer
  would need to keep the line count in sync.
- Terminal width is available via `console.width`.
- Zero new dependencies required — stdlib + existing Rich.
