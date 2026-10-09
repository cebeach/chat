# Projects

A project is a named directory that holds its own conversations, a system prompt and a
file of standing notes the model always sees. Use one for a long piece of work (a novel,
a spec, a research question) so the same background does not have to be retyped. With no
project selected the app behaves exactly as before.

## What is in a project

```
~/.local/share/chat/projects/<name>/
  project.toml     optional settings for this project
  project.md       standing notes, sent with every prompt
  system.md        the project's system prompt
  conversations/   saved conversations (same format as the global directory)
```

`projects_dir` in [the config file](configuration.md) moves the parent directory.

- **Names** are ASCII letters, digits, `-` and `_` only. Anything else is refused (unlike
  conversation names, which are cleaned up), so `my novel` is an error, not `my_novel`.
- **`project.md`** is plain markdown that you and `/remember` both maintain. It is sent
  in the system message after the system prompt, under a `# Project notes` heading, so
  it costs tokens on every turn: [`/info`](statistics.md#info) shows the cost and warns
  when it passes a quarter of the window. Edit it in any editor; the change is picked up
  before the next message and one line says so. An edit changes the start of the prompt,
  so the server has to process the whole conversation again once; edit deliberately.
- **`system.md`** is the whole system prompt. If it is empty or missing, the global
  `system_prompt` applies (a project cannot switch the prompt off). It is read when you
  switch to the project and on `/project reload`, not on every message, so editing it
  does not change the prompt under a conversation in progress.
- **`project.toml`** overrides settings for this project: `seed`, `temperature`, `top_p`,
  `min_p`, `repeat_penalty`, `n_predict`, `context_check` and `reserve_output_tokens`.
  Everything else (the server URL, saving, thinking) stays global. Unknown keys are
  ignored; a `system_prompt` key is ignored with a note, because the prompt lives in
  `system.md`.

```toml
temperature = 0.7
reserve_output_tokens = 1500
```

## Using projects

```
>>> /project new novel
>>> /project use novel
>>> /remember Anna is left-handed and never uses contractions
```

Or start in one: `python chat.py --project novel`. See [`/project`](commands.md#project)
for all the subcommands.

**Switching.** `/project use` and `/project leave` save the current conversation to the
directory it belongs to (even if `auto_save` is off, because the switch empties it), then
start an empty conversation. If the save fails the switch is cancelled and nothing
changes. A switch, a leave and a reload also reset `/set`, `/config` and `/stats` to what
the config files say, and print a line saying so.

**Saving and loading.** `/save`, `/load`, `/cat`, `/conversations` and auto-save use the
project's `conversations/` directory, and Tab completes its names. `/load` inside a
project keeps the project's system prompt and ignores the one stored in the file. The
saved file does not carry the notes, so a conversation loaded later sees the notes of the
day. [`conv2txt.py`](conversations.md#converting-to-plain-text) takes any path, so give it
a file from the project's directory.

**Reloading.** After editing `system.md` or `project.toml`, run `/project reload`: it
reads the three files again and keeps the conversation.

## /remember

`/remember <text>` asks the model to merge a note into `project.md`. The model gets the
current file and your note and returns the complete new file, which is shown as a diff.
Nothing is written unless you answer `y`; anything else, Ctrl-C or Ctrl-D leaves the file
alone. The conversation itself is not touched.

It refuses, with the reason and without changing anything, when:

- there is no project, or the server or its context window size cannot be read;
- the reply could not fit in the window (room is kept for a reply as long as the notes plus
  a note, and for some thinking when there is room). This check applies even with
  `context_check` off, and `reserve_output_tokens` is not added to it;
- the reply was cut off, is empty, or is under half the size of the old file;
- `project.md` changed while you were deciding (edited elsewhere): run it again.

`/remember` always asks for a reply at temperature 0, whatever you set with `/set`. A small
model may merge badly; read the diff before you answer `y`.

## Limits

- Nothing is searched or summarized: the notes are sent whole, every time. Keep them short.
- A project has no delete command; remove its directory by hand.
