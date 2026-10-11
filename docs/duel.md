# /duel: two sides write the same story

`/duel` is experimental. Two *sides* draft the same brief at the same time, each on its own
llama-server slot, then read each other's draft, argue over the objections, revise once, and a
neutral judge recommends one. You choose what goes into the conversation.

It makes **no claim that a duel beats a single draft.** Nothing in the tests asserts output
quality, and a duel costs about ten model turns where a normal message costs one. Use it when
you want two takes and a structured argument over them.

## What you need

llama-server must have at least two slots:

```bash
llama-server --port 8001 -m <model> --parallel 2
```

With fewer, or when the app cannot read the slot count from `/props`, `/duel` refuses with a
message. Side A is pinned to slot 0 and side B to slot 1; the judge uses slot 0. A server with
more than two slots leaves slots 2 and up alone.

## Words used here

- **Turn**: one model generation by one conversation.
- **Phase**: a step in which every *participant* takes one turn, at the same time when there
  are two. A phase ends when all its turns have finished; only then does the next start or
  can you be asked anything.
- **Duel**: the fixed sequence below, run once, followed by the final choice.

| # | Phase | Who takes part | What they do |
|---|---|---|---|
| 1 | `draft` | both sides | write the story from the brief |
| 2 | `cross-examine` | both sides | read the opponent's draft and list issues, or say `NO ISSUES` |
| 3 | `defend` | each side with an issue against its draft | `ACCEPT` or `REBUT` each |
| 4 | `reply` | each side whose issue was rebutted | `MAINTAIN` or `WITHDRAW` it |
| 5 | `revise` | each side with a standing issue against its draft, or a steer | rewrite; may `DECLINE` an issue with a reason |
| 6 | `judge` | a neutral judge | recommends `Text 1`, `Text 2` or none |

A phase nobody takes part in is **skipped**: it takes no number and one line says why, for
example `No issues raised; skipping defend, reply and revise.` The shortest duel is three
phases and five turns (draft, cross-examine with no issues on both sides, judge); the longest
is six phases and eleven turns. `merge` (below) adds one phase on request.

There is one pass only. A second pass costs as much again and tends to make both sides
converge on the same flat text; the one pass already ends in a revised text.

## Using it

```
>>> /duel --auto Write a short story about a lighthouse keeper who finds a door.
>>> /duel """                       multiline brief, closed with """
>>> /duel -- set the scene in a lighthouse     (-- ends flag and subcommand parsing)
>>> /duel                           usage, settings, slot count, context assumption
>>> /duel ledger                    the last duel's issues
>>> /duel set <key> <value>         change a setting for this session ('default' resets)
```

`@@<path>` includes work in the brief as in a normal message. `--auto` runs without pauses; it
can also be made the default with `/duel set auto on`. The first word is a subcommand only when
it is exactly `set` or `ledger`.

The sides start fresh: they see the brief, the conversation's current system prompt (plus a
*stance* for each side) and the project notes, **not** the earlier messages of the
conversation. The notes are copied when the duel starts; editing `project.md` during a duel is
not seen. Running a duel overwrites what the two slots had cached, so the next ordinary message
re-processes its prompt.

### Pauses: `duel>`

Without `--auto` the duel pauses between phases (never inside one, and not after the last):

| Input | Effect |
|---|---|
| Enter | continue |
| `a: <note>`, `b: <note>`, `both: <note>` | steer that side's revision. A steer makes the side take part in `revise`. Refused once `revise` has run or been skipped |
| `accept N` / `drop N` | override issue N. `accept` makes it stand, `drop` withdraws it. Allowed until `revise` has run or been skipped |
| `ledger` | show the issues |
| `show a` / `show b` | print that side's current text |
| `auto` | no more pauses until the final choice |
| `stop` | go to the final choice now (also Ctrl-C or end of input) |
| `?` | help |

Who takes part in a phase is decided when it is about to start, after your pause, so an
`accept` can bring `revise` back and a `drop` can cancel it. A side sees only the threads it is
party to, never the other side's steers.

### Issues

An issue has an axis (`plot`, `character`, `dialogue`, `pacing`, `voice` by default; change
with `duel_axes` or `/duel set axes`), a quote, an objection and an alternative. Its status moves
`raised` → `accepted` or `rebutted` → `maintained` or `withdrawn` → `addressed` or `declined`.
A missing answer lets the issue stand: an unanswered issue is accepted in `defend` and maintained
in `reply`, a `cross-examine` reply that is not in the format becomes one `general` issue, and a
revision without a `DECLINE n: reason` line counts as addressing the issue. A model that ignores
the formats therefore never crashes the duel. Replies are parsed on the answer part only, never
on the model's thinking.

### The final choice

The judge's recommendation is shown with its reasons and the mapping from "Text 1/Text 2" to
sides, then:

| Input | Effect |
|---|---|
| `a` / `b` | use that side's text (labelled `draft` or `revised`) |
| `m a` / `m b` | **merge**: the judge rewrites that side's text taking strengths of the other. Bare `m` uses the recommended side and is refused if there is none |
| `c` | use the merged text (offered after a merge; a later `m` replaces it) |
| `j` | run the judge now, when there is no recommendation |
| `s` | show every candidate again |
| `q` | discard; nothing is added (also Ctrl-C or end of input) |

Only the chosen text enters the conversation, as your brief and one reply, and the
conversation is auto-saved as after any reply. `/retry` then regenerates a single normal draft
of the brief in place of the duel's text; the duel's own record keeps the original.

## When something goes wrong

A phase **ends abnormally** when a turn is interrupted (Ctrl-C), refused for context, cut off,
fails with a server error, or gives an empty answer in `draft`, `revise`, `judge` or `merge`
(in the other phases an empty answer is stored as `(no answer)` and the rules above apply). The
whole phase is then discarded, for both sides, including a side whose own turn finished, and the
duel goes no further: you land at the final choice with the current texts. If `draft` itself
fails there are no texts, so the duel ends and nothing is added. A failed `judge` leaves the
menu without a recommendation (`j` retries); a failed `merge` shows the menu again.

## Context

Each side holds the brief, the drafts and every exchange. There is no automatic compaction.
Before a phase the app counts every participant's prompt (see
[statistics](statistics.md#the-pre-send-check)); each one must fit alone, and then the phase
must fit the **pool**:

- `duel_context = "shared"` (the default): the two slots share one KV pool of the reported size,
  as with `--kv-unified`. Every participant's prompt, its reply room and the tokens the idle
  slot still holds must add up to the reported `n_ctx`.
- `duel_context = "split"`: each slot has the reported size to itself (a server started
  without `--kv-unified` reports half the pool per slot). Each prompt need only fit alone.

`/props` does not say which mode is in force, so this is an *assumption*, shown in the header.
Without `--kv-unified` use `split`; with `shared` such a server leaves each side about a
quarter of the pool. Slots 2 and up are not counted. With `context_check` off nothing is
refused, as for a normal message.

The stats line under a side's turn shows how many tokens its slot holds (`ctx`) but not
a share of the window: `/props` cannot say whether that window is the slot's own or the pool
both sides share, so a percentage could mislead.

## Settings

`config.toml` keys (all optional; `/duel set` changes them for the session, `default` resets):

| Key | `/duel set` name | Default | Meaning |
|---|---|---|---|
| `duel_axes` | `axes` | plot, character, dialogue, pacing, voice | axes `cross-examine` may use (comma-separated in `/duel set`) |
| `duel_stance_a`, `duel_stance_b` | `stance_a`, `stance_b` | plot-driven / character-driven | instruction that makes the sides differ |
| `duel_temperature_a`, `duel_temperature_b` | `temperature_a`, `temperature_b` | unset | temperature of one side |
| `duel_seed_a`, `duel_seed_b` | `seed_a`, `seed_b` | unset | seed of one side (`-1` is unset) |
| `duel_context` | `context` | `shared` | `shared` or `split`, see above |
| (none) | `auto` | `off` | `on` or `off`: skip the pauses |

Precedence for the sampling options: the duel setting, then a `/set` override (which applies to
both sides), then the server's default. If both sides end up with the same fixed seed, side B
uses that seed plus one. Both sides are the same weights, so the opposition is only as deep as
the stances make it. The values do not appear in `/config` and a project's `project.toml`
cannot override them. A project switch resets them.

## The record

Every duel that starts writes `duel-YYYYMMDD-HHMMSS.json` in a `duels/` folder inside the
conversations directory (the project's, in a project). It holds the brief, the drafts, the
ledger, steers, the judge's output and the order it saw the texts in, every merge, how the
sequence `ended` (`normal`, `stopped`, or `abnormal:<PHASE>:<cause>`) and what you `choice`
(`a`, `b`, `merged:a`, `merged:b`, or `none`). It is written atomically, is not a conversation
(`/conversations` and `/load` do not see it), and the last one feeds `/duel ledger`.

## Caching

Pinning keeps each side's prefix in its own slot, but reuse is partial: thinking is never sent
back, so for a reasoning model the next prompt diverges from the slot's tokens at the previous
reply, and some models cannot drop part of a sequence at all. Expect re-processing. No speed
claim is made.
