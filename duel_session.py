"""The /duel session: phases, pauses and the final choice (docs/duel.md).

chat.py's names reach this module only through the `host` every function that prints or asks
is handed (chat._duel_host(), built per call; each member looks its chat name up when used).
The members: auto_save, conversations_dir, fits_window, initial_duel, prepare_user_message,
reply_model, sync_notes, sync_server_info, update_think_tags, validated, display_error,
display_info, display_ledger, display_stats, display_assistant_stream, echo_suppressed,
get_multiline_input, console and option_keys. This module never imports chat.
"""

import queue
import random
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from requests.exceptions import ConnectionError, HTTPError
from rich.markup import escape

import duel
import tokens
from conversation import Conversation, split_think
from duel import Phase
from llama_client import ThinkTagsError


@dataclass
class _Job:
    """One participant's turn in a phase: whose conversation, what to say, which slot."""

    key: str  # "a", "b" or duel.JUDGE
    conversation: Conversation
    text: str
    slot: int
    options: dict


@dataclass
class _Turn:
    """What a finished turn produced: the answer part, the thinking split off it, the stats."""

    answer: str
    thinking: str
    stats: dict
    model: str


@dataclass
class _PhaseResult:
    ok: bool
    cause: str | None = None  # one of duel.CAUSES when not ok
    turns: dict = field(default_factory=dict)


class _Part:
    """A participant's side of the thread hand-off: a worker fills it, the main thread reads it."""

    def __init__(self, job):
        self.job = job
        self.queue = queue.Queue()  # ("start", stream) / ("tok", text) / ("end", None)
        self.tokens = []  # every token received, for the full text whatever was displayed
        self.done = threading.Event()
        self.error = None
        self.stream = None
        self.stats = {}
        self.cut_off = False


def _turn_worker(client, model, part, messages, stop):
    """Run one turn in a thread. Every exception is handed to the main thread through `part`."""
    try:
        stream = client.chat(
            model=model,
            messages=messages,
            options=part.job.options,
            refreshed=True,
            id_slot=part.job.slot,
        )
        part.stream = stream
        if stop.is_set():
            stream.close()
            return
        part.queue.put(("start", stream))
        for token in stream:
            if stop.is_set():
                break
            part.tokens.append(token)
            part.queue.put(("tok", token))
        part.stats = stream.stats
        part.cut_off = bool(stream.truncated or stream.limited or not stream.stats)
    except BaseException as exc:
        part.error = exc
    finally:
        part.done.set()
        part.queue.put(("end", None))


def _await_start(part):
    """The part's stream once its request is answered, or None when it failed first."""
    kind, value = part.queue.get()
    if kind == "start":
        return value
    part.queue.put((kind, value))
    return None


def _drain(part):
    """The part's tokens: those already received, then the rest live while it is still generating."""
    while True:
        kind, value = part.queue.get()
        if kind != "tok":
            return
        yield value


def _duel_error_text(exc, label):
    if isinstance(exc, ConnectionError):
        return "Lost connection to llama-server. Is it still running?"
    if isinstance(exc, ThinkTagsError):
        return str(exc)
    if isinstance(exc, HTTPError):
        return f"llama-server error: {exc}"
    return f"{label}: {exc}"


def _reserve(state):
    """Output room kept per prompt, as _fits_window computes it."""
    return max(state.config.get("reserve_output_tokens", 0), state.options.get("n_predict", 0))


def _price_phase(phase, jobs, client, state, held, host):
    """Price every participant, then the pool. Returns (priced, None) or (None, cause).

    Each participant is priced alone by _fits_window, which prints its own refusal; only if
    all pass is the joint rule applied. `priced` maps a job key to (needed, n_ctx) or None
    (pricing unavailable: nothing is refused, the request that follows meets any real problem).
    """
    priced = {}
    for job in jobs:
        candidate = job.conversation.get_messages() + [{"role": "user", "content": job.text}]
        outcome = f"Not sent ({duel.side_label(job.key)}, {phase.value})"
        ok, result = host.fits_window(client, candidate, state, outcome=outcome)
        if not ok:
            return None, "interrupted" if result is None else "context"
        priced[job.key] = result
    if state.duel["context"] == "shared" and all(p is not None for p in priced.values()):
        n_ctx = min(p[1] for p in priced.values())
        needed = [p[0] for p in priced.values()]
        used = {job.slot for job in jobs}
        idle = sum(tokens_held for slot, tokens_held in held.items() if slot not in used)
        reserve = _reserve(state)
        if not tokens.fits_together(needed, idle, n_ctx, reserve):
            outcome = f"Not sent ({phase.value})"
            host.display_error(tokens.refusal_together(needed, idle, n_ctx, reserve, outcome))
            return None, "context"
    return priced, None


def _run_phase(phase, jobs, client, state, held, host):
    """Run one phase: every job takes one turn, concurrently when there are several.

    Returns a _PhaseResult. A phase that ends abnormally is discarded as a whole: each
    participant's conversation loses what the phase appended, and the caller must not advance.
    `held` maps a slot to the tokens it still holds; it is updated here. Everything runs with
    the keyboard echo off; threads exist only inside this call.
    """
    with host.echo_suppressed():
        priced, cause = _price_phase(phase, jobs, client, state, held, host=host)
        if priced is None:
            return _PhaseResult(False, cause)
        marks = [(job.conversation, len(job.conversation.messages)) for job in jobs]
        for job in jobs:
            job.conversation.add_user(job.text)
        if any(p is None for p in priced.values()):
            client.refresh()  # pricing did not; the workers then skip it
        reserve = _reserve(state)
        stop = threading.Event()
        interrupted = threading.Event()
        parts = [_Part(job) for job in jobs]
        threads = [
            threading.Thread(
                target=_turn_worker,
                args=(client, state.model, part, part.job.conversation.get_messages(), stop),
                daemon=True,
            )
            for part in parts
        ]
        # The first participant streams live; each other one is shown, in full, through the
        # same display once it is its turn (everything received so far, then the rest live).
        # Starting the threads is inside the try: a Ctrl-C can arrive before the display begins.
        try:
            for thread in threads:
                thread.start()
            for part in parts:
                stream = _await_start(part)
                if stream is not None:
                    host.sync_server_info(state, stream.server_model, stream.server_n_ctx)
                    host.update_think_tags(state, stream.think_tags)
                    host.display_assistant_stream(
                        _drain(part),
                        think_end=stream.think_tags[1] if stream.think_tags else None,
                        label=f"{duel.side_label(part.job.key)}:",
                        interrupted=interrupted,
                    )
                if interrupted.is_set() or part.error is not None:
                    break
                if state.show_stats:
                    host.display_stats(part.stats)
        except KeyboardInterrupt:
            interrupted.set()

        complete = all(p.done.is_set() and p.error is None for p in parts)
        cause = None
        if interrupted.is_set() and not complete:
            host.console.print()
            host.display_info(f"{phase.value} interrupted.")
            cause = "interrupted"
        elif any(p.error is not None for p in parts):
            for part in parts:
                if part.error is not None:
                    host.display_error(_duel_error_text(part.error, duel.side_label(part.job.key)))
            cause = "server_error"
        elif any(p.cut_off for p in parts):
            for part in parts:
                if part.cut_off:
                    host.display_error(
                        f"{duel.side_label(part.job.key)}'s reply in {phase.value} was cut off "
                        "(context window full or token limit), so the phase is discarded."
                    )
            cause = "cut_off"

        turns = {}
        if cause is None:
            for part in parts:
                text = "".join(part.tokens)
                tags = part.stream.think_tags
                thinking, answer = split_think(text, tags)
                if not duel.usable_text(phase, answer):
                    if phase in duel.EMPTY_IS_ABNORMAL:
                        host.display_error(
                            f"{duel.side_label(part.job.key)} gave no answer in {phase.value}, "
                            "so the phase is discarded."
                        )
                        cause = "empty_answer"
                        break
                    answer = duel.NO_ANSWER
                turns[part.job.key] = _Turn(
                    answer, thinking, part.stats, host.reply_model(part.stream, state)
                )

        if cause is not None:
            stop.set()
            for part in parts:
                if part.stream is not None:
                    part.stream.close()
            for conversation, size in marks:
                del conversation.messages[size:]
            for part in parts:  # the slot also holds what the discarded turn used
                if priced[part.job.key] is not None:
                    needed = priced[part.job.key][0] + reserve
                    held[part.job.slot] = max(held.get(part.job.slot, 0), needed)
            return _PhaseResult(False, cause)

        for part in parts:
            turn = turns[part.job.key]
            part.job.conversation.add_assistant(turn.answer, model=turn.model, thinking=turn.thinking or None)
            if turn.stats.get("context_tokens"):
                held[part.job.slot] = turn.stats["context_tokens"]
        for thread in threads:
            thread.join(timeout=5)
        return _PhaseResult(True, None, turns)


# The slot each conversation of a duel is pinned to; the judge shares side A's.
_DUEL_SLOTS = {"a": 0, "b": 1, duel.JUDGE: 0}

# Injectable by tests: the random order in which the judge sees the two texts.
_duel_rng = random.Random()

_DUEL_KEYS = (
    "axes",
    "stance_a",
    "stance_b",
    "temperature_a",
    "temperature_b",
    "seed_a",
    "seed_b",
    "context",
    "auto",
)

_DUEL_PAUSE_HELP = (
    "Enter: continue.  a: / b: / both: <note> steer a side's revision.  ledger: show the issues.\n"
    "accept N / drop N: override an issue.  show a|b: print a side's current text.\n"
    "auto: no more pauses.  stop: go to the final choice now.  ?: this help."
)


@dataclass
class _Duel:
    """One duel in progress: the ledger, the sides' conversations and texts, the record."""

    brief: str
    includes: list
    started: datetime
    auto: bool
    ledger: duel.Ledger = field(default_factory=duel.Ledger)
    sides: dict = field(default_factory=dict)  # "a"/"b" -> Conversation
    options: dict = field(default_factory=dict)  # "a"/"b" -> request options
    held: dict = field(default_factory=lambda: {0: 0, 1: 0})
    texts: dict = field(default_factory=dict)  # side -> current story text
    kinds: dict = field(default_factory=dict)  # side -> "draft" or "revised"
    models: dict = field(default_factory=dict)  # side -> model that wrote the current text
    number: int = 0  # phases announced so far
    phases: list = field(default_factory=list)
    judge: dict | None = None
    merges: list = field(default_factory=list)
    ended: str | None = None
    choice: str | None = None
    record_path: Path | None = None
    record_warned: bool = False
    settings: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)


def _duel_usage(client, state, host):
    """/duel alone: usage, the settings and what the server offers."""
    client.refresh()
    slots = client.server_slots if client.server_slots is not None else "unknown"
    n_ctx = f"{client.server_n_ctx:,}" if client.server_n_ctx else "unknown"
    setting = state.duel
    host.display_info('Usage: /duel [--auto] <brief>   (""" for a multiline brief; -- ends flag parsing)')
    host.display_info(
        "       /duel set <key> <value>   keys: " + ", ".join(_DUEL_KEYS) + "; 'default' resets"
    )
    host.display_info("       /duel ledger              show the last duel's issues")
    host.display_info(
        f"Server slots: {slots} (a duel needs 2: llama-server --parallel 2); context n_ctx: {n_ctx}"
    )
    host.display_info(
        f"Context assumption: {setting['context']} (shared: both sides share one pool of n_ctx; split: each has n_ctx)"
    )
    host.display_info(
        f"Axes: {', '.join(setting['axes'])}; auto pauses: {'off' if setting['auto'] else 'on'}"
    )
    for side in duel.SIDES:
        extra = [
            f"{k.split('_')[0]} {setting[k]}"
            for k in (f"temperature_{side}", f"seed_{side}")
            if setting[k] is not None
        ]
        suffix = f" ({', '.join(extra)})" if extra else ""
        host.display_info(f"Side {side.upper()}{suffix}: {escape(setting[f'stance_{side}'])}")


def _duel_set(key, value, state, host):
    """/duel set <key> <value>: change a setting for this session ('default' resets it)."""
    if key not in _DUEL_KEYS:
        host.display_error(f"Unknown duel setting: {key}. Available: {', '.join(_DUEL_KEYS)}")
        return
    value = value.strip()
    if not value:
        host.display_info(f"{key}: {state.duel[key]}")
        return
    defaults = host.initial_duel(state.global_config, quiet=True)
    if value == "default":
        state.duel[key] = defaults[key]
        host.display_info(f"{key} reset to {defaults[key]}.")
        return
    try:
        if key == "axes":
            axes = [a.strip() for a in value.split(",")]
            if not all(axes):
                raise ValueError("axes must be a comma-separated list of names")
            new = axes
        elif key.startswith("stance_"):
            new = value
        elif key == "context":
            if value not in ("shared", "split"):
                raise ValueError("context must be shared or split")
            new = value
        elif key == "auto":
            if value not in ("on", "off"):
                raise ValueError("auto must be on or off")
            new = value == "on"
        else:
            option = key.split("_")[0]
            try:
                number = host.option_keys[option](value)
            except ValueError:
                raise ValueError(
                    f"{key} must be {host.option_keys[option].__name__} (or 'default')"
                ) from None
            new = host.validated(option, number)
            if option == "seed" and new == -1:
                new = None
    except ValueError as exc:
        host.display_error(f"{key}: {exc}.")
        return
    state.duel[key] = new
    host.display_info(f"{key} set to {new}.")


def _duel_ledger(state, host):
    record = state.duel_last
    if record is None:
        host.display_info("No duel yet.")
        return
    host.display_ledger([duel.Issue(**i) for i in record["ledger"]["issues"]])


def _duel_record(d):
    """The record dict for the duel as it stands (rebuilt on every save)."""
    d.record.update(
        {
            "started": d.started.isoformat(),
            "brief": d.brief,
            "includes": d.includes,
            "settings": d.settings,
            "options": d.options,
            "drafts": d.record.get("drafts", {}),
            "texts": dict(d.texts),
            "kinds": dict(d.kinds),
            "ledger": d.ledger.to_dict(),
            "phases": d.phases,
            "judge": d.judge,
            "merges": d.merges,
            "ended": d.ended,
            "choice": d.choice,
        }
    )
    return d.record


def _save_duel_record(d, state, host):
    """Write (the first time) or rewrite the record; a failure warns once and the duel goes on."""
    record = _duel_record(d)
    try:
        if d.record_path is None:
            directory = Path(host.conversations_dir(state)).expanduser() / "duels"
            d.record_path = duel.write_record(directory, record, d.started)
        else:
            duel.rewrite_record(d.record_path, record)
    except OSError as exc:
        if not d.record_warned:
            host.display_error(f"Could not write the duel record: {exc}")
            d.record_warned = True


def _show_text(title, text, host):
    host.console.print(f"[bold]{escape(title)}[/bold]")
    host.console.print(text, markup=False, highlight=False)


def _duel_pause(d, after, host):
    """The duel> prompt between two phases. Returns "stop" or None (continue)."""
    step = duel.next_phase(after, d.ledger)
    if step is not None:
        host.display_info(f"Next: {step[0].value}. Enter to continue, ? for help.")
    while True:
        try:
            line = input("duel> ").strip()
        except (EOFError, KeyboardInterrupt):
            host.console.print()
            return "stop"
        low = line.lower()
        if not line:
            return None
        steer = re.match(r"(a|b|both)\s*:\s*(.+)$", line, re.I | re.S)
        override = re.match(r"(accept|drop)\s+#?(\d+)$", low)
        show = re.match(r"show\s+(a|b)$", low)
        if steer:
            try:
                d.ledger.add_steer(steer.group(1).lower(), steer.group(2).strip())
            except duel.Refused as exc:
                host.display_error(str(exc))
            else:
                target = (
                    "both sides" if steer.group(1).lower() == "both" else f"Side {steer.group(1).upper()}"
                )
                host.display_info(f"Queued for {target}'s revision.")
        elif override:
            try:
                (d.ledger.override_accept if override.group(1) == "accept" else d.ledger.override_drop)(
                    int(override.group(2))
                )
            except duel.Refused as exc:
                host.display_error(str(exc))
            else:
                host.display_info(
                    f"Issue {override.group(2)} {'accepted' if override.group(1) == 'accept' else 'dropped'}."
                )
        elif low == "ledger":
            host.display_ledger(d.ledger.issues)
        elif show:
            side = show.group(1)
            _show_text(
                f"Side {side.upper()} ({d.kinds.get(side, 'no text yet')})", d.texts.get(side, ""), host=host
            )
        elif low == "auto":
            d.auto = True
            host.display_info("No more pauses until the final choice.")
            return None
        elif low == "stop":
            return "stop"
        elif low == "?":
            host.console.print(_DUEL_PAUSE_HELP, markup=False, highlight=False)
        else:
            host.display_error("Unknown command. ? for help.")


def _duel_jobs(d, phase, who, state):
    """The jobs of `phase`: each participant's conversation, prompt and slot."""
    ledger, brief, axes = d.ledger, d.brief, state.duel["axes"]
    jobs = []
    for side in who:
        if phase == Phase.DRAFT:
            text = duel.draft_prompt(brief)
        elif phase == Phase.CROSS:
            text = duel.cross_prompt(brief, d.texts[duel.other(side)], axes)
        elif phase == Phase.DEFEND:
            text = duel.defend_prompt(brief, ledger.with_status(duel.RAISED, against=side))
        elif phase == Phase.REPLY:
            text = duel.reply_prompt(brief, ledger.with_status(duel.REBUTTED, by=side))
        else:  # REVISE
            text = duel.revise_prompt(
                brief, d.texts[side], ledger.standing(against=side), ledger.steers[side]
            )
        jobs.append(_Job(side, d.sides[side], text, _DUEL_SLOTS[side], d.options[side]))
    return jobs


def _announce_phase(d, phase, who, host):
    d.number += 1
    who_note = ""
    if len(who) == 1 and who[0] in duel.SIDES:
        who_note = f" ({duel.side_label(who[0])})"
    host.console.print(f"[info]Phase {d.number} · {phase.value}{who_note}[/info]")


def _apply_phase(d, phase, result, state, host):
    """Fold a finished phase into the ledger and texts."""
    for side, turn in result.turns.items():
        if phase == Phase.DRAFT:
            d.texts[side], d.kinds[side], d.models[side] = turn.answer, "draft", turn.model
            d.record.setdefault("drafts", {})[side] = turn.answer
        elif phase == Phase.CROSS:
            raised = duel.apply_cross(d.ledger, side, turn.answer, state.duel["axes"])
            word = "issue" if len(raised) == 1 else "issues"
            host.display_info(f"{duel.side_label(side)} raised {len(raised) or 'no'} {word}.")
        elif phase == Phase.DEFEND:
            duel.apply_defend(d.ledger, side, turn.answer)
        elif phase == Phase.REPLY:
            duel.apply_reply(d.ledger, side, turn.answer)
        elif phase == Phase.REVISE:
            d.texts[side] = duel.apply_revise(d.ledger, side, turn.answer)
            d.kinds[side], d.models[side] = "revised", turn.model


def _run_judge(d, client, state, host):
    """JUDGE on a fresh neutral conversation. Returns the _PhaseResult."""
    order = duel.judge_order(_duel_rng)
    conversation = duel.new_conversation(duel.JUDGE_SYSTEM)
    text = duel.judge_prompt(d.brief, [d.texts[s] for s in order])
    d.number += 1
    host.console.print(f"[info]Phase {d.number} · judge[/info]")
    job = _Job(duel.JUDGE, conversation, text, _DUEL_SLOTS[duel.JUDGE], dict(state.options))
    result = _run_phase(Phase.JUDGE, [job], client, state, d.held, host=host)
    entry = {"n": d.number, "phase": "JUDGE", "ok": result.ok, "cause": result.cause}
    d.phases.append(entry)
    if result.ok:
        reasons, side = duel.parse_judge(result.turns[duel.JUDGE].answer, order)
        d.judge = {
            "order": list(order),
            "text": result.turns[duel.JUDGE].answer,
            "reasons": reasons,
            "recommendation": side,
        }
    return result


def _run_merge(d, base, client, state, host):
    """MERGE with `base` as the base side. Returns (strengths, text) or None (message shown)."""
    conversation = duel.new_conversation(duel.JUDGE_SYSTEM)
    text = duel.merge_prompt(d.brief, d.texts[base], d.texts[duel.other(base)])
    d.number += 1
    host.console.print(f"[info]Phase {d.number} · merge (base {duel.side_label(base)})[/info]")
    job = _Job(duel.JUDGE, conversation, text, _DUEL_SLOTS[duel.JUDGE], dict(state.options))
    result = _run_phase(Phase.MERGE, [job], client, state, d.held, host=host)
    entry = {"n": d.number, "phase": "MERGE", "ok": result.ok, "cause": result.cause}
    d.phases.append(entry)
    if not result.ok:
        return None
    turn = result.turns[duel.JUDGE]
    parsed = duel.parse_merge(turn.answer)
    if parsed is None:
        entry.update(ok=False, cause="format")
        host.display_error(f"The merge reply had no {duel.REWRITE_MARK} line, so it was discarded.")
        return None
    d.merges.append({"base": base, "strengths": parsed[0], "text": parsed[1]})
    return parsed[0], parsed[1], turn.model


def _show_candidates(d, host):
    for side in duel.SIDES:
        _show_text(f"Side {side.upper()} ({d.kinds[side]})", d.texts[side], host=host)


def _final_choice(d, client, state, host):
    """The final-choice menu. Returns (text, model, choice) or None when the user discards."""
    merged = None  # (base, text, model)
    if d.judge is not None:
        host.console.print(f"[bold]Judge ({duel.order_note(tuple(d.judge['order']))}):[/bold]")
        host.console.print(d.judge["reasons"], markup=False, highlight=False)
        side = d.judge["recommendation"]
        host.display_info(
            f"Recommendation: {duel.side_label(side)}." if side else "The judge made no recommendation."
        )
    while True:
        options = (
            ["a", "b", "m [a|b]"] + ([] if d.judge and d.judge["recommendation"] else ["j"]) + ["s", "q"]
        )
        if merged:
            options.insert(3, "c")
        try:
            line = input(f"Use which? [{' / '.join(options)}]> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            host.console.print()
            return None
        word, _, rest = line.partition(" ")
        rest = rest.strip()
        if word in ("a", "b") and not rest:
            return d.texts[word], d.models[word], word
        if word == "c" and merged:
            return merged[1], merged[2], f"merged:{merged[0]}"
        if word == "q":
            return None
        if word == "s":
            _show_candidates(d, host=host)
            if merged:
                _show_text(f"Merged (base {duel.side_label(merged[0])})", merged[1], host=host)
        elif word == "j":
            if d.judge and d.judge["recommendation"]:
                host.display_info("The judge has already recommended a side.")
                continue
            result = _run_judge(d, client, state, host=host)
            _save_duel_record(d, state, host=host)
            if result.ok and d.judge:
                side = d.judge["recommendation"]
                host.display_info(
                    f"Recommendation: {duel.side_label(side)} ({duel.order_note(tuple(d.judge['order']))})."
                    if side
                    else "The judge made no recommendation."
                )
        elif word == "m":
            base = rest if rest in duel.SIDES else None
            if rest and base is None:
                host.display_error("Usage: m [a|b]")
                continue
            if base is None and d.judge:
                base = d.judge["recommendation"]
            if base is None:
                host.display_error("There is no recommendation to take the base from. Use m a or m b.")
                continue
            outcome = _run_merge(d, base, client, state, host=host)
            _save_duel_record(d, state, host=host)
            if outcome:
                merged = (base, outcome[1], outcome[2])
                _show_text(f"Merged text (base {duel.side_label(base)}); c uses it", outcome[1], host=host)
        else:
            host.display_error("Choose one of: " + ", ".join(options))


def _start_duel(brief_text, auto, client, conversation, state, host):
    """Check the server and build the duel; None (with the reason shown) when it cannot start."""
    client.refresh()
    if client.server_slots is None:
        host.display_error("Could not read the slot count from llama-server.")
        return None
    if client.server_slots < 2:
        host.display_error("Duel needs two slots. Start llama-server with --parallel 2.")
        return None
    if brief_text == '"""':
        brief_text = host.get_multiline_input()
        if brief_text is None:
            host.console.print()
            return None
    if not brief_text.strip():
        return None
    prepared = host.prepare_user_message(brief_text)
    if prepared is None:
        return None
    brief, includes = prepared
    with host.echo_suppressed():
        host.sync_notes(conversation, state, client)
    setting = state.duel
    d = _Duel(brief=brief, includes=includes, started=datetime.now(), auto=auto or setting["auto"])
    for side in duel.SIDES:
        prompt = duel.side_system(conversation.system_prompt, setting[f"stance_{side}"])
        d.sides[side] = duel.new_conversation(prompt, conversation.project_notes)
    d.options = duel.side_options(state.options, setting, client.sampling_defaults())
    d.settings = {k: setting[k] for k in _DUEL_KEYS if k != "auto"}
    d.settings["auto"] = d.auto
    return d


def _show_header(d, client, state, host):
    n_ctx = f"{client.server_n_ctx:,}" if client.server_n_ctx else "unknown"
    host.display_info(
        f"Duel on {client.server_slots} slots (side A on slot 0, side B on slot 1); context {n_ctx} tokens, "
        f"assumed {d.settings['context']} between the sides."
    )
    for side in duel.SIDES:
        extra = [f"{k} {d.options[side][k]}" for k in ("temperature", "seed") if k in d.options[side]]
        suffix = f" ({', '.join(extra)})" if extra else ""
        host.display_info(f"Side {side.upper()}{suffix}: {escape(d.settings[f'stance_{side}'])}")


def _run_duel(cmd, client, conversation, state, host):
    d = _start_duel(cmd.brief, cmd.auto, client, conversation, state, host=host)
    if d is None:
        return
    _show_header(d, client, state, host=host)
    state.duel_last = _duel_record(d)
    _save_duel_record(d, state, host=host)

    after = None
    d.ended = "normal"
    while True:
        # Pauses sit between phases only, so not after the last one.
        if after is not None and not d.auto and duel.next_phase(after, d.ledger) is not None:
            if _duel_pause(d, after, host=host) == "stop":
                d.ended = "stopped"
                break
        step = duel.next_phase(after, d.ledger)
        if step is None:
            break
        phase, who, skipped = step
        if skipped:
            host.display_info(duel.skip_note(skipped))
            if Phase.REVISE in skipped:
                d.ledger.close()
        if phase == Phase.JUDGE:
            result = _run_judge(d, client, state, host=host)
        else:
            _announce_phase(d, phase, who, host=host)
            result = _run_phase(phase, _duel_jobs(d, phase, who, state), client, state, d.held, host=host)
            d.phases.append({"n": d.number, "phase": phase.name, "ok": result.ok, "cause": result.cause})
            if result.ok:
                _apply_phase(d, phase, result, state, host=host)
                if phase == Phase.REVISE:
                    d.ledger.close()
        if not result.ok:
            d.ended = f"abnormal:{phase.name}:{result.cause}"
            if phase == Phase.DRAFT:
                host.display_error(
                    "The drafts could not be completed, so the duel ends and nothing is added."
                )
            else:
                host.display_info("This phase was discarded; the duel goes no further.")
            break
        after = phase
        _save_duel_record(d, state, host=host)

    if len(d.texts) < 2:
        d.choice = "none"
        _save_duel_record(d, state, host=host)
        return
    _save_duel_record(d, state, host=host)
    chosen = _final_choice(d, client, state, host=host)
    d.choice = chosen[2] if chosen else "none"
    _save_duel_record(d, state, host=host)
    if chosen is None:
        host.display_info("Discarded; nothing was added to the conversation.")
        return
    text, model, _ = chosen
    conversation.add_user(d.brief, includes=d.includes)
    conversation.add_assistant(text, model=model or state.model)
    with host.echo_suppressed():
        host.auto_save(conversation, state)
    host.display_info("Added to the conversation.")
