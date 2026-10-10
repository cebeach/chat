#!/usr/bin/env python3
"""AI Chat — a terminal chat application powered by the llama.cpp server."""

import argparse
import copy
import math
from pathlib import Path
import queue
import random
import re
import shlex
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime

from requests.exceptions import ConnectionError, HTTPError, RequestException
from rich.markup import escape

import duel
import projects
from config import apply_project, load_config
from conversation import Conversation, split_think
from duel import Phase
import tokens
from llama_client import LlamaClient, ThinkTagsError
from ui import (
    console,
    display_assistant_stream,
    display_cat_conversation,
    display_config,
    describe_think_tags,
    display_context_warning,
    display_conversation_info,
    display_conversations,
    display_error,
    display_included_files,
    display_ledger,
    display_info,
    display_notes_heavy,
    display_options,
    display_projects,
    display_prompt_size,
    format_option,
    format_server_option,
    option_rows,
    display_stats,
    display_truncated_warning,
    echo_suppressed,
    get_multiline_input,
    get_user_input,
    init_readline,
    print_help,
    print_welcome,
    save_readline_history,
)


@dataclass
class State:
    model: str
    config: dict
    context_length: int | None
    options: dict = field(default_factory=dict)
    show_stats: bool = True
    retry_text: str | None = None
    auto_save_name: str = ""
    pending_includes: list = field(default_factory=list)
    # Active thinking tags as (start, end, source), or None. See docs/thinking-tags.md.
    think_tags: tuple | None = None
    # The active project (None: no project; behavior is as without the feature).
    project: str | None = None
    project_dir: Path | None = None
    # The project's system prompt and where it came from, kept so /load can restore them.
    project_prompt: str = ""
    project_source_file: str | None = None
    # The config.toml values; state.config is these plus the active project's settings.
    global_config: dict = field(default_factory=dict)
    # Digest of the project.md text the conversation currently carries.
    notes_digest: str = ""
    # The session copy of the /duel settings (see _initial_duel) and the last duel's record.
    duel: dict = field(default_factory=lambda: _initial_duel({}, quiet=True))
    duel_last: dict | None = None


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with the model served by llama-server")
    parser.add_argument(
        "--url",
        default=None,
        help="Override llama-server URL",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Start in this project (see /project)",
    )
    return parser.parse_args()


# The sampling options a session can override: the type /set parses and what the server
# accepts. The server clamps most out-of-range values silently and answers a few with
# HTTP 400, so the app checks first and refuses what it would otherwise get wrong.
_OPTION_KEYS = {
    "seed": int,
    "temperature": float,
    "top_p": float,
    "min_p": float,
    "repeat_penalty": float,
    "n_predict": int,
}
_OPTION_RULES = {
    "seed": (lambda v: -1 <= v <= 0xFFFFFFFF, "an integer from -1 (random) to 4294967295"),
    "temperature": (lambda v: v >= 0, "a number >= 0 (0 is greedy)"),
    "top_p": (lambda v: 0 <= v <= 1, "a number from 0 to 1 (1 disables it)"),
    "min_p": (lambda v: 0 <= v <= 1, "a number from 0 to 1 (0 disables it)"),
    "repeat_penalty": (lambda v: v > 0, "a number > 0 (1 disables it)"),
    "n_predict": (
        lambda v: v >= 1,
        "an integer >= 1 (`/set n_predict default` returns to the server's limit)",
    ),
}


def _validated(key, value):
    """`value` as option `key`'s type, or ValueError saying what the option accepts.

    A bool is refused (it is an int in Python), as is a float for an integer option
    (TOML `n_predict = 5.5`); an int is fine for a float option (`temperature = 1`).
    """
    kind = _OPTION_KEYS[key]
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok or (kind is int and not isinstance(value, int)) or not math.isfinite(value):
        raise ValueError(f"must be {kind.__name__}")
    value = kind(value)
    accepts, description = _OPTION_RULES[key]
    if not accepts(value):
        raise ValueError(f"must be {description}")
    return value


def _initial_options(config, source="config.toml", quiet=False):
    """The session's overrides from `config`: only the keys that are set and valid.

    source names the file in an error message; quiet suppresses it.

    State.options holds nothing else. /props reports the server's launch defaults but
    never what a request sent, so the app must remember what it sends; every other
    option is left out of the request and the server's own default applies.
    """
    options = {}
    for key in _OPTION_KEYS:
        if config.get(key) is None:
            continue
        try:
            options[key] = _validated(key, config[key])
        except ValueError as exc:
            if not quiet:
                display_error(f"{source}: {key} {exc}; ignored.")
    return options


def _initial_duel(config, source="config.toml", quiet=False):
    """The session's /duel settings from `config`; a bad value warns and falls back to its default.

    Keys are the config names without the `duel_` prefix. `auto` has no config key: it
    starts off in every session. A seed of -1 ("random") counts as unset. quiet suppresses
    the warnings (a project switch calls this again for the same file).
    """
    settings = {"auto": False}

    def bad(key, why):
        if not quiet:
            display_error(f"{source}: duel_{key} {why}; ignored.")

    axes = config.get("duel_axes", duel.DEFAULT_AXES)
    if isinstance(axes, list) and axes and all(isinstance(a, str) and a.strip() for a in axes):
        settings["axes"] = [a.strip() for a in axes]
    else:
        bad("axes", "must be a non-empty list of non-empty strings")
        settings["axes"] = list(duel.DEFAULT_AXES)

    for side, default in (("a", duel.DEFAULT_STANCE_A), ("b", duel.DEFAULT_STANCE_B)):
        value = config.get(f"duel_stance_{side}", default)
        if isinstance(value, str) and value.strip():
            settings[f"stance_{side}"] = value.strip()
        else:
            bad(f"stance_{side}", "must be a non-empty string")
            settings[f"stance_{side}"] = default

    context = config.get("duel_context", "shared")
    if context in ("shared", "split"):
        settings["context"] = context
    else:
        bad("context", "must be 'shared' or 'split'")
        settings["context"] = "shared"

    for side in duel.SIDES:
        for option in ("temperature", "seed"):
            key = f"{option}_{side}"
            settings[key] = None
            value = config.get(f"duel_{key}")
            if value is None:
                continue
            try:
                value = _validated(option, value)
            except ValueError as exc:
                bad(key, str(exc))
                continue
            settings[key] = None if option == "seed" and value == -1 else value
    return settings


# Bounds the memory a read and the /tokenize request body can take. It is not the
# context limit: whether a prompt fits the window is decided by _fits_window().
_READ_FILE_MAX_BYTES = 8 * 1024 * 1024


def _read_file(path: str) -> tuple[bool, str]:
    """Read a UTF‑8 text file of at most 8 MB.

    Returns (True, content) on success or (False, error_msg) on failure.
    """
    # Resolve the path relative to cwd, expand user (~)
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return False, "File not found."

    if resolved.stat().st_size > _READ_FILE_MAX_BYTES:
        return False, "File too large to read."

    try:
        content = resolved.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read file: {e}"
    return True, content


# `@@<path>` splices a file into a prompt; `\@@<` is a literal `@@<`.
_INCLUDE_OPEN = re.compile(r"\\@@<|@@<")
_INCLUDE_PATH = re.compile(r"[^>\n]*")


def _trailing_newlines(s):
    return len(s) - len(s.rstrip("\n"))


def _leading_newlines(s):
    return len(s) - len(s.lstrip("\n"))


def build_message(segments):
    """Read the files in segments and splice them into one flat message.

    segments is a list of ("text", str) and ("file", typed_path, shown_name)
    tuples. Returns (flat_text, includes, errors). Any error (a file that is
    missing, too large or unreadable, or a message with nothing left to send)
    leaves flat_text empty: the caller must not send it.

    A file's leading and trailing newlines are dropped and it is set off from
    adjacent text by a blank line. Newlines typed around it are never removed,
    and none are added at the very start or end of the message. A file that is
    only whitespace counts as empty: it adds no text but still separates its
    neighbours.
    """
    includes, errors, files = [], [], []
    for seg in segments:
        if seg[0] != "file":
            continue
        _, typed, shown = seg
        ok, content = _read_file(typed)
        if not ok:
            errors.append(f"{shown}: {content}")
            continue
        path = Path(typed).expanduser().resolve()
        include = {"typed": typed, "path": str(path), "bytes": path.stat().st_size}
        if not content.strip():
            include["empty"] = True
        includes.append(include)
        files.append(content)
    if errors:
        return "", [], errors

    out = ""
    after_file = False  # the last piece was a file: the next real text needs a blank line
    file_no = 0
    for i, seg in enumerate(segments):
        if seg[0] == "text":
            piece = seg[1]
            # The token and the horizontal whitespace touching it are replaced.
            if i > 0 and segments[i - 1][0] == "file":
                piece = piece.lstrip(" \t")
            if i + 1 < len(segments) and segments[i + 1][0] == "file":
                piece = piece.rstrip(" \t")
            if not piece:
                continue
            if not piece.strip():  # only newlines: keep them, add no separator
                out += piece
                continue
            if after_file and out:
                out += "\n" * max(0, 2 - _trailing_newlines(out) - _leading_newlines(piece))
            out += piece
            after_file = False
        else:
            content = files[file_no]
            file_no += 1
            if content.strip():
                if out:
                    out += "\n" * max(0, 2 - _trailing_newlines(out))
                out += content.strip("\r\n")
            after_file = True

    if not out.strip():
        names = ", ".join(seg[2] for seg in segments if seg[0] == "file")
        return "", [], [f"Nothing to send: {names} is empty."]
    return out, includes, []


def expand_includes(text):
    """Replace each `@@<path>` in text with the file's contents.

    Returns (flat_text, includes, errors); see build_message. The model gets
    only flat_text. `\\@@<` writes a literal `@@<`, and a `@@<` with no closing
    `>` on its line is an error rather than a placeholder sent as typed.
    Included files are not themselves expanded.
    """
    if "@@<" not in text:
        return text, [], []
    segments, errors, literal = [], [], []
    pos = 0
    while True:
        m = _INCLUDE_OPEN.search(text, pos)
        if m is None:
            break
        literal.append(text[pos : m.start()])
        if m.group() != "@@<":  # \@@< : keep the literal @@<
            literal.append("@@<")
            pos = m.end()
            continue
        body = _INCLUDE_PATH.match(text, m.end())
        end = body.end()
        if end < len(text) and text[end] == ">":
            typed = body.group().strip()
            if typed:
                segments.append(("text", "".join(literal)))
                literal = []
                segments.append(("file", typed, f"@@<{typed}>"))
            else:
                errors.append("@@<>: no file name between the brackets.")
            pos = end + 1
        else:
            errors.append(f"Unterminated include: @@<{body.group()} (write \\@@< for a literal @@<)")
            pos = end
    literal.append(text[pos:])
    segments.append(("text", "".join(literal)))
    if errors:
        # Still report any missing files, so one fix pass finds everything.
        _, _, more = build_message(segments)
        return "", [], errors + [e for e in more if not e.startswith("Nothing to send")]
    return build_message(segments)


def _prepare_user_message(text):
    """Expand `@@<path>` includes in a typed message.

    Returns (flat_text, includes), or None when the message must not be sent:
    the errors are printed and nothing reaches the conversation or the model.
    """
    flat, includes, errors = expand_includes(text)
    if errors:
        for error in errors:
            display_error(escape(error))
        return None
    display_included_files(includes)
    return flat, includes


def _server_option(client, key):
    """The server's launch value for option `key`, as text; "unavailable" if /props cannot be read."""
    server = client.sampling_defaults()
    return format_server_option(key, server.get(key) if server else None)


def _handle_set(args, state, client):
    """Handle the /set command: list, query, or modify a model option.

    The server's values are read from /props on every call. An option the user has not
    set is absent from state.options, so nothing is sent for it.
    """
    if not args:
        display_options(option_rows(_OPTION_KEYS, client.sampling_defaults(), state.options))
        return
    parts = args.strip().split(None, 1)
    key = parts[0]
    if key not in _OPTION_KEYS:
        display_error(f"Unknown option: {key}. Available: {', '.join(_OPTION_KEYS)}")
    elif len(parts) < 2:
        server = _server_option(client, key)
        if key in state.options:
            shown = format_option(key, state.options[key], reported=False)
            display_info(f"{key}: {shown} (set for this session; the server's is {server})")
        else:
            display_info(f"{key}: {server} (server)")
    elif parts[1] == "default":
        state.options.pop(key, None)
        display_info(f"{key} reset; the server's value ({_server_option(client, key)}) applies.")
    else:
        try:
            value = _OPTION_KEYS[key](parts[1])
        except ValueError:
            display_error(f"{key} must be {_OPTION_KEYS[key].__name__} (or 'default').")
            return
        try:
            state.options[key] = _validated(key, value)
        except ValueError as exc:
            display_error(f"{key} {exc}.")
            return
        display_info(f"{key} set to {format_option(key, state.options[key], reported=False)}.")


# Boolean settings that /config can toggle for the session, with their defaults.
_CONFIG_TOGGLES = {"save_thinking": True, "context_check": True}


def _handle_config(args, state, client=None):
    """Handle /config: show settings, or toggle/set a boolean setting."""
    if client is not None:
        # Re-read the server first so every row (and the toggle warning below)
        # is current even if the server was restarted since the last turn.
        client.refresh()
        _sync_server_info(state, client.server_model, client.server_n_ctx)
        _update_think_tags(state, client.think_tags)
    if not args.strip():
        server = client.sampling_defaults() if client is not None else None
        rows = option_rows(_OPTION_KEYS, server, state.options)
        shown = {**state.config, "conversations_dir": _conversations_dir(state)}
        display_config(shown, state.model, rows, state.think_tags, project=state.project)
        return
    parts = args.split()
    key = parts[0]
    if key not in _CONFIG_TOGGLES:
        display_error(f"Unknown setting: {key}. Available: {', '.join(sorted(_CONFIG_TOGGLES))}")
        return
    if len(parts) == 1:
        state.config[key] = not state.config.get(key, _CONFIG_TOGGLES[key])
    elif len(parts) == 2 and parts[1] in ("on", "off"):
        state.config[key] = parts[1] == "on"
    else:
        display_error(f"Usage: /config {key} on|off (no argument toggles)")
        return
    msg = f"{key}: {'on' if state.config[key] else 'off'}"
    if key == "save_thinking" and not state.config[key] and state.think_tags is None:
        # Do not let a no-op look like it works.
        msg += " (no thinking tags known for the current model, so its replies cannot be separated and are saved whole)"
    display_info(msg)


def _omit_thinking(state):
    """True when saved files should leave out the thinking field."""
    return not state.config.get("save_thinking", _CONFIG_TOGGLES["save_thinking"])


def _think_override(config):
    """The (start, end) override from the config file, or None unless both are set."""
    start, end = config.get("think_start", ""), config.get("think_end", "")
    return (start, end) if start and end else None


def _update_think_tags(state, tags, announce=True):
    """Make `tags` the active thinking tags.

    When the pair changes between turns (including to None) one info line says
    so, because a model swap while save_thinking is already off produces no
    other signal. The tags are Rich-escaped by describe_think_tags.
    """
    changed = (state.think_tags or (None,))[:2] != (tags or (None,))[:2]
    state.think_tags = tags
    if changed and announce:
        text = describe_think_tags(tags)
        display_info(
            f"thinking tags: {text}" if tags else "thinking tags: none detected for the current model"
        )


def _store_reply(conversation, response, chat_stream, state):
    """Add the streamed reply to the conversation, thinking split off its content.

    Split once, with the tags detected for this very turn, so a saved file never
    needs them again. Without tags the reply is stored whole.
    """
    thinking, answer = split_think(response, chat_stream.think_tags)
    conversation.add_assistant(answer, model=_reply_model(chat_stream, state), thinking=thinking)


def _sync_server_info(state, model, n_ctx, announce=True):
    """Make state.model and state.context_length follow what the server reports.

    A value of None (not reported, or /props unreadable) leaves the previous one.
    When the model changes (for example the server was restarted with another
    one) a single info line says so; the names are Rich-escaped.
    """
    old = state.model
    if n_ctx is not None:
        state.context_length = n_ctx
    if model and model != old:
        state.model = model
        if announce:
            ctx = f" (context {state.context_length:,} tokens)" if state.context_length else ""
            display_info(f"model: {escape(old)} → {escape(model)}{ctx}")


def _reply_model(chat_stream, state):
    """The model that produced a reply: the server's own statement in the final
    chunk, else (interrupted stream, or no final chunk) the model refreshed just
    before sending."""
    return chat_stream.model or state.model


def _think_end(state):
    """The tag that closes the active model's thinking block, or None."""
    return state.think_tags[1] if state.think_tags else None


def _fits_window(client, messages, state, outcome="Not sent"):
    """Price `messages` against the context window before anything is sent.

    Returns (ok, priced). ok is False only when the prompt cannot fit (the error is
    printed) or Ctrl-C cancelled the counting; the caller must not send. A prompt over 80% of the window is allowed with
    a warning. priced is (needed, n_ctx), or None when nothing was counted: context_check
    is off, the window is unknown, or the server could not be asked. In that last case
    nothing is refused here, because the call that follows meets the same problem and
    reports it (and, in chat.py's turn, takes the unanswered message back).
    """
    if not state.config.get("context_check", True):
        return True, None
    try:
        needed, n_ctx = client.check_fit(messages, model=state.model)
    except KeyboardInterrupt:
        console.print()
        display_info("Cancelled.")
        return False, None
    except (RequestException, ValueError, KeyError):
        return True, None
    if not n_ctx:
        return True, None
    # A reply capped with /set n_predict needs no more room than that.
    reserve = max(state.config.get("reserve_output_tokens", 0), state.options.get("n_predict", 0))
    if not tokens.fits(needed, n_ctx, reserve):
        display_error(tokens.refusal(needed, n_ctx, reserve, outcome))
        return False, (needed, n_ctx)
    if tokens.near_limit(needed, n_ctx):
        display_context_warning(needed, n_ctx)
    return True, (needed, n_ctx)


def _conversations_dir(state):
    """Where conversations are saved and loaded: the project's directory, else the global one."""
    if state.project_dir is not None:
        return str(projects.conversations_path(state.project_dir))
    return state.config["conversations_dir"]


def _projects_dir(state):
    return state.config["projects_dir"]


def _system_prompt_fits(text, client, conversation, state):
    """True when `text` as the system prompt, with the stored messages, fits the window.

    Runs with the keyboard echo off like a model turn, because counting a large prompt
    takes a moment. When it does not fit the error is printed and the old prompt stays.
    """
    if not text.strip():
        return True
    probe = copy.copy(conversation)  # same messages and notes, the new prompt
    probe.system_prompt = text
    candidate = probe.get_messages()
    with echo_suppressed():
        ok, _ = _fits_window(client, candidate, state, outcome="System prompt not changed")
    return ok


# ---- projects (see docs/projects.md) ----

# Reserved on top of the notes-sized reply for a thinking model's reasoning in /remember.
_REMEMBER_THINK_ALLOWANCE = 2000


def _read_scope(name, state):
    """Everything a project switch needs, read up front; None (error shown) on any failure.

    Reading first means a bad project.toml or an unreadable file leaves the session untouched.
    """
    try:
        path = projects.project_dir(_projects_dir(state), name)
    except ValueError as exc:
        display_error(str(exc))
        return None
    if not path.is_dir():
        display_error(f"No project named '{name}'. Create it with /project new {name}.")
        return None
    try:
        settings, ignored = projects.load_settings(path)
        prompt = projects.read_system_prompt(path)
        notes = projects.read_notes(path)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        display_error(f"Cannot read project '{name}': {exc}")
        return None
    return {
        "name": name,
        "dir": path,
        "settings": settings,
        "ignored": ignored,
        "prompt": prompt,
        "notes": notes,
    }


def _reset_session(state, settings):
    """Settings back to config.toml plus a project's `settings`; the toggles with no key to their defaults."""
    state.config = apply_project(state.global_config, settings)
    state.options = _initial_options(state.global_config, quiet=True)
    state.options.update(_initial_options(settings, source="project.toml"))
    state.show_stats = True
    state.duel = _initial_duel(state.global_config, quiet=True)
    state.duel_last = None


def _apply_scope(conversation, state, scope):
    """Make `scope` (from _read_scope, or None for no project) the active one. Cannot fail."""
    settings = scope["settings"] if scope else {}
    _reset_session(state, settings)
    prompt = scope["prompt"] if scope else ""
    state.project = scope["name"] if scope else None
    state.project_dir = scope["dir"] if scope else None
    state.project_prompt = prompt or state.global_config.get("system_prompt", "")
    state.project_source_file = str(scope["dir"] / "system.md") if prompt else None
    conversation.system_prompt = state.project_prompt
    conversation.source_file = state.project_source_file
    conversation.project_notes = scope["notes"] if scope else ""
    state.notes_digest = projects.notes_digest(conversation.project_notes)


def _announce_scope(scope, client, conversation):
    """The lines that follow a switch or reload: ignored keys and a heavy-notes warning."""
    if not scope:
        return
    for key in scope["ignored"]:
        display_info(f"project.toml: {key} is ignored; put the system prompt in system.md.")
    notes = conversation.project_notes
    if not notes:
        return
    try:
        count = client.count_tokens(notes, add_special=False)
        n_ctx = client.server_n_ctx
    except (RequestException, ValueError, KeyError):
        return
    if isinstance(count, int) and isinstance(n_ctx, int) and tokens.notes_heavy(count, n_ctx):
        display_notes_heavy(count, n_ctx)


def _switch_project(name, client, conversation, state):
    """Switch to project `name`, or to no project when it is None. Returns True if switched.

    Everything fallible is read first. The current conversation is then saved (even with
    auto_save off) and, only if that worked, replaced by an empty one.
    """
    scope = _read_scope(name, state) if name else None
    if name and scope is None:
        return False
    if not _auto_save(conversation, state, force=True):
        display_error("Not switched: the current conversation could not be saved.")
        return False
    _apply_scope(conversation, state, scope)
    conversation.messages.clear()
    state.auto_save_name = "auto_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    state.retry_text = None
    state.pending_includes = []
    if scope:
        display_info(f"Project: {escape(name)} ({scope['dir']})")
    else:
        display_info("No project selected.")
    display_info("Session settings were reset to the configured ones (/set, /config, /stats).")
    _announce_scope(scope, client, conversation)
    return True


def _reload_project(client, conversation, state):
    """Re-read system.md, project.md and project.toml; the conversation is kept."""
    scope = _read_scope(state.project, state)
    if scope is None:
        return
    _apply_scope(conversation, state, scope)
    display_info(f"Project reloaded: {escape(state.project)}")
    display_info("Session settings were reset to the configured ones (/set, /config, /stats).")
    _announce_scope(scope, client, conversation)


def _sync_notes(conversation, state, client):
    """Pick up an outside edit of project.md; the conversation carries the file's current text.

    Called before each send, so the pre-send check prices the notes that will be sent. An
    edit invalidates the server's cached prompt prefix, so one line says so.
    """
    if state.project_dir is None:
        return
    try:
        text = projects.read_notes(state.project_dir)
    except (OSError, UnicodeDecodeError) as exc:
        display_error(f"Cannot read project.md: {exc}; using the notes loaded earlier.")
        return
    digest = projects.notes_digest(text)
    if digest == state.notes_digest:
        return
    conversation.project_notes = text
    state.notes_digest = digest
    try:
        count = client.count_tokens(text, add_special=False) if text else 0
    except (RequestException, ValueError, KeyError):
        count = None
    if not isinstance(count, int):
        display_info("project.md changed.")
        return
    display_info(f"project.md changed: {count:,} tokens")
    n_ctx = state.context_length
    if n_ctx and tokens.notes_heavy(count, n_ctx):
        display_notes_heavy(count, n_ctx)


def _handle_project(args, client, conversation, state):
    parts = args.split(None, 1)
    sub = parts[0] if parts else "info"
    name = parts[1].strip() if len(parts) > 1 else ""
    if sub == "info":
        _project_info(client, conversation, state)
    elif sub == "list":
        display_projects(projects.list_projects(_projects_dir(state)), state.project)
    elif sub == "new":
        if not name:
            display_error("Usage: /project new <name>")
            return
        try:
            path = projects.create_project(_projects_dir(state), name)
        except (ValueError, FileExistsError, OSError) as exc:
            display_error(str(exc))
            return
        display_info(f"Project created: {path}. Use it with /project use {name}.")
    elif sub == "use":
        if not name:
            display_error("Usage: /project use <name>")
            return
        _switch_project(name, client, conversation, state)
    elif sub == "reload":
        if not state.project:
            display_error("No project selected. /project use <name>.")
        else:
            _reload_project(client, conversation, state)
    elif sub == "leave":
        if not state.project:
            display_error("No project selected.")
        else:
            _switch_project(None, client, conversation, state)
    else:
        display_error("Usage: /project [info | list | new <name> | use <name> | reload | leave]")


def _project_info(client, conversation, state):
    if not state.project:
        display_info("No project selected. /project list shows the projects; /project use <name> picks one.")
        return
    display_info(f"Project: {escape(state.project)} ({state.project_dir})")
    notes = conversation.project_notes
    try:
        count = client.count_tokens(notes, add_special=False) if notes else 0
        size = f"{count:,} tokens"
    except (RequestException, ValueError, KeyError):
        size = f"{len(notes):,} characters"
    display_info(f"project.md: {size}")
    source = "system.md" if state.project_source_file else "global system prompt"
    display_info(f"System prompt: {source}")
    try:
        settings, _ = projects.load_settings(state.project_dir)
    except ValueError:
        settings = {}
    if settings:
        shown = ", ".join(f"{k} = {v}" for k, v in settings.items())
        display_info(f"project.toml: {escape(shown)}")


def _remember_room(messages, notes, note, client, state):
    """True when the merge prompt and a reply the size of the notes fit the window.

    Not _fits_window: this check applies even with context_check off, because a reply cut
    off mid-file must never reach the approval step, and reserve_output_tokens is for chat
    replies. A thinking allowance is reserved on top when there is room for it. Prints
    why and returns False when /remember cannot go ahead.
    """
    outcome = "Not remembered"
    try:
        needed, n_ctx = client.check_fit(messages, model=state.model)
        reply = client.count_tokens(f"{notes}\n{note}", add_special=False)
    except KeyboardInterrupt:
        console.print()
        display_info("Cancelled.")
        return False
    except (RequestException, ValueError, KeyError) as exc:
        display_error(f"{outcome}: could not check the context window ({exc}).")
        return False
    if not n_ctx:
        display_error(f"{outcome}: the context window size is unknown.")
        return False
    if tokens.fits(needed, n_ctx, reply + _REMEMBER_THINK_ALLOWANCE):
        return True
    if tokens.fits(needed, n_ctx, reply):
        return True  # no room for the allowance; a model that thinks long is caught by the cut-off check
    display_error(tokens.refusal(needed, n_ctx, reply, outcome))
    return False


def _handle_remember(args, client, conversation, state):
    """/remember <text>: ask the model to merge the note into project.md, then show the diff.

    The conversation is not touched. Nothing is written unless the user answers y, the
    reply was complete and project.md is unchanged since the model saw it.
    """
    if not state.project_dir:
        display_error("No project selected. /project use <name>.")
        return
    note = args.strip()
    if not note:
        display_error("Usage: /remember <text>")
        return
    with echo_suppressed():
        _sync_notes(conversation, state, client)
        notes = conversation.project_notes
        messages = projects.remember_messages(notes, note)
        if not _remember_room(messages, notes, note, client, state):
            return
        try:
            text, cut_off, think_tags = client.complete(
                state.model, messages, options={"temperature": 0}, refreshed=True
            )
        except KeyboardInterrupt:
            console.print()
            display_info("Cancelled. project.md is unchanged.")
            return
        except ConnectionError:
            display_error("Lost connection to llama-server. Is it still running?")
            return
        except (ThinkTagsError, HTTPError) as exc:
            display_error(f"Not remembered: {exc}")
            return
    if cut_off:
        display_error("Not remembered: the model's reply was cut off, so project.md is unchanged.")
        return
    new = projects.clean_reply(text, think_tags)
    if not new:
        display_error("Not remembered: the model's reply was empty. project.md is unchanged.")
        return
    if projects.shrank_too_much(notes, new):
        display_error("Not remembered: the reply is under half the size of project.md, so it was rejected.")
        return
    diff = projects.diff_text(notes, new)
    if not diff:
        display_info("No change: project.md already says that.")
        return
    console.print(diff, markup=False, highlight=False)
    try:
        answer = input("Write this to project.md? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        console.print()
        answer = ""
    if answer.strip().lower() != "y":
        display_info("project.md unchanged.")
        return
    try:
        current = projects.read_notes(state.project_dir)
        if projects.notes_digest(current) != state.notes_digest:
            display_error(
                "project.md changed while you were deciding, so nothing was written. Run /remember again."
            )
            return
        projects.write_notes(state.project_dir, new)
    except (OSError, UnicodeDecodeError) as exc:
        display_error(f"Could not write project.md: {exc}")
        return
    conversation.project_notes = new
    state.notes_digest = projects.notes_digest(new)
    display_info("project.md updated.")


# ---- /duel (see docs/duel.md) ----


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


def _price_phase(phase, jobs, client, state, held):
    """Price every participant, then the pool. Returns (priced, None) or (None, cause).

    Each participant is priced alone by _fits_window, which prints its own refusal; only if
    all pass is the joint rule applied. `priced` maps a job key to (needed, n_ctx) or None
    (pricing unavailable: nothing is refused, the request that follows meets any real problem).
    """
    priced = {}
    for job in jobs:
        candidate = job.conversation.get_messages() + [{"role": "user", "content": job.text}]
        outcome = f"Not sent ({duel.side_label(job.key)}, {phase.value})"
        ok, result = _fits_window(client, candidate, state, outcome=outcome)
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
            display_error(tokens.refusal_together(needed, idle, n_ctx, reserve, outcome))
            return None, "context"
    return priced, None


def _run_phase(phase, jobs, client, state, held):
    """Run one phase: every job takes one turn, concurrently when there are several.

    Returns a _PhaseResult. A phase that ends abnormally is discarded as a whole: each
    participant's conversation loses what the phase appended, and the caller must not advance.
    `held` maps a slot to the tokens it still holds; it is updated here. Everything runs with
    the keyboard echo off; threads exist only inside this call.
    """
    with echo_suppressed():
        priced, cause = _price_phase(phase, jobs, client, state, held)
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
                    _sync_server_info(state, stream.server_model, stream.server_n_ctx)
                    _update_think_tags(state, stream.think_tags)
                    display_assistant_stream(
                        _drain(part),
                        think_end=stream.think_tags[1] if stream.think_tags else None,
                        label=f"{duel.side_label(part.job.key)}:",
                        interrupted=interrupted,
                    )
                if interrupted.is_set() or part.error is not None:
                    break
                if state.show_stats:
                    display_stats(part.stats)
        except KeyboardInterrupt:
            interrupted.set()

        complete = all(p.done.is_set() and p.error is None for p in parts)
        cause = None
        if interrupted.is_set() and not complete:
            console.print()
            display_info(f"{phase.value} interrupted.")
            cause = "interrupted"
        elif any(p.error is not None for p in parts):
            for part in parts:
                if part.error is not None:
                    display_error(_duel_error_text(part.error, duel.side_label(part.job.key)))
            cause = "server_error"
        elif any(p.cut_off for p in parts):
            for part in parts:
                if part.cut_off:
                    display_error(
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
                        display_error(
                            f"{duel.side_label(part.job.key)} gave no answer in {phase.value}, "
                            "so the phase is discarded."
                        )
                        cause = "empty_answer"
                        break
                    answer = duel.NO_ANSWER
                turns[part.job.key] = _Turn(answer, thinking, part.stats, _reply_model(part.stream, state))

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


def _duel_usage(client, state):
    """/duel alone: usage, the settings and what the server offers."""
    client.refresh()
    slots = client.server_slots if client.server_slots is not None else "unknown"
    n_ctx = f"{client.server_n_ctx:,}" if client.server_n_ctx else "unknown"
    setting = state.duel
    display_info('Usage: /duel [--auto] <brief>   (""" for a multiline brief; -- ends flag parsing)')
    display_info("       /duel set <key> <value>   keys: " + ", ".join(_DUEL_KEYS) + "; 'default' resets")
    display_info("       /duel ledger              show the last duel's issues")
    display_info(f"Server slots: {slots} (a duel needs 2: llama-server --parallel 2); context n_ctx: {n_ctx}")
    display_info(
        f"Context assumption: {setting['context']} (shared: both sides share one pool of n_ctx; split: each has n_ctx)"
    )
    display_info(f"Axes: {', '.join(setting['axes'])}; auto pauses: {'off' if setting['auto'] else 'on'}")
    for side in duel.SIDES:
        extra = [
            f"{k.split('_')[0]} {setting[k]}"
            for k in (f"temperature_{side}", f"seed_{side}")
            if setting[k] is not None
        ]
        suffix = f" ({', '.join(extra)})" if extra else ""
        display_info(f"Side {side.upper()}{suffix}: {escape(setting[f'stance_{side}'])}")


def _duel_set(key, value, state):
    """/duel set <key> <value>: change a setting for this session ('default' resets it)."""
    if key not in _DUEL_KEYS:
        display_error(f"Unknown duel setting: {key}. Available: {', '.join(_DUEL_KEYS)}")
        return
    value = value.strip()
    if not value:
        display_info(f"{key}: {state.duel[key]}")
        return
    defaults = _initial_duel(state.global_config, quiet=True)
    if value == "default":
        state.duel[key] = defaults[key]
        display_info(f"{key} reset to {defaults[key]}.")
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
                number = _OPTION_KEYS[option](value)
            except ValueError:
                raise ValueError(f"{key} must be {_OPTION_KEYS[option].__name__} (or 'default')") from None
            new = _validated(option, number)
            if option == "seed" and new == -1:
                new = None
    except ValueError as exc:
        display_error(f"{key}: {exc}.")
        return
    state.duel[key] = new
    display_info(f"{key} set to {new}.")


def _duel_ledger(state):
    record = state.duel_last
    if record is None:
        display_info("No duel yet.")
        return
    display_ledger([duel.Issue(**i) for i in record["ledger"]["issues"]])


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


def _save_duel_record(d, state):
    """Write (the first time) or rewrite the record; a failure warns once and the duel goes on."""
    record = _duel_record(d)
    try:
        if d.record_path is None:
            directory = Path(_conversations_dir(state)).expanduser() / "duels"
            d.record_path = duel.write_record(directory, record, d.started)
        else:
            duel.rewrite_record(d.record_path, record)
    except OSError as exc:
        if not d.record_warned:
            display_error(f"Could not write the duel record: {exc}")
            d.record_warned = True


def _show_text(title, text):
    console.print(f"[bold]{escape(title)}[/bold]")
    console.print(text, markup=False, highlight=False)


def _duel_pause(d, after):
    """The duel> prompt between two phases. Returns "stop" or None (continue)."""
    step = duel.next_phase(after, d.ledger)
    if step is not None:
        display_info(f"Next: {step[0].value}. Enter to continue, ? for help.")
    while True:
        try:
            line = input("duel> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
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
                display_error(str(exc))
            else:
                target = (
                    "both sides" if steer.group(1).lower() == "both" else f"Side {steer.group(1).upper()}"
                )
                display_info(f"Queued for {target}'s revision.")
        elif override:
            try:
                (d.ledger.override_accept if override.group(1) == "accept" else d.ledger.override_drop)(
                    int(override.group(2))
                )
            except duel.Refused as exc:
                display_error(str(exc))
            else:
                display_info(
                    f"Issue {override.group(2)} {'accepted' if override.group(1) == 'accept' else 'dropped'}."
                )
        elif low == "ledger":
            display_ledger(d.ledger.issues)
        elif show:
            side = show.group(1)
            _show_text(f"Side {side.upper()} ({d.kinds.get(side, 'no text yet')})", d.texts.get(side, ""))
        elif low == "auto":
            d.auto = True
            display_info("No more pauses until the final choice.")
            return None
        elif low == "stop":
            return "stop"
        elif low == "?":
            console.print(_DUEL_PAUSE_HELP, markup=False, highlight=False)
        else:
            display_error("Unknown command. ? for help.")


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


def _announce_phase(d, phase, who):
    d.number += 1
    who_note = ""
    if len(who) == 1 and who[0] in duel.SIDES:
        who_note = f" ({duel.side_label(who[0])})"
    console.print(f"[info]Phase {d.number} · {phase.value}{who_note}[/info]")


def _apply_phase(d, phase, result, state):
    """Fold a finished phase into the ledger and texts."""
    for side, turn in result.turns.items():
        if phase == Phase.DRAFT:
            d.texts[side], d.kinds[side], d.models[side] = turn.answer, "draft", turn.model
            d.record.setdefault("drafts", {})[side] = turn.answer
        elif phase == Phase.CROSS:
            raised = duel.apply_cross(d.ledger, side, turn.answer, state.duel["axes"])
            word = "issue" if len(raised) == 1 else "issues"
            display_info(f"{duel.side_label(side)} raised {len(raised) or 'no'} {word}.")
        elif phase == Phase.DEFEND:
            duel.apply_defend(d.ledger, side, turn.answer)
        elif phase == Phase.REPLY:
            duel.apply_reply(d.ledger, side, turn.answer)
        elif phase == Phase.REVISE:
            d.texts[side] = duel.apply_revise(d.ledger, side, turn.answer)
            d.kinds[side], d.models[side] = "revised", turn.model


def _run_judge(d, client, state):
    """JUDGE on a fresh neutral conversation. Returns the _PhaseResult."""
    order = duel.judge_order(_duel_rng)
    conversation = duel.new_conversation(duel.JUDGE_SYSTEM)
    text = duel.judge_prompt(d.brief, [d.texts[s] for s in order])
    d.number += 1
    console.print(f"[info]Phase {d.number} · judge[/info]")
    job = _Job(duel.JUDGE, conversation, text, _DUEL_SLOTS[duel.JUDGE], dict(state.options))
    result = _run_phase(Phase.JUDGE, [job], client, state, d.held)
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


def _run_merge(d, base, client, state):
    """MERGE with `base` as the base side. Returns (strengths, text) or None (message shown)."""
    conversation = duel.new_conversation(duel.JUDGE_SYSTEM)
    text = duel.merge_prompt(d.brief, d.texts[base], d.texts[duel.other(base)])
    d.number += 1
    console.print(f"[info]Phase {d.number} · merge (base {duel.side_label(base)})[/info]")
    job = _Job(duel.JUDGE, conversation, text, _DUEL_SLOTS[duel.JUDGE], dict(state.options))
    result = _run_phase(Phase.MERGE, [job], client, state, d.held)
    entry = {"n": d.number, "phase": "MERGE", "ok": result.ok, "cause": result.cause}
    d.phases.append(entry)
    if not result.ok:
        return None
    turn = result.turns[duel.JUDGE]
    parsed = duel.parse_merge(turn.answer)
    if parsed is None:
        entry.update(ok=False, cause="format")
        display_error(f"The merge reply had no {duel.REWRITE_MARK} line, so it was discarded.")
        return None
    d.merges.append({"base": base, "strengths": parsed[0], "text": parsed[1]})
    return parsed[0], parsed[1], turn.model


def _show_candidates(d):
    for side in duel.SIDES:
        _show_text(f"Side {side.upper()} ({d.kinds[side]})", d.texts[side])


def _final_choice(d, client, state):
    """The final-choice menu. Returns (text, model, choice) or None when the user discards."""
    merged = None  # (base, text, model)
    if d.judge is not None:
        console.print(f"[bold]Judge ({duel.order_note(tuple(d.judge['order']))}):[/bold]")
        console.print(d.judge["reasons"], markup=False, highlight=False)
        side = d.judge["recommendation"]
        display_info(
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
            console.print()
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
            _show_candidates(d)
            if merged:
                _show_text(f"Merged (base {duel.side_label(merged[0])})", merged[1])
        elif word == "j":
            if d.judge and d.judge["recommendation"]:
                display_info("The judge has already recommended a side.")
                continue
            result = _run_judge(d, client, state)
            _save_duel_record(d, state)
            if result.ok and d.judge:
                side = d.judge["recommendation"]
                display_info(
                    f"Recommendation: {duel.side_label(side)} ({duel.order_note(tuple(d.judge['order']))})."
                    if side
                    else "The judge made no recommendation."
                )
        elif word == "m":
            base = rest if rest in duel.SIDES else None
            if rest and base is None:
                display_error("Usage: m [a|b]")
                continue
            if base is None and d.judge:
                base = d.judge["recommendation"]
            if base is None:
                display_error("There is no recommendation to take the base from. Use m a or m b.")
                continue
            outcome = _run_merge(d, base, client, state)
            _save_duel_record(d, state)
            if outcome:
                merged = (base, outcome[1], outcome[2])
                _show_text(f"Merged text (base {duel.side_label(base)}); c uses it", outcome[1])
        else:
            display_error("Choose one of: " + ", ".join(options))


def _start_duel(brief_text, auto, client, conversation, state):
    """Check the server and build the duel; None (with the reason shown) when it cannot start."""
    client.refresh()
    if client.server_slots is None:
        display_error("Could not read the slot count from llama-server.")
        return None
    if client.server_slots < 2:
        display_error("Duel needs two slots. Start llama-server with --parallel 2.")
        return None
    if brief_text == '"""':
        brief_text = get_multiline_input()
        if brief_text is None:
            console.print()
            return None
    if not brief_text.strip():
        return None
    prepared = _prepare_user_message(brief_text)
    if prepared is None:
        return None
    brief, includes = prepared
    with echo_suppressed():
        _sync_notes(conversation, state, client)
    setting = state.duel
    d = _Duel(brief=brief, includes=includes, started=datetime.now(), auto=auto or setting["auto"])
    for side in duel.SIDES:
        prompt = duel.side_system(conversation.system_prompt, setting[f"stance_{side}"])
        d.sides[side] = duel.new_conversation(prompt, conversation.project_notes)
    d.options = duel.side_options(state.options, setting, client.sampling_defaults())
    d.settings = {k: setting[k] for k in _DUEL_KEYS if k != "auto"}
    d.settings["auto"] = d.auto
    return d


def _show_header(d, client, state):
    n_ctx = f"{client.server_n_ctx:,}" if client.server_n_ctx else "unknown"
    display_info(
        f"Duel on {client.server_slots} slots (side A on slot 0, side B on slot 1); context {n_ctx} tokens, "
        f"assumed {d.settings['context']} between the sides."
    )
    for side in duel.SIDES:
        extra = [f"{k} {d.options[side][k]}" for k in ("temperature", "seed") if k in d.options[side]]
        suffix = f" ({', '.join(extra)})" if extra else ""
        display_info(f"Side {side.upper()}{suffix}: {escape(d.settings[f'stance_{side}'])}")


def _handle_duel(args, client, conversation, state):
    """/duel: two slots write the same story and challenge each other (docs/duel.md)."""
    cmd = duel.parse_args(args)
    if cmd.kind == "usage":
        _duel_usage(client, state)
    elif cmd.kind == "error":
        display_error(cmd.error)
    elif cmd.kind == "ledger":
        _duel_ledger(state)
    elif cmd.kind == "set":
        _duel_set(cmd.key, cmd.value, state)
    else:
        _run_duel(cmd, client, conversation, state)


def _run_duel(cmd, client, conversation, state):
    d = _start_duel(cmd.brief, cmd.auto, client, conversation, state)
    if d is None:
        return
    _show_header(d, client, state)
    state.duel_last = _duel_record(d)
    _save_duel_record(d, state)

    after = None
    d.ended = "normal"
    while True:
        # Pauses sit between phases only, so not after the last one.
        if after is not None and not d.auto and duel.next_phase(after, d.ledger) is not None:
            if _duel_pause(d, after) == "stop":
                d.ended = "stopped"
                break
        step = duel.next_phase(after, d.ledger)
        if step is None:
            break
        phase, who, skipped = step
        if skipped:
            display_info(duel.skip_note(skipped))
            if Phase.REVISE in skipped:
                d.ledger.close()
        if phase == Phase.JUDGE:
            result = _run_judge(d, client, state)
        else:
            _announce_phase(d, phase, who)
            result = _run_phase(phase, _duel_jobs(d, phase, who, state), client, state, d.held)
            d.phases.append({"n": d.number, "phase": phase.name, "ok": result.ok, "cause": result.cause})
            if result.ok:
                _apply_phase(d, phase, result, state)
                if phase == Phase.REVISE:
                    d.ledger.close()
        if not result.ok:
            d.ended = f"abnormal:{phase.name}:{result.cause}"
            if phase == Phase.DRAFT:
                display_error("The drafts could not be completed, so the duel ends and nothing is added.")
            else:
                display_info("This phase was discarded; the duel goes no further.")
            break
        after = phase
        _save_duel_record(d, state)

    if len(d.texts) < 2:
        d.choice = "none"
        _save_duel_record(d, state)
        return
    _save_duel_record(d, state)
    chosen = _final_choice(d, client, state)
    d.choice = chosen[2] if chosen else "none"
    _save_duel_record(d, state)
    if chosen is None:
        display_info("Discarded; nothing was added to the conversation.")
        return
    text, model, _ = chosen
    conversation.add_user(d.brief, includes=d.includes)
    conversation.add_assistant(text, model=model or state.model)
    with echo_suppressed():
        _auto_save(conversation, state)
    display_info("Added to the conversation.")


def handle_command(cmd, args, client, conversation, state):
    """Handle a slash command. Returns True if the REPL should continue."""
    if cmd == "/?":
        print_help()

    elif cmd == "/exit":
        _auto_save(conversation, state)
        display_info("Goodbye!")
        return False

    elif cmd == "/clear":
        conversation.clear()
        display_info("Conversation cleared.")

    elif cmd == "/system":
        if not args:
            current = conversation.system_prompt or "(none)"
            display_info(f"Current system prompt: {current}")
        elif args.strip() == '"""':
            text = get_multiline_input()
            if text is not None and _system_prompt_fits(text, client, conversation, state):
                conversation.system_prompt = text
                display_info("System prompt set.")
        else:
            # Attempt to interpret args as a file path
            # Resolve path relative to cwd, expand user
            resolved_path = Path(args).expanduser().resolve()
            cwd = Path.cwd()
            try:
                # Python 3.9+ has is_relative_to
                if resolved_path.is_relative_to(cwd):
                    in_cwd = True
                else:
                    in_cwd = False
            except AttributeError:
                # For older Python, compare commonpath
                in_cwd = (
                    resolved_path.is_relative_to(cwd) if hasattr(resolved_path, "is_relative_to") else False
                )
            # Fallback for older versions
            if not in_cwd:
                # Check if resolved_path is under cwd by walking parents
                try:
                    in_cwd = cwd in resolved_path.parents or resolved_path == cwd
                except Exception:
                    in_cwd = False
            if in_cwd and resolved_path.is_file():
                ok, result = _read_file(str(resolved_path))
                if not ok:
                    display_error(result)
                elif _system_prompt_fits(result, client, conversation, state):
                    conversation.system_prompt = result
                    conversation.source_file = str(resolved_path)
                    display_info("System prompt set from file.")
            elif _system_prompt_fits(args, client, conversation, state):
                # Treat as plain string prompt
                conversation.system_prompt = args
                conversation.source_file = None
                display_info("System prompt set.")

    elif cmd == "/save":
        name = args.strip() or None
        try:
            conv_dir = _conversations_dir(state)
            filepath = conversation.save(
                conv_dir,
                name=name,
                model=state.model,
                omit_thinking=_omit_thinking(state),
            )
            display_info(f"Conversation saved: {filepath}")
        except OSError as e:
            display_error(f"Failed to save: {e}")

    elif cmd == "/load":
        name = args.strip()
        if not name:
            display_error("Usage: /load <name>")
        else:
            try:
                conv_dir = _conversations_dir(state)
                loaded_conv, loaded_model = Conversation.load(conv_dir, name)
                # Apply the whole saved conversation against the model being served
                # now. The model names recorded in the file are information only:
                # they never select a model and are never sent to the server.
                conversation.messages = loaded_conv.messages
                if state.project:
                    # The project's prompt wins over the one saved in the file.
                    conversation.system_prompt = state.project_prompt
                    conversation.source_file = state.project_source_file
                else:
                    conversation.system_prompt = loaded_conv.system_prompt
                    # Where that system prompt came from (None clears a stale value).
                    conversation.source_file = loaded_conv.source_file
                saved_with = f", saved with model: {escape(loaded_model)}" if loaded_model else ""
                display_info(
                    f"Loaded conversation: {name} ({len(conversation.messages)} messages{saved_with})"
                )
            except FileNotFoundError:
                display_error(f"No saved conversation named '{name}'.")
            except Exception as e:
                display_error(f"Failed to load: {e}")

    elif cmd == "/cat":
        name = args.strip()
        if not name:
            display_error("Usage: /cat <name>")
        else:
            try:
                conv_dir = _conversations_dir(state)
                loaded_conv, loaded_model = Conversation.load(conv_dir, name)
                display_cat_conversation(name, loaded_conv, loaded_model)
            except FileNotFoundError:
                display_error(f"No saved conversation named '{name}'.")
            except Exception as e:
                display_error(f"Failed to read conversation: {e}")

    elif cmd == "/recall":
        arg = args.strip()
        if not arg:
            display_error("Usage: /recall <pair_number>")
        else:
            try:
                pair_index = int(arg)
            except ValueError:
                display_error("Pair number must be an integer.")
                return True
            try:
                conversation.recall(pair_index)
                display_info(f"Recalled pair {pair_index} into context.")
            except IndexError as e:
                display_error(str(e))

    elif cmd == "/conversations":
        conv_dir = _conversations_dir(state)
        conversations = Conversation.list_saved(conv_dir)
        display_conversations(conversations)

    elif cmd == "/stats":
        state.show_stats = not state.show_stats
        status = "on" if state.show_stats else "off"
        display_info(f"Stats display: {status}")

    elif cmd == "/set":
        _handle_set(args, state, client)

    elif cmd == "/duel":
        _handle_duel(args, client, conversation, state)

    elif cmd == "/retry":
        if len(conversation.messages) < 2:
            display_error("Nothing to retry (need at least one exchange).")
        elif conversation.messages[-1]["role"] != "assistant":
            display_error("Last message is not an assistant response.")
        else:
            conversation.messages.pop()  # remove assistant
            state.retry_text = conversation.messages[-1]["content"]
            # The text is already flat; carry the files it came from, do not re-expand.
            state.pending_includes = conversation.messages[-1].get("includes", [])
            conversation.messages.pop()  # remove user (REPL will re-add)

    elif cmd == "/read":
        if not args:
            display_error("Usage: /read <path> [<path> ...]")
            return True
        # Support multiple filenames; quote paths containing spaces
        try:
            filenames = shlex.split(args)
        except ValueError as e:
            display_error(f"Could not parse paths: {e}")
            return True
        # Same splice rule and failure behaviour as @@<path>: any problem file aborts.
        segments = [("file", name, name) for name in filenames]
        combined, includes, errors = build_message(segments)
        if errors:
            for error in errors:
                display_error(escape(error))
            return True
        state.retry_text = combined
        state.pending_includes = includes
        console.print("User:")
        console.print(combined)
        display_included_files(includes)

    elif cmd == "/config":
        _handle_config(args, state, client)

    elif cmd == "/project":
        _handle_project(args, client, conversation, state)

    elif cmd == "/remember":
        _handle_remember(args, client, conversation, state)

    elif cmd == "/info":
        try:
            counts = tokens.breakdown(client, conversation, model=state.model)
        except (RequestException, ValueError, KeyError):
            counts = None  # the server could not be asked; the rest of the table still shows
        display_conversation_info(conversation.summary(), counts, state.context_length)
        if counts and counts.get("notes") and tokens.notes_heavy(counts["notes"], counts["n_ctx"]):
            display_notes_heavy(counts["notes"], counts["n_ctx"])

    else:
        display_error(f"Unknown command: {cmd}. Type /? for available commands.")

    return True


def _auto_save(conversation, state, force=False):
    """Silently auto-save the conversation if enabled; force saves even when it is not.

    Returns False only when the write failed, so a caller that is about to discard the
    conversation (a project switch) can refuse; nothing to save counts as success.
    """
    if not force and not state.config.get("auto_save", True):
        return True
    if not conversation.messages:
        return True
    try:
        conv_dir = _conversations_dir(state)
        conversation.save(
            conv_dir,
            name=state.auto_save_name,
            model=state.model,
            omit_thinking=_omit_thinking(state),
        )
    except OSError:
        return False
    return True


def main():
    config = load_config()
    args = parse_args()

    client = LlamaClient(args.url or config["llama_url"], think_override=_think_override(config))

    if not client.is_available():
        display_error(
            "Cannot connect to llama-server. Make sure it's running with: llama-server --port 8001 -m <model>"
        )
        sys.exit(1)

    # The model is whatever the server is serving; it is never chosen here.
    client.refresh()
    if not client.server_model:
        display_error("Failed to read the server's properties (GET /props). Is llama-server fully started?")
        sys.exit(1)

    state = State(
        model=client.server_model,
        config=config,
        context_length=client.server_n_ctx,
        options=_initial_options(config),
        duel=_initial_duel(config),
        auto_save_name="auto_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        global_config=dict(config),
    )
    # Startup value for /config; no announcement (the /config row covers it).
    _update_think_tags(state, client.think_tags, announce=False)
    conversation = Conversation(system_prompt=config["system_prompt"])

    init_readline(lambda: _conversations_dir(state), lambda: _projects_dir(state))
    if args.project and not _switch_project(args.project, client, conversation, state):
        sys.exit(1)
    print_welcome(state.model, state.project)

    # Main REPL
    try:
        while True:
            user_input = get_user_input()

            if user_input is None:  # EOF
                console.print()
                break

            text = user_input.strip()
            if not text:
                continue

            # Multiline input mode
            if text == '"""':
                text = get_multiline_input()
                if text is None:
                    console.print()
                    break
                if not text.strip():
                    continue

            # Handle commands
            history = list(conversation.messages)  # what /retry may take away, if the send is refused
            if text.startswith("/"):
                parts = text.split(None, 1)
                cmd = parts[0].lower()
                cmd_args = parts[1] if len(parts) > 1 else ""
                if not handle_command(cmd, cmd_args, client, conversation, state):
                    break
                # Check if /retry or /read set text to re-send
                if state.retry_text is None:
                    # Commands don't become messages
                    continue
                text = state.retry_text
                state.retry_text = None
                # /read and /retry hand over the files the text came from
                includes, state.pending_includes = state.pending_includes, []
            else:
                # Not a command: splice in any @@<path> files. A bad include aborts the send,
                # so nothing is added to the conversation and the model never sees the text.
                prepared = _prepare_user_message(text)
                if prepared is None:
                    continue
                text, includes = prepared

            # Echo is off for the whole turn (check, request, reply, stats, autosave) and comes
            # back just before the next prompt: readline entered with ECHO off draws nothing.
            with echo_suppressed():
                # An outside edit of project.md joins the prompt before it is priced.
                _sync_notes(conversation, state, client)
                # The message joins the conversation only once its prompt is known to fit.
                candidate = conversation.get_messages() + [{"role": "user", "content": text}]
                fits, priced = _fits_window(client, candidate, state)
                if not fits:
                    conversation.messages[:] = history  # a refused /retry loses nothing
                else:
                    if priced and includes:
                        display_prompt_size(*priced)
                    conversation.add_user(text, includes=includes)
                    try:
                        chat_stream = client.chat(
                            model=state.model,
                            messages=conversation.get_messages(),
                            options=state.options,
                            refreshed=priced is not None,
                        )
                        # Before the reply is shown and before any autosave, so they already
                        # know this model, its context length and its tags.
                        _sync_server_info(state, chat_stream.server_model, chat_stream.server_n_ctx)
                        _update_think_tags(state, chat_stream.think_tags)
                        response = display_assistant_stream(chat_stream, think_end=_think_end(state))
                        _store_reply(conversation, response, chat_stream, state)
                        if chat_stream.truncated:
                            display_truncated_warning(not conversation.messages[-1]["content"])
                        if state.show_stats:
                            display_stats(chat_stream.stats)
                        _auto_save(conversation, state)
                    except KeyboardInterrupt:
                        console.print()
                        display_info("Response interrupted.")
                    except ConnectionError:
                        display_error("Lost connection to llama-server. Is it still running?")
                        # Remove the unanswered user message
                        conversation.messages.pop()
                    except ThinkTagsError as e:
                        display_error(str(e))
                        conversation.messages.pop()
                    except HTTPError as e:
                        display_error(f"llama-server error: {e}")
                        conversation.messages.pop()

                console.print()
    finally:
        _auto_save(conversation, state)
        save_readline_history()


if __name__ == "__main__":
    main()
