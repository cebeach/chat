#!/usr/bin/env python3
"""AI Chat — a terminal chat application powered by the llama.cpp server."""

import argparse
import math
from pathlib import Path
import re
import shlex
import sys
from dataclasses import dataclass, field
from datetime import datetime

from requests.exceptions import ConnectionError, HTTPError, RequestException
from rich.markup import escape

from config import load_config
from conversation import Conversation, split_think
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
    display_info,
    display_options,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with the model served by llama-server")
    parser.add_argument(
        "--url",
        default=None,
        help="Override llama-server URL",
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


def _initial_options(config):
    """The session's overrides from config.toml: only the keys that are set and valid.

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
            display_error(f"config.toml: {key} {exc}; ignored.")
    return options


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
        display_config(state.config, state.model, rows, state.think_tags)
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


def _system_prompt_fits(text, client, conversation, state):
    """True when `text` as the system prompt, with the stored messages, fits the window.

    Runs with the keyboard echo off like a model turn, because counting a large prompt
    takes a moment. When it does not fit the error is printed and the old prompt stays.
    """
    if not text.strip():
        return True
    candidate = [{"role": "system", "content": text}]
    candidate += [m for m in conversation.get_messages() if m["role"] != "system"]
    with echo_suppressed():
        ok, _ = _fits_window(client, candidate, state, outcome="System prompt not changed")
    return ok


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
            conv_dir = state.config["conversations_dir"]
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
                conv_dir = state.config["conversations_dir"]
                loaded_conv, loaded_model = Conversation.load(conv_dir, name)
                # Apply the whole saved conversation against the model being served
                # now. The model names recorded in the file are information only:
                # they never select a model and are never sent to the server.
                conversation.messages = loaded_conv.messages
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
                conv_dir = state.config["conversations_dir"]
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
        conv_dir = state.config["conversations_dir"]
        conversations = Conversation.list_saved(conv_dir)
        display_conversations(conversations)

    elif cmd == "/stats":
        state.show_stats = not state.show_stats
        status = "on" if state.show_stats else "off"
        display_info(f"Stats display: {status}")

    elif cmd == "/set":
        _handle_set(args, state, client)

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

    elif cmd == "/info":
        try:
            counts = tokens.breakdown(client, conversation, model=state.model)
        except (RequestException, ValueError, KeyError):
            counts = None  # the server could not be asked; the rest of the table still shows
        display_conversation_info(conversation.summary(), counts, state.context_length)

    else:
        display_error(f"Unknown command: {cmd}. Type /? for available commands.")

    return True


def _auto_save(conversation, state):
    """Silently auto-save the conversation if enabled."""
    if not state.config.get("auto_save", True):
        return
    if not conversation.messages:
        return
    try:
        conv_dir = state.config["conversations_dir"]
        conversation.save(
            conv_dir,
            name=state.auto_save_name,
            model=state.model,
            omit_thinking=_omit_thinking(state),
        )
    except OSError:
        pass


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
        auto_save_name="auto_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    # Startup value for /config; no announcement (the /config row covers it).
    _update_think_tags(state, client.think_tags, announce=False)
    conversation = Conversation(system_prompt=config["system_prompt"])

    init_readline(config["conversations_dir"])
    print_welcome(state.model)

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
