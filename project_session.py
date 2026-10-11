"""Project switching, project.md notes and /remember (docs/projects.md).

Never imports chat: everything it needs from chat.py (the display functions, the console,
auto-save, the option checks) arrives in a ProjectHost, the `host` argument. See
chat._project_host().
"""

from datetime import datetime

from requests.exceptions import ConnectionError, HTTPError, RequestException
from rich.markup import escape

import projects
import tokens
from config import apply_project
from llama_client import ThinkTagsError


# Reserved on top of the notes-sized reply for a thinking model's reasoning in /remember.
_REMEMBER_THINK_ALLOWANCE = 2000


def _read_scope(name, state, host):
    """Everything a project switch needs, read up front; None (error shown) on any failure.

    Reading first means a bad project.toml or an unreadable file leaves the session untouched.
    """
    try:
        path = projects.project_dir(host.projects_dir(state), name)
    except ValueError as exc:
        host.display_error(str(exc))
        return None
    if not path.is_dir():
        host.display_error(f"No project named '{name}'. Create it with /project new {name}.")
        return None
    try:
        settings, ignored, problems = projects.load_settings_checked(path)
        prompt = projects.read_system_prompt(path)
        notes = projects.read_notes(path)
    except (ValueError, OSError, UnicodeDecodeError) as exc:
        host.display_error(f"Cannot read project '{name}': {exc}")
        return None
    return {
        "name": name,
        "dir": path,
        "settings": settings,
        "ignored": ignored,
        "problems": problems,
        "prompt": prompt,
        "notes": notes,
    }


def _reset_session(state, settings, host):
    """Settings back to config.toml plus a project's `settings`; the toggles with no key to their defaults."""
    state.config = apply_project(state.global_config, settings)
    state.options = host.initial_options(state.global_config, quiet=True)
    state.options.update(host.initial_options(settings, source="project.toml"))
    state.show_stats = True
    state.duel = host.initial_duel(state.global_config, quiet=True)
    state.duel_last = None


def _apply_scope(conversation, state, scope, host):
    """Make `scope` (from _read_scope, or None for no project) the active one. Cannot fail."""
    settings = scope["settings"] if scope else {}
    _reset_session(state, settings, host=host)
    prompt = scope["prompt"] if scope else ""
    state.project = scope["name"] if scope else None
    state.project_dir = scope["dir"] if scope else None
    state.project_prompt = prompt or state.global_config.get("system_prompt", "")
    state.project_source_file = str(scope["dir"] / "system.md") if prompt else None
    conversation.system_prompt = state.project_prompt
    conversation.source_file = state.project_source_file
    conversation.project_notes = scope["notes"] if scope else ""
    state.notes_digest = projects.notes_digest(conversation.project_notes)


def _announce_scope(scope, client, conversation, host):
    """The lines that follow a switch or reload: ignored keys, unknown keys and a heavy-notes warning."""
    if not scope:
        return
    for key in scope["ignored"]:
        host.display_info(f"project.toml: {key} is ignored; put the system prompt in system.md.")
    for line in scope["problems"]:
        host.display_warning(line)
    notes = conversation.project_notes
    if not notes:
        return
    try:
        count = client.count_tokens(notes, add_special=False)
        n_ctx = client.server_n_ctx
    except (RequestException, ValueError, KeyError):
        return
    if isinstance(count, int) and isinstance(n_ctx, int) and tokens.notes_heavy(count, n_ctx):
        host.display_notes_heavy(count, n_ctx)


def _switch_project(name, client, conversation, state, host):
    """Switch to project `name`, or to no project when it is None. Returns True if switched.

    Everything fallible is read first. The current conversation is then saved (even with
    auto_save off) and, only if that worked, replaced by an empty one.
    """
    scope = _read_scope(name, state, host=host) if name else None
    if name and scope is None:
        return False
    if not host.auto_save(conversation, state, force=True):
        host.display_error("Not switched: the current conversation could not be saved.")
        return False
    _apply_scope(conversation, state, scope, host=host)
    conversation.messages.clear()
    state.auto_save_name = "auto_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    state.retry_text = None
    state.pending_includes = []
    if scope:
        host.display_info(f"Project: {escape(name)} ({scope['dir']})")
    else:
        host.display_info("No project selected.")
    host.display_info("Session settings were reset to the configured ones (/set, /config, /stats).")
    _announce_scope(scope, client, conversation, host=host)
    return True


def _reload_project(client, conversation, state, host):
    """Re-read system.md, project.md and project.toml; the conversation is kept."""
    scope = _read_scope(state.project, state, host=host)
    if scope is None:
        return
    _apply_scope(conversation, state, scope, host=host)
    host.display_info(f"Project reloaded: {escape(state.project)}")
    host.display_info("Session settings were reset to the configured ones (/set, /config, /stats).")
    _announce_scope(scope, client, conversation, host=host)


def _sync_notes(conversation, state, client, host):
    """Pick up an outside edit of project.md; the conversation carries the file's current text.

    Called before each send, so the pre-send check prices the notes that will be sent. An
    edit invalidates the server's cached prompt prefix, so one line says so.
    """
    if state.project_dir is None:
        return
    try:
        text = projects.read_notes(state.project_dir)
    except (OSError, UnicodeDecodeError) as exc:
        host.display_error(f"Cannot read project.md: {exc}; using the notes loaded earlier.")
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
        host.display_info("project.md changed.")
        return
    host.display_info(f"project.md changed: {count:,} tokens")
    n_ctx = state.context_length
    if n_ctx and tokens.notes_heavy(count, n_ctx):
        host.display_notes_heavy(count, n_ctx)


def _handle_project(args, client, conversation, state, host):
    parts = args.split(None, 1)
    sub = parts[0] if parts else "info"
    name = parts[1].strip() if len(parts) > 1 else ""
    if sub == "info":
        _project_info(client, conversation, state, host=host)
    elif sub == "list":
        host.display_projects(projects.list_projects(host.projects_dir(state)), state.project)
    elif sub == "new":
        if not name:
            host.display_error("Usage: /project new <name>")
            return
        try:
            path = projects.create_project(host.projects_dir(state), name)
        except (ValueError, FileExistsError, OSError) as exc:
            host.display_error(str(exc))
            return
        host.display_info(f"Project created: {path}. Use it with /project use {name}.")
    elif sub == "use":
        if not name:
            host.display_error("Usage: /project use <name>")
            return
        _switch_project(name, client, conversation, state, host=host)
    elif sub == "reload":
        if not state.project:
            host.display_error("No project selected. /project use <name>.")
        else:
            _reload_project(client, conversation, state, host=host)
    elif sub == "leave":
        if not state.project:
            host.display_error("No project selected.")
        else:
            _switch_project(None, client, conversation, state, host=host)
    else:
        host.display_error("Usage: /project [info | list | new <name> | use <name> | reload | leave]")


def _project_info(client, conversation, state, host):
    if not state.project:
        host.display_info(
            "No project selected. /project list shows the projects; /project use <name> picks one."
        )
        return
    host.display_info(f"Project: {escape(state.project)} ({state.project_dir})")
    notes = conversation.project_notes
    try:
        count = client.count_tokens(notes, add_special=False) if notes else 0
        size = f"{count:,} tokens"
    except (RequestException, ValueError, KeyError):
        size = f"{len(notes):,} characters"
    host.display_info(f"project.md: {size}")
    source = "system.md" if state.project_source_file else "global system prompt"
    host.display_info(f"System prompt: {source}")
    try:
        settings, _ = projects.load_settings(state.project_dir)
    except ValueError:
        settings = {}
    if settings:
        shown = ", ".join(f"{k} = {v}" for k, v in settings.items())
        host.display_info(f"project.toml: {escape(shown)}")


def _remember_room(messages, notes, note, client, state, host):
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
        host.console.print()
        host.display_info("Cancelled.")
        return False
    except (RequestException, ValueError, KeyError) as exc:
        host.display_error(f"{outcome}: could not check the context window ({exc}).")
        return False
    if not n_ctx:
        host.display_error(f"{outcome}: the context window size is unknown.")
        return False
    if tokens.fits(needed, n_ctx, reply + _REMEMBER_THINK_ALLOWANCE):
        return True
    if tokens.fits(needed, n_ctx, reply):
        return True  # no room for the allowance; a model that thinks long is caught by the cut-off check
    host.display_error(tokens.refusal(needed, n_ctx, reply, outcome))
    return False


def _handle_remember(args, client, conversation, state, host):
    """/remember <text>: ask the model to merge the note into project.md, then show the diff.

    The conversation is not touched. Nothing is written unless the user answers y, the
    reply was complete and project.md is unchanged since the model saw it.
    """
    if not state.project_dir:
        host.display_error("No project selected. /project use <name>.")
        return
    note = args.strip()
    if not note:
        host.display_error("Usage: /remember <text>")
        return
    with host.echo_suppressed():
        _sync_notes(conversation, state, client, host=host)
        notes = conversation.project_notes
        messages = projects.remember_messages(notes, note)
        if not _remember_room(messages, notes, note, client, state, host=host):
            return
        try:
            text, cut_off, think_tags = client.complete(
                state.model, messages, options={"temperature": 0}, refreshed=True
            )
        except KeyboardInterrupt:
            host.console.print()
            host.display_info("Cancelled. project.md is unchanged.")
            return
        except ConnectionError:
            host.display_error("Lost connection to llama-server. Is it still running?")
            return
        except (ThinkTagsError, HTTPError) as exc:
            host.display_error(f"Not remembered: {exc}")
            return
    if cut_off:
        host.display_error("Not remembered: the model's reply was cut off, so project.md is unchanged.")
        return
    new = projects.clean_reply(text, think_tags)
    if not new:
        host.display_error("Not remembered: the model's reply was empty. project.md is unchanged.")
        return
    if projects.shrank_too_much(notes, new):
        host.display_error(
            "Not remembered: the reply is under half the size of project.md, so it was rejected."
        )
        return
    diff = projects.diff_text(notes, new)
    if not diff:
        host.display_info("No change: project.md already says that.")
        return
    host.console.print(diff, markup=False, highlight=False)
    try:
        answer = input("Write this to project.md? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        host.console.print()
        answer = ""
    if answer.strip().lower() != "y":
        host.display_info("project.md unchanged.")
        return
    try:
        current = projects.read_notes(state.project_dir)
        if projects.notes_digest(current) != state.notes_digest:
            host.display_error(
                "project.md changed while you were deciding, so nothing was written. Run /remember again."
            )
            return
        projects.write_notes(state.project_dir, new)
    except (OSError, UnicodeDecodeError) as exc:
        host.display_error(f"Could not write project.md: {exc}")
        return
    conversation.project_notes = new
    state.notes_digest = projects.notes_digest(new)
    host.display_info("project.md updated.")
