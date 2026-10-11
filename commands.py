"""The slash-command table: one entry per command (docs/commands.md).

`ui.py` derives the command list, the help table and tab-completion from it;
`chat.handle_command` looks the handler up by name. Stdlib only; imports neither
`chat` nor `ui`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Subcommands:
    """Completion of a command's first word, and what the word after some of them names."""

    names: tuple
    arguments: tuple = ()  # (subcommand, completion kind) pairs, e.g. ("use", "project")


@dataclass(frozen=True)
class Command:
    name: str
    handler_name: str  # a function in chat's namespace, called as (args, client, conversation, state)
    rows: tuple  # help table rows: (usage, description)
    aliases: tuple = ()
    complete: object = None  # None, "conversation", or Subcommands


COMMANDS = (
    Command(
        "/cat",
        "_handle_cat",
        (("/cat <name>", "Print a saved conversation to the console"),),
        complete="conversation",
    ),
    Command("/clear", "_handle_clear", (("/clear", "Clear conversation history"),)),
    Command(
        "/config",
        "_cmd_config",
        (
            (
                "/config",
                "Show configuration; '/config save_thinking on|off' controls whether <think> blocks are saved (omit on/off to toggle)",
            ),
        ),
    ),
    Command("/conversations", "_handle_conversations", (("/conversations", "List saved conversations"),)),
    Command(
        "/duel",
        "_handle_duel",
        (
            (
                "/duel [--auto] <brief>",
                "Two slots write the same story and challenge each other (/duel for usage; needs --parallel 2)",
            ),
        ),
    ),
    Command("/exit", "_handle_exit", (("/exit", "Quit the application"),)),
    Command("/help", "_handle_help", (("/help", "Show this help message"),), aliases=("/?",)),
    Command("/info", "_handle_info", (("/info", "Show conversation and context window statistics"),)),
    Command(
        "/load", "_handle_load", (("/load <name>", "Load a saved conversation"),), complete="conversation"
    ),
    Command(
        "/project",
        "_handle_project",
        (
            (
                "/project",
                "Show the active project (/project list | new <name> | use <name> | reload | leave)",
            ),
        ),
        complete=Subcommands(
            ("info", "list", "new", "use", "reload", "leave"),
            (("use", "project"), ("new", "project")),
        ),
    ),
    Command("/read", "_handle_read", (("/read <path>", "Read a text file into the conversation"),)),
    Command("/recall", "_handle_recall", (("/recall <n>", "Recall message pair n into context"),)),
    Command(
        "/remember",
        "_handle_remember",
        (
            (
                "/remember <text>",
                "Merge a note into the project's project.md (shown as a diff for approval)",
            ),
        ),
    ),
    Command("/retry", "_handle_retry", (("/retry", "Regenerate the last response"),)),
    Command("/save", "_handle_save", (("/save <name>", "Save conversation (default: timestamp)"),)),
    Command(
        "/set",
        "_cmd_set",
        (
            ("/set", "Show model options and this session's overrides of the server's values"),
            ("/set <key> <val>", "Set a model option (or 'default' to reset)"),
        ),
    ),
    Command("/stats", "_handle_stats", (("/stats", "Toggle token and context stats display"),)),
    Command(
        "/system",
        "_handle_system",
        (
            (
                "/system <prompt>",
                'Set the system prompt (use """ for multiline or a path to a file within the current directory)',
            ),
        ),
    ),
)

_BY_NAME = {name: c for c in COMMANDS for name in (c.name, *c.aliases)}


def names():
    """Every name and alias, sorted."""
    return sorted(_BY_NAME)


def lookup(name):
    """The Command answering to name (or an alias of it), or None."""
    return _BY_NAME.get(name)


def check_handlers(namespace):
    """Raise if a command's handler_name is not defined in namespace (chat's globals)."""
    missing = [c.handler_name for c in COMMANDS if not callable(namespace.get(c.handler_name))]
    if missing:
        raise RuntimeError(f"commands.py names handlers that do not exist: {', '.join(missing)}")
