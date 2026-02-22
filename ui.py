import readline
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from conversation import Conversation

HISTORY_FILE = Path.home() / ".local" / "share" / "chat" / "history"
HISTORY_MAX = 1000

COMMANDS = [
    "/?",
    "/cat",
    "/clear",
    "/config",
    "/conversations",
    "/exit",
    "/info",
    "/load",
    "/model",
    "/models",
    "/recall",
    "/retry",
    "/save",
    "/set",
    "/stats",
    "/system",
]

_conversations_dir = ""


def set_conversations_dir(path):
    """Set the conversations directory for tab-completion of /load."""
    global _conversations_dir
    _conversations_dir = path


theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "user_label": "bold green",
        "assistant_label": "bold blue",
    }
)

console = Console(theme=theme)


def print_welcome(model):
    console.print()
    console.print("[bold]AI Chat[/bold] (Ollama)", style="info")
    console.print(f"Model: [bold]{model}[/bold]")
    console.print("Type [bold]/?[/bold] for commands, [bold]/exit[/bold] to quit.")
    console.print()


def print_help():
    table = Table(title="Commands", show_header=True, header_style="bold")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("/help", "Show this help message")
    table.add_row("/exit", "Quit the application")
    table.add_row("/cat <name>", "Print a saved conversation to the console")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/models", "List available models")
    table.add_row("/model <name>", "Switch to a different model")
    table.add_row("/system <prompt>", "Set the system prompt")
    table.add_row("/recall <n>", "Recall message pair n into context")
    table.add_row("/retry", "Regenerate the last response")
    table.add_row("/save <name>", "Save conversation (default: timestamp)")
    table.add_row("/load <name>", "Load a saved conversation")
    table.add_row("/conversations", "List saved conversations")
    table.add_row("/set", "Show model options (seed, temperature, top_p)")
    table.add_row("/set <key> <val>", "Set a model option (or 'default' to reset)")
    table.add_row("/info", "Show conversation summary statistics")
    table.add_row("/stats", "Toggle token stats display")
    table.add_row("/config", "Show current configuration")
    table.add_row('"""', "Enter multiline input mode (or use Shift+Enter / Alt+Enter / paste)")
    console.print(table)


def display_models(models, current_model):
    table = Table(title="Available Models", show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Active")
    for m in models:
        marker = "*" if m == current_model else ""
        table.add_row(m, marker)
    console.print(table)


def display_conversations(conversations):
    if not conversations:
        console.print("[info]No saved conversations.[/info]")
        return
    table = Table(title="Saved Conversations", show_header=True, header_style="bold")
    table.add_column("Name", style="bold cyan")
    table.add_column("File")
    for name, filepath in conversations:
        table.add_row(name, str(filepath))
    console.print(table)


OLLAMA_DEFAULTS = {
    "seed": 0,
    "temperature": 0.8,
    "top_p": 0.9,
}


def display_config(config, current_model, options=None):
    table = Table(title="Configuration", show_header=True, header_style="bold")
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value")
    table.add_row("model", current_model)
    table.add_row("default_model", config["default_model"] or "(none)")
    table.add_row("system_prompt", config["system_prompt"] or "(none)")
    table.add_row("ollama_url", config["ollama_url"])
    table.add_row("conversations_dir", config["conversations_dir"])
    if options is not None:
        for key in sorted(options):
            val = options[key]
            if val is not None:
                table.add_row(key, str(val))
            else:
                table.add_row(key, f"{OLLAMA_DEFAULTS[key]} [dim](default)[/dim]")
    console.print(table)


def display_options(options):
    """Display current model options in a table."""
    table = Table(title="Model Options", show_header=True, header_style="bold")
    table.add_column("Option", style="bold cyan")
    table.add_column("Value")
    for key in sorted(options):
        val = options[key]
        table.add_row(key, str(val) if val is not None else "(default)")
    console.print(table)


def _format_timestamp(iso_str):
    """Format an ISO timestamp string for display."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ""


def display_cat_conversation(name, conversation, model):
    """Print a saved conversation's messages to the console."""
    console.print()
    console.print(f"[bold]Conversation:[/bold] {name}")
    if model:
        console.print(f"[bold]Model:[/bold] {model}")
    if conversation.system_prompt:
        console.print(f"[bold]System prompt:[/bold] {conversation.system_prompt}")
    console.print()

    if not conversation.messages:
        console.print("[dim]  (no messages)[/dim]")
        return

    pair_index = 0
    for msg in conversation.messages:
        ts = msg.get("timestamp", "")
        ts_display = f"  [dim]{_format_timestamp(ts)}[/dim]" if ts else ""

        if msg["role"] == "user":
            pair_index += 1
            console.print(
                f"[dim]\\[{pair_index}][/dim] [user_label]You:[/user_label]{ts_display}"
            )
        else:
            console.print(
                f"[dim]\\[{pair_index}][/dim] [assistant_label]Assistant:[/assistant_label]{ts_display}"
            )
        console.print(msg["content"])
        console.print()


def display_conversation_info(summary, last_stats=None):
    table = Table(title="Conversation Info", show_header=True, header_style="bold")
    table.add_column("Statistic", style="bold cyan")
    table.add_column("Value")
    user = summary["user_messages"]
    asst = summary["assistant_messages"]
    table.add_row("Messages", f"{summary['messages']} ({user} you, {asst} AI)")
    table.add_row("Words", f"{summary['words']:,}")
    table.add_row("Characters", f"{summary['characters']:,}")
    if last_stats and "prompt_eval_count" in last_stats:
        table.add_row("Prompt tokens", f"{last_stats['prompt_eval_count']:,}")
    console.print(table)


def display_info(msg):
    console.print(f"[info]{msg}[/info]")


def display_error(msg):
    console.print(f"[error]{msg}[/error]")


def display_assistant_stream(token_generator):
    """Print streamed tokens live with word-wrap.

    Returns the full response text.
    """
    console.print("[assistant_label]Assistant:[/assistant_label]")
    full_text = ""
    term_width = console.width or 80
    col = 0  # current column position
    word_buf = ""  # incomplete word being accumulated
    visual_lines = 0  # lines emitted (for erasure)

    def _flush_word(word):
        """Write a complete word, wrapping to next line if needed."""
        nonlocal col, visual_lines
        if col + len(word) > term_width and col > 0:
            sys.stdout.write("\n")
            visual_lines += 1
            col = 0
        sys.stdout.write(word)
        col += len(word)

    try:
        for token in token_generator:
            full_text += token
            word_buf += token

            # Process explicit newlines first
            while "\n" in word_buf:
                before, _, word_buf = word_buf.partition("\n")
                if before:
                    _flush_word(before)
                sys.stdout.write("\n")
                visual_lines += 1
                col = 0

            # Flush complete words (delimited by spaces)
            while " " in word_buf:
                word, _, word_buf = word_buf.partition(" ")
                _flush_word(word)
                # Write the space (wrap first if at edge)
                if col >= term_width:
                    sys.stdout.write("\n")
                    visual_lines += 1
                    col = 0
                sys.stdout.write(" ")
                col += 1

            sys.stdout.flush()

        # Flush any remaining partial word
        if word_buf:
            _flush_word(word_buf)
            sys.stdout.flush()
    except KeyboardInterrupt:
        full_text += " [interrupted]"
    finally:
        # Account for the last line if it has content
        if col > 0:
            visual_lines += 1

        # Move to next line after streaming completes
        sys.stdout.write("\n")
        sys.stdout.flush()

    return full_text


def display_context_warning(used, limit):
    """Display a warning when context usage is high."""
    pct = used / limit * 100
    console.print(
        f"[warning]Warning: context window {pct:.0f}% full "
        f"({used:,} / {limit:,} tokens)[/warning]"
    )


def display_stats(stats):
    """Display token generation stats in a dim line."""
    if not stats:
        return
    parts = []
    if "eval_count" in stats:
        parts.append(f"{stats['eval_count']} tokens")
    if "tokens_per_second" in stats:
        parts.append(f"{stats['tokens_per_second']:.1f} tok/s")
    if "prompt_eval_count" in stats:
        parts.append(f"{stats['prompt_eval_count']} prompt tokens")
    if parts:
        console.print(f"[dim]  {' | '.join(parts)}[/dim]")


def _command_completer(text, state):
    """Readline completer for slash commands and /load arguments."""
    line = readline.get_line_buffer().lstrip()
    if (line.startswith("/load ") or line.startswith("/cat ")) and _conversations_dir:
        names = [n for n, _ in Conversation.list_saved(_conversations_dir)]
        matches = [n for n in names if n.startswith(text)]
    elif text.startswith("/"):
        matches = [c for c in COMMANDS if c.startswith(text)]
    else:
        matches = []
    if state < len(matches):
        return matches[state]
    return None


def init_readline():
    """Load readline history from disk and configure tab-completion."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(HISTORY_MAX)
    readline.set_completer(_command_completer)
    readline.set_completer_delims(" ")
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set enable-bracketed-paste on")
    readline.parse_and_bind(r'"\M-\C-m": "\n"')
    readline.parse_and_bind(r'"\e[13;2u": "\n"')    # Shift+Enter — Kitty keyboard protocol
    readline.parse_and_bind(r'"\e[27;2;13~": "\n"')  # Shift+Enter — xterm modifyOtherKeys


def save_readline_history():
    """Save readline history to disk."""
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


def get_user_input():
    """Prompt the user for input with ollama-style placeholder.

    Shows: >>> Send a message (/? for help)
    Placeholder is grey and disappears as soon as the user types.
    Ctrl-C clears the current line and re-prompts.
    Returns None on EOF (Ctrl-D).
    """
    import termios
    import tty

    PROMPT = ">>> "
    PLACEHOLDER = "Send a message (/? for help)"

    # Readline prompt: bold green >>>, with ANSI codes wrapped in \x01/\x02
    # so readline correctly calculates visible width.
    rl_prompt = f"\x01\033[1;32m\x02{PROMPT}\x01\033[0m\x02"

    while True:
        # Print prompt and placeholder
        sys.stdout.write(f"\033[1;32m{PROMPT}\033[0m")
        sys.stdout.write(f"\033[90m{PLACEHOLDER}\033[0m")
        sys.stdout.write(f"\033[{len(PLACEHOLDER)}D")
        sys.stdout.flush()

        # Read one character in raw mode to detect first keystroke
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Erase the entire prompt + placeholder line, reposition cursor
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

        if ch == "\x03":  # Ctrl-C
            print()
            continue
        if ch == "\x04":  # Ctrl-D
            print()
            return None
        if ch == "\r" or ch == "\n":  # Enter with no input
            print()
            return ""

        # Stuff the first character into readline's input buffer
        # so it appears as part of the editable line.
        # stuff_char exists at runtime in CPython but is not in the
        # type stubs, so use hasattr + getattr to avoid type errors.
        _stuff_char = getattr(readline, "stuff_char", None)
        if _stuff_char is not None:
            _stuff_char(ord(ch))
        else:
            # Fallback: use pre_input_hook to insert the character
            def insert_char():
                readline.insert_text(ch)
                readline.redisplay()
                readline.set_pre_input_hook(None)

            readline.set_pre_input_hook(insert_char)

        try:
            return input(rl_prompt)
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            return None


def get_multiline_input():
    """Read lines until a closing \"\"\" is entered.

    Returns the joined text, or None on EOF.
    """
    console.print('[info]  ... entering multiline mode (type """ to finish)[/info]')
    lines = []
    try:
        while True:
            line = input("... ")
            if line.strip() == '"""':
                break
            lines.append(line)
    except EOFError:
        return None
    return "\n".join(lines)
