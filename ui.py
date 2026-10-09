import contextlib
import re
import readline
import signal
import sys
import termios
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.theme import Theme

from conversation import Conversation

HISTORY_FILE = Path.home() / ".local" / "share" / "chat" / "history"
HISTORY_MAX = 1000

COMMANDS = [
    "/read",
    "/?",
    "/cat",
    "/clear",
    "/config",
    "/conversations",
    "/exit",
    "/info",
    "/load",
    "/recall",
    "/retry",
    "/save",
    "/set",
    "/stats",
    "/system",
]

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
    console.print("[bold]AI Chat[/bold] (llama.cpp)", style="info")
    console.print(f"Model: [bold]{model}[/bold]")
    console.print("Type [bold]/?[/bold] for commands, [bold]/exit[/bold] to quit.")
    console.print()


def print_help():
    table = Table(title="Commands", show_header=True, header_style="bold")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("/cat <name>", "Print a saved conversation to the console")
    table.add_row("/clear", "Clear conversation history")
    table.add_row(
        "/config",
        "Show configuration; '/config save_thinking on|off' controls whether <think> blocks are saved (omit on/off to toggle)",
    )
    table.add_row("/conversations", "List saved conversations")
    table.add_row("/exit", "Quit the application")
    table.add_row("/help", "Show this help message")
    table.add_row("/info", "Show conversation and context window statistics")
    table.add_row("/load <name>", "Load a saved conversation")
    table.add_row("/read <path>", "Read a text file into the conversation")
    table.add_row("/recall <n>", "Recall message pair n into context")
    table.add_row("/retry", "Regenerate the last response")
    table.add_row("/save <name>", "Save conversation (default: timestamp)")
    table.add_row("/set", "Show model options (seed, temperature, top_p)")
    table.add_row("/set <key> <val>", "Set a model option (or 'default' to reset)")
    table.add_row("/stats", "Toggle token and context stats display")
    table.add_row(
        "/system <prompt>",
        'Set the system prompt (use """ for multiline or a path to a file within the current directory)',
    )
    table.add_row('"""', "Enter multiline input mode (or use Shift+Enter / Alt+Enter / paste)")
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


LLAMA_DEFAULTS = {
    "seed": "random",
    "temperature": 0.8,
    "top_p": 0.95,
}


def describe_think_tags(think_tags):
    """Rich-safe text for the active thinking tags, e.g. "<think> … </think> (detected)".

    Tags may contain square brackets ([THINK], [/THINK]), which Rich would parse
    as markup (an unmatched closing tag raises MarkupError), so they are escaped.
    """
    if not think_tags:
        return "none detected"
    start, end, source = think_tags
    return f"{escape(start)} … {escape(end)} ({source})"


def display_config(config, current_model, options=None, think_tags=None):
    table = Table(title="Configuration", show_header=True, header_style="bold")
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value")
    table.add_row("model", current_model)
    table.add_row("system_prompt", config["system_prompt"] or "(none)")
    table.add_row("llama_url", config["llama_url"])
    table.add_row("conversations_dir", config["conversations_dir"])
    table.add_row("save_thinking", "on" if config.get("save_thinking", True) else "off")
    table.add_row("think_tags", describe_think_tags(think_tags))
    if options is not None:
        for key in sorted(options):
            val = options[key]
            if val is not None:
                table.add_row(key, str(val))
            else:
                table.add_row(key, f"{LLAMA_DEFAULTS[key]} [dim](default)[/dim]")
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


# Drawn after the tag that closes a thinking block: one blank line before the
# rule and two after it. conv2txt.py --keep-thinking has its own THINK_RULE.
THINK_DELIMITER = "\n\n*** END OF THINKING ***\n\n\n"


def format_size(n):
    """A byte count for people: '412 B' below 1 KB, else '2.1 KB'."""
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB"


def _include_size(inc):
    return "empty" if inc.get("empty") else format_size(inc.get("bytes", 0))


def display_included_files(includes):
    """One dim line per file spliced into the message about to be sent."""
    for inc in includes:
        console.print(f"[dim]Included {escape(inc['typed'])} ({_include_size(inc)})[/dim]")


def display_cat_conversation(name, conversation, model):
    """Print a saved conversation's messages to the console.

    Thinking is left out: an assistant message prints its content only, and a
    reply that was only thinking is skipped. Message text is escaped so that
    text that merely looks like Rich markup (for example "[/THINK]" or
    "[/path]") prints literally instead of raising.
    """
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

        if msg["role"] == "assistant" and not msg["content"] and msg.get("thinking"):
            continue
        if msg["role"] == "user":
            pair_index += 1
            console.print(f"[dim]\\[{pair_index}][/dim] [user_label]You:[/user_label]{ts_display}")
        else:
            model_display = f"  [dim]{escape(msg['model'])}[/dim]" if msg.get("model") else ""
            console.print(
                f"[dim]\\[{pair_index}][/dim] [assistant_label]Assistant:[/assistant_label]"
                f"{ts_display}{model_display}"
            )
        console.print(escape(msg["content"]))
        if msg.get("includes"):
            listed = "; ".join(f"{escape(inc['typed'])}, {_include_size(inc)}" for inc in msg["includes"])
            console.print(f"[dim]\\[included: {listed}][/dim]")
        console.print()


def display_conversation_info(summary, counts=None, context_length=None):
    """The /info table. `counts` is tokens.breakdown()'s dict (None: not available)."""
    table = Table(title="Conversation Info", show_header=True, header_style="bold")
    table.add_column("Statistic", style="bold cyan")
    table.add_column("Value")
    user = summary["user_messages"]
    asst = summary["assistant_messages"]
    table.add_row("Messages", f"{summary['messages']} ({user} you, {asst} AI)")
    table.add_row("Words", f"{summary['words']:,}")
    table.add_row("Characters", f"{summary['characters']:,}")
    if summary.get("included_files"):
        table.add_row(
            "Included files", f"{summary['included_files']} ({format_size(summary['included_bytes'])})"
        )
    if counts:
        context_length = counts.get("n_ctx") or context_length
        table.add_row("Tokens: system prompt", f"{counts['system']:,}")
        table.add_row("Tokens: your messages", f"{counts['user']:,}")
        table.add_row("Tokens: AI replies", f"{counts['assistant']:,}")
        table.add_row("Tokens: template ≈", f"{counts['overhead']:,}")
        table.add_row("Prompt tokens", f"{counts['total']:,}")
    table.add_row("Context window", f"{context_length:,} tokens" if context_length else "unknown")
    if counts and context_length:
        table.add_row("Window used", f"{counts['total'] / context_length * 100:.1f}%")
    console.print(table)


def display_info(msg):
    console.print(f"[info]{msg}[/info]")


def display_error(msg):
    console.print(f"[error]{msg}[/error]")


class ThinkSeparator:
    """Streaming helper: draw THINK_DELIMITER after the tag that closes a thinking block.

    Some models (Gemma, gpt-oss) run straight from the closing tag into the
    answer (`...144.<channel|>144`); others write a blank line after it. feed()
    takes each streamed piece and returns the text to draw for it: the same
    piece, with THINK_DELIMITER inserted right after the closing tag and every "\\n"
    that directly follows the tag dropped, so the delimiter looks the same
    whatever the model wrote. The delimiter ends in newlines so the
    display loop draws it at once instead of holding the rule as a partial word.
    The tag is matched on the accumulated text, so it may arrive split across
    pieces. This is display only: the caller keeps the reply text unchanged.
    """

    def __init__(self, end):
        self.end = end
        self._tail = ""  # last len(end) - 1 characters seen
        self._pending = False  # the text so far ends at a closing tag (plus newlines): drop leading "\n"s

    def feed(self, token):
        if not self.end:
            return token
        shown = []
        cut = 0  # next index of `token` not yet copied to `shown`
        if self._pending and token:
            cut = len(token) - len(token.lstrip("\n"))  # absorbed into the delimiter
            self._pending = cut == len(token)  # all newlines: more may follow in the next piece
        window = self._tail + token
        base = len(self._tail)  # where this piece starts inside the window
        i = window.find(self.end, max(0, base - len(self.end) + 1))  # matches ending inside this piece
        while i != -1:
            end_at = i + len(self.end)
            rel = end_at - base
            shown.append(token[cut:rel])
            shown.append(THINK_DELIMITER)
            cut = rel
            run = end_at
            while run < len(window) and window[run] == "\n":
                run += 1
            cut = rel + (run - end_at)
            self._pending = run == len(window)  # newlines may continue in the next piece
            i = window.find(self.end, end_at)
        shown.append(token[cut:])
        keep = len(self.end) - 1
        self._tail = window[max(0, len(window) - keep) :] if keep > 0 else ""
        return "".join(shown)


def display_assistant_stream(token_generator, think_end=None):
    """Print streamed tokens live with word-wrap.

    think_end is the tag that closes the model's thinking block, if known; a
    delimiter line is then drawn after it (see ThinkSeparator). The returned text is
    always exactly what the model produced.

    Returns the full response text.
    """
    now_ts = datetime.now().isoformat()
    console.print(f"[assistant_label]Assistant:[/assistant_label] {now_ts}\n", end="")
    full_text = ""
    term_width = console.width or 80
    col = 0  # current column position
    word_buf = ""  # incomplete word being accumulated
    visual_lines = 0  # lines emitted (for erasure)
    separator = ThinkSeparator(think_end) if think_end else None

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
            full_text += token  # the stored reply: never altered
            word_buf += separator.feed(token) if separator else token  # what is drawn

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
    console.print(f"[warning]Warning: context window {pct:.0f}% full ({used:,} / {limit:,} tokens)[/warning]")


def display_truncated_warning(answer_missing=False):
    """Warn that the reply was stopped because the context window filled up.

    answer_missing: the reply was still inside its thinking block, so nothing of the
    answer was written (and the exchange will not be sent with later messages).
    """
    msg = "Warning: the reply was cut off because the context window is full."
    if answer_missing:
        msg += " The model was still thinking, so there is no answer."
    console.print(
        f"[warning]{msg} Free space with /clear, or restart llama-server with a larger -c.[/warning]"
    )


def display_prompt_size(needed, n_ctx):
    """A dim line with the size of the prompt about to be sent, after a file was included."""
    console.print(
        f"[dim]Prompt: {needed:,} tokens ({needed / n_ctx * 100:.0f}% of the {n_ctx:,}-token window)[/dim]"
    )


def display_stats(stats):
    """Display token generation stats in a dim line.

    "generated" is everything the model produced, thinking and answer alike; the prompt
    figure is the server's count for this request. See docs/statistics.md.
    """
    if not stats:
        return
    parts = []
    if "completion_tokens" in stats:
        parts.append(f"{stats['completion_tokens']} generated (thinking + answer)")
    if "tokens_per_second" in stats:
        parts.append(f"{stats['tokens_per_second']:.1f} tok/s")
    if "prompt_tokens" in stats:
        parts.append(f"{stats['prompt_tokens']} prompt tokens")
    if parts:
        console.print(f"[dim]  {' | '.join(parts)}[/dim]")


def init_readline(conversations_dir):
    """Load readline history from disk and configure tab-completion."""

    def completer(text, state):
        line = readline.get_line_buffer().lstrip()
        if (line.startswith("/load ") or line.startswith("/cat ")) and conversations_dir:
            names = [n for n, _ in Conversation.list_saved(conversations_dir)]
            matches = [n for n in names if n.startswith(text)]
        elif text.startswith("/"):
            matches = [c for c in COMMANDS if c.startswith(text)]
        else:
            matches = []
        return matches[state] if state < len(matches) else None

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(HISTORY_MAX)
    readline.set_completer(completer)
    readline.set_completer_delims(" ")
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set enable-bracketed-paste on")
    readline.parse_and_bind(r'"\M-\C-m": "\n"')
    readline.parse_and_bind(r'"\e[13;2u": "\n"')  # Shift+Enter — Kitty keyboard protocol
    readline.parse_and_bind(r'"\e[27;2;13~": "\n"')  # Shift+Enter — xterm modifyOtherKeys


def save_readline_history():
    """Save readline history to disk."""
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


# Readline prompt: bold green >>>, with the ANSI codes wrapped in \x01/\x02 so readline
# measures the visible width correctly.
PROMPT = "\x01\033[1;32m\x02>>> \x01\033[0m\x02"


def _discard_pending_input():
    """Drop keys typed while a reply was streaming, so they are not submitted."""
    if sys.stdin.isatty():
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except termios.error:
            pass


def _redraw_after_resume(signum, frame):
    """Ctrl-Z then fg at the prompt: draw the prompt and the line being typed again.

    Python runs readline with its own signal handling off (readline.c: rl_catch_signals = 0), and
    readline redraws incrementally, so after the shell has taken over the screen it still believes
    its prompt is visible and shows it only at the next new line. SIGWINCH cannot be used to force a
    redraw (rl_resize_terminal() redraws only if the size changed), and the readline module has no
    forced redisplay, so this writes the visible prompt and the line buffer on a cleared row. The
    cursor ends at the end of the line: if it was mid-line, later edits are drawn a few columns off.
    """
    sys.stdout.write("\r\033[K" + re.sub("[\x01\x02]", "", PROMPT) + readline.get_line_buffer())
    sys.stdout.flush()


def get_user_input():
    """Prompt the user for input. Readline reads every key itself.

    Shows: >>>
    Ctrl-C clears the current line and re-prompts.
    Returns None on EOF (Ctrl-D).
    """
    _discard_pending_input()  # once per call, so a re-prompt after Ctrl-C keeps what is typed next
    while True:
        # Only while waiting at the prompt: a SIGCONT at any other time needs no redraw.
        previous = signal.signal(signal.SIGCONT, _redraw_after_resume) or signal.SIG_DFL
        try:
            return input(PROMPT)
        except KeyboardInterrupt:
            print()
        except EOFError:
            print()
            return None
        finally:
            signal.signal(signal.SIGCONT, previous)


def _set_tty(fd, attrs):
    """Apply attrs now. A Ctrl-C that lands during the call must not skip it; any tty error is ignored."""
    while True:
        try:
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            return
        except KeyboardInterrupt:  # retry; the interrupt is dropped (at exit the turn is ending anyway;
            continue  # at entry the user has to press Ctrl-C again)
        except termios.error:  # nothing useful to do; must not replace the caller's own exception
            return


@contextlib.contextmanager
def echo_suppressed():
    """Keep the tty from echoing keys typed during a turn. Restored on every exit that runs `finally`.

    Readline entered with ECHO off draws none of the line being typed, so this must be
    left before the next prompt, never held across input().
    """
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        yield
        return
    quiet = list(saved)
    quiet[3] &= ~termios.ECHO  # c_lflag; ICANON and ISIG stay as found
    try:
        _set_tty(fd, quiet)
        yield
    finally:
        _set_tty(fd, saved)


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
