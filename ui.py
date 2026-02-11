import readline
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.theme import Theme

HISTORY_FILE = Path.home() / ".local" / "share" / "chat" / "history"
HISTORY_MAX = 1000

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
    console.print("Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.")
    console.print()


def print_help():
    table = Table(title="Commands", show_header=True, header_style="bold")
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")
    table.add_row("/help", "Show this help message")
    table.add_row("/exit", "Quit the application")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/models", "List available models")
    table.add_row("/model <name>", "Switch to a different model")
    table.add_row("/system <prompt>", "Set the system prompt")
    table.add_row("/save [name]", "Save conversation (default: timestamp)")
    table.add_row("/load <name>", "Load a saved conversation")
    table.add_row("/conversations", "List saved conversations")
    table.add_row("/stats", "Toggle token stats display")
    table.add_row("/config", "Show current configuration")
    table.add_row('"""', "Enter multiline input mode")
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


def display_config(config, current_model):
    table = Table(title="Configuration", show_header=True, header_style="bold")
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value")
    table.add_row("model", current_model)
    table.add_row("default_model", config["default_model"] or "(none)")
    table.add_row("system_prompt", config["system_prompt"] or "(none)")
    table.add_row("ollama_url", config["ollama_url"])
    table.add_row("conversations_dir", config["conversations_dir"])
    console.print(table)


def display_info(msg):
    console.print(f"[info]{msg}[/info]")


def display_error(msg):
    console.print(f"[error]{msg}[/error]")


def display_assistant_stream(token_generator):
    """Print streamed tokens live, then re-render as markdown.

    Returns the full response text.
    """
    console.print("[assistant_label]Assistant:[/assistant_label]")
    full_text = ""
    try:
        for token in token_generator:
            sys.stdout.write(token)
            sys.stdout.flush()
            full_text += token
    except KeyboardInterrupt:
        full_text += " [interrupted]"
    finally:
        # Clear the raw streamed output and re-render as markdown
        # Move cursor up and clear each line we printed
        raw_lines = full_text.count("\n") + 1
        for _ in range(raw_lines):
            sys.stdout.write("\033[A\033[2K")
        sys.stdout.flush()
        console.print(Markdown(full_text))

    return full_text


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


def init_readline():
    """Load readline history from disk."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(HISTORY_MAX)


def save_readline_history():
    """Save readline history to disk."""
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


def get_user_input():
    """Prompt the user for input. Returns None on EOF."""
    try:
        console.print("[user_label]You:[/user_label]", end=" ")
        return input()
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
