#!/usr/bin/env python3
"""AI Chat — a terminal chat application powered by Ollama."""

import argparse
from pathlib import Path
import sys
from dataclasses import dataclass, field
from datetime import datetime

from requests.exceptions import ConnectionError, HTTPError

from config import load_config
from conversation import Conversation
from llama_client import LlamaClient
from ollama_client import OllamaClient
from ui import (
    console,
    display_assistant_stream,
    display_cat_conversation,
    display_config,
    display_context_warning,
    display_conversation_info,
    display_conversations,
    display_error,
    display_info,
    display_models,
    display_options,
    display_stats,
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
    last_stats: dict = field(default_factory=dict)
    retry_text: str | None = None
    auto_save_name: str = ""


def parse_args(config):
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model")
    parser.add_argument(
        "--model",
        "-m",
        default=config["default_model"] or None,
        help="Model to use (defaults to first available)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "llama"],
        default=config["backend"],
        help="Backend to use: ollama or llama (default: %(default)s)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override server URL",
    )
    return parser.parse_args()


_OPTION_KEYS = {"seed": int, "temperature": float, "top_p": float}


def _read_file(path: str, config: dict) -> tuple[bool, str]:
    """Read a UTF‑8 text file with size limit.

    Returns (True, content) on success or (False, error_msg) on failure.
    """
    # Resolve the path relative to cwd, expand user (~)
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return False, "File not found."

    # Enforce size limit from config
    limit_kb = config.get("read_file_max_kb", 32)
    if resolved.stat().st_size > limit_kb * 1024:
        return False, f"File too large ({limit_kb} KB max)."

    try:
        content = resolved.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read file: {e}"
    return True, content


def _handle_set(args, state):
    """Handle the /set command: list, query, or modify a model option."""
    if not args:
        display_options(state.options)
        return
    parts = args.strip().split(None, 1)
    key = parts[0]
    if key not in _OPTION_KEYS:
        display_error(f"Unknown option: {key}. Available: {', '.join(sorted(_OPTION_KEYS))}")
    elif len(parts) < 2:
        val = state.options[key]
        display_info(f"{key}: {val if val is not None else 'default'}")
    elif parts[1] == "default":
        state.options[key] = None
        display_info(f"{key} reset to default.")
    else:
        try:
            state.options[key] = _OPTION_KEYS[key](parts[1])
            display_info(f"{key} set to {state.options[key]}.")
        except ValueError:
            display_error(f"{key} must be {_OPTION_KEYS[key].__name__} (or 'default').")


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

    elif cmd == "/models":
        try:
            models = client.list_models()
            display_models(models, state.model)
        except (ConnectionError, HTTPError) as e:
            display_error(f"Failed to list models: {e}")

    elif cmd == "/model":
        if not args:
            display_info(f"Current model: {state.model}")
        else:
            state.model = args
            state.context_length = client.get_context_length(args)
            display_info(f"Switched to model: {args}")

    elif cmd == "/system":
        if not args:
            current = conversation.system_prompt or "(none)"
            display_info(f"Current system prompt: {current}")
        elif args.strip() == '"""':
            text = get_multiline_input()
            if text is not None:
                conversation.system_prompt = text
                display_info("System prompt set.")
        else:
            conversation.system_prompt = args
            display_info("System prompt set.")

    elif cmd == "/save":
        name = args.strip() or None
        try:
            conv_dir = state.config["conversations_dir"]
            filepath = conversation.save(conv_dir, name=name, model=state.model)
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
                conversation.messages = loaded_conv.messages
                conversation.system_prompt = loaded_conv.system_prompt
                if loaded_model:
                    state.model = loaded_model
                display_info(
                    f"Loaded conversation: {name} "
                    f"({len(conversation.messages)} messages, model: {state.model})"
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
        _handle_set(args, state)

    elif cmd == "/retry":
        if len(conversation.messages) < 2:
            display_error("Nothing to retry (need at least one exchange).")
        elif conversation.messages[-1]["role"] != "assistant":
            display_error("Last message is not an assistant response.")
        else:
            conversation.messages.pop()  # remove assistant
            state.retry_text = conversation.messages[-1]["content"]
            conversation.messages.pop()  # remove user (REPL will re-add)

    elif cmd == "/read":
        if not args:
            display_error("Usage: /read <path>")
            return True
        ok, result = _read_file(args, state.config)
        if ok:
            # Store content to be sent on the next loop iteration
            state.retry_text = result
            console.print("User:")
            console.print(result)
        else:
            display_error(result)
        return True

    elif cmd == "/config":
        display_config(state.config, state.model, state.options)

    elif cmd == "/info":
        display_conversation_info(conversation.summary(), state.last_stats)

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
        conversation.save(conv_dir, name=state.auto_save_name, model=state.model)
    except OSError:
        pass


def main():
    config = load_config()
    args = parse_args(config)

    if args.backend == "llama":
        url = args.url or config["llama_url"]
        client = LlamaClient(url)
        unavailable_msg = (
            "Cannot connect to llama-server. Make sure it's running with: llama-server --port 8001 -m <model>"
        )
    else:
        url = args.url or config["ollama_url"]
        client = OllamaClient(url)
        unavailable_msg = "Cannot connect to Ollama. Make sure it's running with: ollama serve"

    if not client.is_available():
        display_error(unavailable_msg)
        sys.exit(1)

    # Resolve model
    model = args.model
    if not model:
        try:
            models = client.list_models()
        except (ConnectionError, HTTPError) as e:
            display_error(f"Failed to list models: {e}")
            sys.exit(1)

        if not models:
            if args.backend == "llama":
                display_error("No models found. Make sure llama-server is loaded with a model.")
            else:
                display_error("No models found. Pull one first with: ollama pull <model>")
            sys.exit(1)
        model = models[0]

    state = State(
        model=model,
        config=config,
        context_length=client.get_context_length(model),
        options={
            "seed": config["seed"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
        },
        auto_save_name="auto_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    conversation = Conversation(system_prompt=config["system_prompt"])

    init_readline(config["conversations_dir"])
    print_welcome(model, args.backend)

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
            if text.startswith("/"):
                parts = text.split(None, 1)
                cmd = parts[0].lower()
                cmd_args = parts[1] if len(parts) > 1 else ""
                if not handle_command(cmd, cmd_args, client, conversation, state):
                    break
                # Check if /retry set text to re-send
                if state.retry_text is not None:
                    text = state.retry_text
                    state.retry_text = None
                else:
                    continue

            # Send message to model
            conversation.add_user(text)
            try:
                chat_stream = client.chat(
                    model=state.model,
                    messages=conversation.get_messages(),
                    options=state.options,
                )
                response = display_assistant_stream(chat_stream)
                conversation.add_assistant(response)
                state.last_stats = chat_stream.stats
                if state.show_stats:
                    display_stats(chat_stream.stats)
                # Context window warning
                prompt_tokens = chat_stream.stats.get("prompt_eval_count", 0)
                if state.context_length and prompt_tokens > 0.8 * state.context_length:
                    display_context_warning(prompt_tokens, state.context_length)
                _auto_save(conversation, state)
            except KeyboardInterrupt:
                console.print()
                display_info("Response interrupted.")
            except ConnectionError:
                display_error("Lost connection to Ollama. Is it still running?")
                # Remove the unanswered user message
                conversation.messages.pop()
            except HTTPError as e:
                display_error(f"Ollama error: {e}")
                conversation.messages.pop()

            console.print()
    finally:
        _auto_save(conversation, state)
        save_readline_history()


if __name__ == "__main__":
    main()
