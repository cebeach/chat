#!/usr/bin/env python3
"""AI Chat — a terminal chat application powered by Ollama."""

import argparse
import sys

from requests.exceptions import ConnectionError, HTTPError

from config import load_config
from conversation import Conversation
from ollama_client import OllamaClient
from ui import (
    console,
    display_assistant_stream,
    display_config,
    display_conversations,
    display_error,
    display_info,
    display_models,
    display_stats,
    get_multiline_input,
    get_user_input,
    init_readline,
    print_help,
    print_welcome,
    save_readline_history,
)


def parse_args(config):
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model")
    parser.add_argument(
        "--model",
        "-m",
        default=config["default_model"] or None,
        help="Model to use (defaults to first available)",
    )
    parser.add_argument(
        "--url",
        default=config["ollama_url"],
        help="Ollama API base URL (default: %(default)s)",
    )
    return parser.parse_args()


def handle_command(cmd, args, client, conversation, state):
    """Handle a slash command. Returns True if the REPL should continue."""
    if cmd == "/help":
        print_help()

    elif cmd == "/exit":
        display_info("Goodbye!")
        return False

    elif cmd == "/clear":
        conversation.clear()
        display_info("Conversation cleared.")

    elif cmd == "/models":
        try:
            models = client.list_models()
            display_models(models, state["model"])
        except (ConnectionError, HTTPError) as e:
            display_error(f"Failed to list models: {e}")

    elif cmd == "/model":
        if not args:
            display_info(f"Current model: {state['model']}")
        else:
            new_model = args
            state["model"] = new_model
            display_info(f"Switched to model: {new_model}")

    elif cmd == "/system":
        if not args:
            current = conversation.system_prompt or "(none)"
            display_info(f"Current system prompt: {current}")
        else:
            conversation.system_prompt = args
            display_info(f"System prompt set.")

    elif cmd == "/save":
        name = args.strip() or None
        try:
            conv_dir = state["config"]["conversations_dir"]
            filepath = conversation.save(conv_dir, name=name, model=state["model"])
            display_info(f"Conversation saved: {filepath}")
        except OSError as e:
            display_error(f"Failed to save: {e}")

    elif cmd == "/load":
        name = args.strip()
        if not name:
            display_error("Usage: /load <name>")
        else:
            try:
                conv_dir = state["config"]["conversations_dir"]
                loaded_conv, loaded_model = Conversation.load(conv_dir, name)
                conversation.messages = loaded_conv.messages
                conversation.system_prompt = loaded_conv.system_prompt
                if loaded_model:
                    state["model"] = loaded_model
                display_info(
                    f"Loaded conversation: {name} "
                    f"({len(conversation.messages)} messages, model: {state['model']})"
                )
            except FileNotFoundError:
                display_error(f"No saved conversation named '{name}'.")
            except Exception as e:
                display_error(f"Failed to load: {e}")

    elif cmd == "/conversations":
        conv_dir = state["config"]["conversations_dir"]
        conversations = Conversation.list_saved(conv_dir)
        display_conversations(conversations)

    elif cmd == "/stats":
        state["show_stats"] = not state.get("show_stats", False)
        status = "on" if state["show_stats"] else "off"
        display_info(f"Stats display: {status}")

    elif cmd == "/config":
        display_config(state["config"], state["model"])

    else:
        display_error(f"Unknown command: {cmd}. Type /help for available commands.")

    return True


def main():
    config = load_config()
    args = parse_args(config)
    client = OllamaClient(args.url)

    # Check Ollama availability
    if not client.is_available():
        display_error(
            "Cannot connect to Ollama. Make sure it's running with: ollama serve"
        )
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
            display_error("No models found. Pull one first with: ollama pull <model>")
            sys.exit(1)
        model = models[0]

    state = {"model": model, "config": config, "show_stats": False}
    conversation = Conversation(system_prompt=config["system_prompt"])

    init_readline()
    print_welcome(model)

    # Main REPL
    try:
        while True:
            try:
                user_input = get_user_input()
            except KeyboardInterrupt:
                console.print()
                display_info("Goodbye!")
                break

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
                continue

            # Send message to model
            conversation.add_user(text)
            try:
                chat_stream = client.chat(
                    model=state["model"],
                    messages=conversation.get_messages(),
                )
                response = display_assistant_stream(chat_stream)
                conversation.add_assistant(response)
                if state["show_stats"]:
                    display_stats(chat_stream.stats)
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
        save_readline_history()


if __name__ == "__main__":
    main()
