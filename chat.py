#!/usr/bin/env python3
"""AI Chat — a terminal chat application powered by Ollama."""

import argparse
import sys
from datetime import datetime

from requests.exceptions import ConnectionError, HTTPError

from config import load_config
from conversation import Conversation
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
    set_conversations_dir,
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
            display_models(models, state["model"])
        except (ConnectionError, HTTPError) as e:
            display_error(f"Failed to list models: {e}")

    elif cmd == "/model":
        if not args:
            display_info(f"Current model: {state['model']}")
        else:
            new_model = args
            state["model"] = new_model
            state["context_length"] = client.get_context_length(new_model)
            display_info(f"Switched to model: {new_model}")

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

    elif cmd == "/cat":
        name = args.strip()
        if not name:
            display_error("Usage: /cat <name>")
        else:
            try:
                conv_dir = state["config"]["conversations_dir"]
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
        conv_dir = state["config"]["conversations_dir"]
        conversations = Conversation.list_saved(conv_dir)
        display_conversations(conversations)

    elif cmd == "/stats":
        state["show_stats"] = not state.get("show_stats", False)
        status = "on" if state["show_stats"] else "off"
        display_info(f"Stats display: {status}")

    elif cmd == "/set":
        option_keys = {"seed": int, "temperature": float, "top_p": float}
        if not args:
            display_options(state["options"])
        else:
            parts = args.strip().split(None, 1)
            key = parts[0]
            if key not in option_keys:
                display_error(
                    f"Unknown option: {key}. "
                    f"Available: {', '.join(sorted(option_keys))}"
                )
            elif len(parts) < 2:
                val = state["options"][key]
                display_info(f"{key}: {val if val is not None else 'default'}")
            else:
                raw = parts[1]
                if raw == "default":
                    state["options"][key] = None
                    display_info(f"{key} reset to default.")
                else:
                    try:
                        state["options"][key] = option_keys[key](raw)
                        display_info(f"{key} set to {state['options'][key]}.")
                    except ValueError:
                        expected = option_keys[key].__name__
                        display_error(f"{key} must be {expected} (or 'default').")

    elif cmd == "/retry":
        if len(conversation.messages) < 2:
            display_error("Nothing to retry (need at least one exchange).")
        elif conversation.messages[-1]["role"] != "assistant":
            display_error("Last message is not an assistant response.")
        else:
            conversation.messages.pop()  # remove assistant
            retry_text = conversation.messages[-1]["content"]
            conversation.messages.pop()  # remove user (REPL will re-add)
            state["retry_text"] = retry_text

    elif cmd == "/config":
        display_config(state["config"], state["model"], state["options"])

    elif cmd == "/info":
        display_conversation_info(conversation.summary(), state.get("last_stats"))

    else:
        display_error(f"Unknown command: {cmd}. Type /? for available commands.")

    return True


def _auto_save(conversation, state):
    """Silently auto-save the conversation if enabled."""
    if not state["config"].get("auto_save", True):
        return
    if not conversation.messages:
        return
    try:
        conv_dir = state["config"]["conversations_dir"]
        conversation.save(conv_dir, name=state["auto_save_name"], model=state["model"])
    except OSError:
        pass


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

    options = {
        "seed": config["seed"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
    }
    auto_save_name = "auto_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    context_length = client.get_context_length(model)
    state = {
        "model": model,
        "config": config,
        "show_stats": False,
        "options": options,
        "auto_save_name": auto_save_name,
        "context_length": context_length,
    }
    conversation = Conversation(system_prompt=config["system_prompt"])

    init_readline()
    set_conversations_dir(config["conversations_dir"])
    print_welcome(model)

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
                if "retry_text" in state:
                    text = state.pop("retry_text")
                else:
                    continue

            # Send message to model
            conversation.add_user(text)
            try:
                chat_stream = client.chat(
                    model=state["model"],
                    messages=conversation.get_messages(),
                    options=state["options"],
                )
                response = display_assistant_stream(chat_stream)
                conversation.add_assistant(response)
                state["last_stats"] = chat_stream.stats
                if state["show_stats"]:
                    display_stats(chat_stream.stats)
                # Context window warning
                ctx_len = state.get("context_length")
                prompt_tokens = chat_stream.stats.get("prompt_eval_count", 0)
                if ctx_len and prompt_tokens > 0.8 * ctx_len:
                    display_context_warning(prompt_tokens, ctx_len)
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
