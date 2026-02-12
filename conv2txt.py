#!/usr/bin/env python3
"""Convert a conversation JSON file to plain text.

Usage:
    python conv2txt.py conversation.json
    python conv2txt.py conversation.json -o output.txt
    python conv2txt.py conversation.json --no-header
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def convert(data, header=True):
    """Convert a conversation dict to plain text lines."""
    lines = []

    if header:
        model = data.get("model", "")
        system_prompt = data.get("system_prompt", "")
        if model:
            lines.append(f"Model: {model}")
        if system_prompt:
            lines.append(f"System prompt: {system_prompt}")
        if lines:
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

    messages = data.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")

        ts_suffix = ""
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                ts_suffix = f" ({dt.strftime('%Y-%m-%d %H:%M:%S')})"
            except (ValueError, TypeError):
                pass

        if role == "user":
            lines.append(f"You{ts_suffix}:")
        elif role == "assistant":
            lines.append(f"Assistant{ts_suffix}:")
        else:
            lines.append(f"{role.capitalize()}{ts_suffix}:")

        lines.append(content)

        if i < len(messages) - 1:
            lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Convert a conversation JSON file to plain text"
    )
    parser.add_argument("input", help="Path to conversation .json file")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit model and system prompt header",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    text = convert(data, header=not args.no_header)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(text)
        print(f"Written to {output_path}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
