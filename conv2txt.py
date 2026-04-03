#!/usr/bin/env python3
"""Convert a conversation JSON file to plain text.

Usage:
    python conv2txt.py conversation.json
    python conv2txt.py conversation.json -o output.txt
    python conv2txt.py conversation.json --no-header
    python conv2txt.py conversation.json -l 80

This script now supports a ``--line-length`` option that controls the maximum
number of characters per line in the output.  All output—including headers,
separator lines and message bodies—is wrapped at the specified width.  Long
words are left intact and may exceed the width if they are longer than the
specified limit.  Double newlines are treated as paragraph separators; single
newlines within a message are wrapped like normal text.  Separator lines that
consist solely of a single repeated character are truncated only when they
exceed the requested width.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import textwrap

# ---------------------------------------------------------------------------
# Helper for wrapping blocks of text
# ---------------------------------------------------------------------------


def wrap_block(text: str, width: int) -> str:
    """Wrap a block of text while preserving paragraph breaks.

    * Paragraphs are separated by **exactly** two consecutive newlines.
    * Within a paragraph, single newlines are treated as normal line breaks
      and may be wrapped.
    * Lines that consist solely of a single repeated character (e.g.
      ``-----`` or ``====``) are truncated to ``width`` characters only if
      they exceed that width.
    * Long words longer than ``width`` are left unbroken.
    * Leading and trailing whitespace for each line is removed.
    * Internal whitespace is preserved.
    * The result ends with a single newline.
    """
    # Strip whitespace around the whole block – this removes a trailing
    # newline that callers might have added.
    text = text.strip()
    if not text:
        return ""

    # Split on two consecutive newlines to get paragraphs.
    paragraphs = text.split("\n\n")
    wrapped_paragraphs = []

    for para in paragraphs:
        lines = para.split("\n")
        wrapped_lines = []
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                # Preserve empty lines within a paragraph as a single
                # blank line; they will be joined later.
                wrapped_lines.append("")
                continue
            # Handle separator lines consisting of a single repeated char.
            if len(set(stripped)) == 1 and len(stripped) > width:
                stripped = stripped[:width]
            # Wrap the line; do not break long words.
            wrapped = textwrap.wrap(
                stripped,
                width=width,
                break_long_words=False,
                replace_whitespace=False,
            )
            wrapped_lines.extend(wrapped)
        wrapped_paragraphs.append("\n".join(wrapped_lines))

    # Join paragraphs with two newlines (preserving the original separation).
    return "\n\n".join(wrapped_paragraphs) + "\n"


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------


def convert(data: dict, header: bool = True, line_length: int = 110) -> str:
    """Convert a conversation dict to plain text lines.

    Parameters
    ----------
    data: dict
        The parsed conversation JSON.
    header: bool
        Whether to include the model/system prompt header.
    line_length: int
        Maximum characters per line.
    """
    lines: list[str] = []

    # Header block – wrapped as a single block.
    if header:
        header_lines = []
        model = data.get("model", "")
        system_prompt = data.get("system_prompt", "")
        if model:
            header_lines.append(f"Model: {model}")
        if system_prompt:
            header_lines.append(f"System prompt: {system_prompt}")
            # Add source file if present
            source_file = data.get("source_file")
            if source_file:
                header_lines.append(f"from: {source_file}")
        if header_lines:
            # Add the separator line of 60 dashes.
            header_lines.append("-" * 60)
            # Join with double newline to match original formatting.
            header_block = "\n\n".join(header_lines)
            wrapped_header = wrap_block(header_block, line_length)
            lines.append(wrapped_header.rstrip("\n"))

    # Message bodies
    messages = data.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")
        source_file = msg.get("source_file")

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

        # Add source file annotation if present
        if source_file:
            lines.append(f"[from: {source_file}]")

        # Wrap the content of the message.
        wrapped_content = wrap_block(content, line_length)
        if wrapped_content:
            lines.append(wrapped_content.rstrip("\n"))

        if i < len(messages) - 1:
            lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a conversation JSON file to plain text")
    parser.add_argument("input", help="Path to conversation .json file")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit model and system prompt header",
    )
    parser.add_argument(
        "-l",
        "--line-length",
        type=int,
        default=110,
        help="Maximum characters per line (default: 110)",
    )
    args = parser.parse_args()

    if args.line_length < 1:
        parser.error("--line-length must be a positive integer")

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

    text = convert(data, header=not args.no_header, line_length=args.line_length)

    # Ensure the text ends with a newline
    if not text.endswith("\n"):
        text += "\n"

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(text)
        print(f"Written to {output_path}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
