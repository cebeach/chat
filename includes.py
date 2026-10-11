"""@@<path> includes: read files and splice them into a prompt as flat text."""

from pathlib import Path
import re


# Bounds the memory a read and the /tokenize request body can take. It is not the
# context limit: whether a prompt fits the window is decided by _fits_window().
_READ_FILE_MAX_BYTES = 8 * 1024 * 1024


def _read_file(path: str) -> tuple[bool, str]:
    """Read a UTF‑8 text file of at most 8 MB.

    Returns (True, content) on success or (False, error_msg) on failure.
    """
    # Resolve the path relative to cwd, expand user (~)
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return False, "File not found."

    if resolved.stat().st_size > _READ_FILE_MAX_BYTES:
        return False, "File too large to read."

    try:
        content = resolved.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read file: {e}"
    return True, content


# `@@<path>` splices a file into a prompt; `\@@<` is a literal `@@<`.
_INCLUDE_OPEN = re.compile(r"\\@@<|@@<")
_INCLUDE_PATH = re.compile(r"[^>\n]*")


def _trailing_newlines(s):
    return len(s) - len(s.rstrip("\n"))


def _leading_newlines(s):
    return len(s) - len(s.lstrip("\n"))


def build_message(segments):
    """Read the files in segments and splice them into one flat message.

    segments is a list of ("text", str) and ("file", typed_path, shown_name)
    tuples. Returns (flat_text, includes, errors). Any error (a file that is
    missing, too large or unreadable, or a message with nothing left to send)
    leaves flat_text empty: the caller must not send it.

    A file's leading and trailing newlines are dropped and it is set off from
    adjacent text by a blank line. Newlines typed around it are never removed,
    and none are added at the very start or end of the message. A file that is
    only whitespace counts as empty: it adds no text but still separates its
    neighbours.
    """
    includes, errors, files = [], [], []
    for seg in segments:
        if seg[0] != "file":
            continue
        _, typed, shown = seg
        ok, content = _read_file(typed)
        if not ok:
            errors.append(f"{shown}: {content}")
            continue
        path = Path(typed).expanduser().resolve()
        include = {"typed": typed, "path": str(path), "bytes": path.stat().st_size}
        if not content.strip():
            include["empty"] = True
        includes.append(include)
        files.append(content)
    if errors:
        return "", [], errors

    out = ""
    after_file = False  # the last piece was a file: the next real text needs a blank line
    file_no = 0
    for i, seg in enumerate(segments):
        if seg[0] == "text":
            piece = seg[1]
            # The token and the horizontal whitespace touching it are replaced.
            if i > 0 and segments[i - 1][0] == "file":
                piece = piece.lstrip(" \t")
            if i + 1 < len(segments) and segments[i + 1][0] == "file":
                piece = piece.rstrip(" \t")
            if not piece:
                continue
            if not piece.strip():  # only newlines: keep them, add no separator
                out += piece
                continue
            if after_file and out:
                out += "\n" * max(0, 2 - _trailing_newlines(out) - _leading_newlines(piece))
            out += piece
            after_file = False
        else:
            content = files[file_no]
            file_no += 1
            if content.strip():
                if out:
                    out += "\n" * max(0, 2 - _trailing_newlines(out))
                out += content.strip("\r\n")
            after_file = True

    if not out.strip():
        names = ", ".join(seg[2] for seg in segments if seg[0] == "file")
        return "", [], [f"Nothing to send: {names} is empty."]
    return out, includes, []


def expand_includes(text):
    """Replace each `@@<path>` in text with the file's contents.

    Returns (flat_text, includes, errors); see build_message. The model gets
    only flat_text. `\\@@<` writes a literal `@@<`, and a `@@<` with no closing
    `>` on its line is an error rather than a placeholder sent as typed.
    Included files are not themselves expanded.
    """
    if "@@<" not in text:
        return text, [], []
    segments, errors, literal = [], [], []
    pos = 0
    while True:
        m = _INCLUDE_OPEN.search(text, pos)
        if m is None:
            break
        literal.append(text[pos : m.start()])
        if m.group() != "@@<":  # \@@< : keep the literal @@<
            literal.append("@@<")
            pos = m.end()
            continue
        body = _INCLUDE_PATH.match(text, m.end())
        end = body.end()
        if end < len(text) and text[end] == ">":
            typed = body.group().strip()
            if typed:
                segments.append(("text", "".join(literal)))
                literal = []
                segments.append(("file", typed, f"@@<{typed}>"))
            else:
                errors.append("@@<>: no file name between the brackets.")
            pos = end + 1
        else:
            errors.append(f"Unterminated include: @@<{body.group()} (write \\@@< for a literal @@<)")
            pos = end
    literal.append(text[pos:])
    segments.append(("text", "".join(literal)))
    if errors:
        # Still report any missing files, so one fix pass finds everything.
        _, _, more = build_message(segments)
        return "", [], errors + [e for e in more if not e.startswith("Nothing to send")]
    return build_message(segments)
