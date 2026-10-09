"""Named projects: a directory with its own conversations, system prompt and notes.

Pure logic and file I/O (no UI, no HTTP), so it can be tested with a temp directory.
See docs/projects.md.

    <projects_dir>/<name>/
        project.toml     optional; the keys in ALLOWED_SETTINGS
        project.md       standing notes, injected into every prompt
        system.md        the project's system prompt (empty: the global one applies)
        conversations/   saved conversations, same format as the global directory
"""

import difflib
import hashlib
import os
import re
import tempfile
import tomllib
from pathlib import Path

from conversation import split_think

# The settings a project may override: the sampling options and the two token-budget keys.
ALLOWED_SETTINGS = (
    "seed",
    "temperature",
    "top_p",
    "min_p",
    "repeat_penalty",
    "n_predict",
    "context_check",
    "reserve_output_tokens",
)

_NAME = re.compile(r"[A-Za-z0-9_-]+")


def check_name(name):
    """`name` unchanged, or ValueError when it is not alphanumerics, '-' and '_' only."""
    if not _NAME.fullmatch(name or ""):
        raise ValueError(f"Invalid project name '{name}': use letters, digits, '-' and '_' only.")
    return name


def project_dir(projects_dir, name):
    """The directory of project `name`; ValueError for an invalid name. It need not exist."""
    return Path(projects_dir).expanduser() / check_name(name)


def list_projects(projects_dir):
    """Names of the existing projects with a valid name, newest first."""
    root = Path(projects_dir).expanduser()
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir() if p.is_dir() and _NAME.fullmatch(p.name)]
    return [p.name for p in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)]


def create_project(projects_dir, name):
    """Create project `name` with empty project.md and system.md; return its directory.

    Raises ValueError for an invalid name and FileExistsError when it exists.
    """
    path = project_dir(projects_dir, name)
    if path.exists():
        raise FileExistsError(f"Project '{name}' already exists.")
    (path / "conversations").mkdir(parents=True)
    (path / "project.md").write_text("", encoding="utf-8")
    (path / "system.md").write_text("", encoding="utf-8")
    return path


def conversations_path(path):
    return Path(path) / "conversations"


def load_settings(path):
    """(settings, ignored): the allowed keys of project.toml and the keys that were dropped.

    A missing file gives ({}, []). `ignored` lists known-removed keys the user should
    move (currently system_prompt). Raises ValueError naming the file when it is not
    valid TOML.
    """
    file = Path(path) / "project.toml"
    if not file.is_file():
        return {}, []
    try:
        with open(file, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ValueError(f"{file}: {exc}") from exc
    settings = {k: data[k] for k in ALLOWED_SETTINGS if k in data}
    if "context_check" in settings and not isinstance(settings["context_check"], bool):
        raise ValueError(f"{file}: context_check must be true or false")
    reserve = settings.get("reserve_output_tokens", 0)
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ValueError(f"{file}: reserve_output_tokens must be an integer >= 0")
    return settings, [k for k in ("system_prompt",) if k in data]


def _read(file):
    try:
        return Path(file).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def read_notes(path):
    """The text of project.md; "" when missing. OSError and UnicodeDecodeError propagate."""
    return _read(Path(path) / "project.md")


def read_system_prompt(path):
    """The text of system.md; "" when missing or blank."""
    text = _read(Path(path) / "system.md")
    return text if text.strip() else ""


def write_notes(path, text):
    """Replace project.md atomically: a crash never leaves a half-written file."""
    target = Path(path) / "project.md"
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".project.md.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def notes_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def diff_text(old, new):
    """Unified diff of the notes, "" when they are equal."""
    lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="project.md",
        tofile="project.md (new)",
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


_REMEMBER_SYSTEM = (
    "You maintain a markdown file of project notes. The user gives you the current file and a "
    "new note. Merge the note into the right section; create a section if none fits. Keep every "
    "existing line unless the note contradicts it, and change nothing else. Output only the "
    "complete new file, with no explanation and no code fence."
)


def remember_messages(notes, note):
    """The two-message prompt that asks the model to merge `note` into `notes`."""
    return [
        {"role": "system", "content": _REMEMBER_SYSTEM},
        {
            "role": "user",
            "content": f"Current file:\n<<<\n{notes}\n>>>\n\nNew note:\n{note}",
        },
    ]


def clean_reply(text, tags):
    """The new file text from a model reply: thinking and one surrounding code fence removed."""
    _, text = split_think(text, tags)
    text = text.strip("\n")
    lines = text.split("\n")
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1])
    return text.strip("\n") + "\n" if text.strip() else ""


# A reply shorter than this share of the old file is taken for a model that lost the notes.
SHRINK_LIMIT = 0.5


def shrank_too_much(old, new):
    """True when `new` is under half the characters of a non-trivial `old`."""
    return bool(old.strip()) and len(new.strip()) < SHRINK_LIMIT * len(old.strip())
