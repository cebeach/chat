"""Configuration file support for AI Chat.

Loads settings from ~/.config/chat/config.toml (TOML format).
CLI arguments override config file values.
"""

from pathlib import Path

import tomllib

CONFIG_DIR = Path.home() / ".config" / "chat"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULTS = {
    "default_model": "",
    "system_prompt": "",
    "ollama_url": "http://localhost:11434",
    "conversations_dir": str(
        Path.home() / ".local" / "share" / "chat" / "conversations"
    ),
    "auto_save": True,
    "seed": None,
    "temperature": None,
    "top_p": None,
}


def load_config():
    """Load config from TOML file, merged with defaults.

    Returns a dict with all config keys guaranteed present.
    Missing file or keys silently fall back to defaults.
    """
    config = dict(DEFAULTS)

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            file_config = tomllib.load(f)
        for key in DEFAULTS:
            if key in file_config:
                config[key] = file_config[key]

    return config
