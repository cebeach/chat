"""Configuration file support for AI Chat.

Loads settings from ~/.config/chat/config.toml (TOML format).
CLI arguments override config file values.
"""

from pathlib import Path

import tomllib

CONFIG_DIR = Path.home() / ".config" / "chat"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULTS = {
    "system_prompt": "",
    "llama_url": "http://127.0.0.1:8001",
    "conversations_dir": str(Path.home() / ".local" / "share" / "chat" / "conversations"),
    "auto_save": True,
    "save_thinking": True,
    # Optional override of the detected thinking tags (both must be non-empty).
    "think_start": "",
    "think_end": "",
    "seed": None,
    "temperature": None,
    "top_p": None,
    # Price every prompt against the context window before sending (see docs/statistics.md).
    "context_check": True,
    # Tokens kept free for the reply when deciding whether a prompt fits; 0 keeps none.
    "reserve_output_tokens": 0,
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
        config.update({k: file_config[k] for k in DEFAULTS if k in file_config})

    return config
