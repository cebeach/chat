"""Configuration file support for AI Chat.

Loads settings from ~/.config/chat/config.toml (TOML format).
CLI arguments override config file values.
"""

from pathlib import Path

import tomllib

import duel

CONFIG_DIR = Path.home() / ".config" / "chat"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULTS = {
    "system_prompt": "",
    "llama_url": "http://127.0.0.1:8001",
    "conversations_dir": str(Path.home() / ".local" / "share" / "chat" / "conversations"),
    # Where named projects live (see docs/projects.md).
    "projects_dir": str(Path.home() / ".local" / "share" / "chat" / "projects"),
    "auto_save": True,
    "save_thinking": True,
    # Optional override of the detected thinking tags (both must be non-empty).
    "think_start": "",
    "think_end": "",
    "seed": None,
    "temperature": None,
    "top_p": None,
    "min_p": None,
    "repeat_penalty": None,
    "n_predict": None,
    # Price every prompt against the context window before sending (see docs/statistics.md).
    "context_check": True,
    # Tokens kept free for the reply when deciding whether a prompt fits; 0 keeps none.
    "reserve_output_tokens": 0,
    # /duel (see docs/duel.md). The temperature and seed of a side are unset unless given.
    "duel_axes": list(duel.DEFAULT_AXES),
    "duel_stance_a": duel.DEFAULT_STANCE_A,
    "duel_stance_b": duel.DEFAULT_STANCE_B,
    "duel_temperature_a": None,
    "duel_temperature_b": None,
    "duel_seed_a": None,
    "duel_seed_b": None,
    # "shared": the two sides share one KV pool of the reported size; "split": each gets that size.
    "duel_context": "shared",
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


def apply_project(config, settings):
    """A new config: `config` with a project's settings laid over it (`config` is not changed)."""
    merged = dict(config)
    merged.update(settings)
    return merged
