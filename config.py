"""Configuration file support for AI Chat.

Loads settings from ~/.config/chat/config.toml (TOML format).
CLI arguments override config file values.
"""

import difflib
import math
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
    # Where llama-server's logs are, to show load-time VRAM in /info (see docs/server-log.md);
    # empty: off. server_log names one log file instead of a directory.
    "server_log_dir": "",
    "server_log": "",
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


def normalize_key(key):
    """`key` as a config key is spelled: lower case, '-' as '_'."""
    return key.lower().replace("-", "_")


def suggest_key(key, candidates):
    """The candidate closest to `key` (a hyphen or case slip is an exact match), or None."""
    close = difflib.get_close_matches(normalize_key(key), candidates, n=1)
    return close[0] if close else None


def unknown_keys(file_config):
    """[(key, suggestion or None)] for each key of a parsed config file that is not in DEFAULTS."""
    return [(key, suggest_key(key, DEFAULTS)) for key in file_config if key not in DEFAULTS]


def _is_bool(value):
    return isinstance(value, bool)


def _is_str(value):
    return isinstance(value, str)


def _is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# The keys whose value is checked here, by type; reserve_output_tokens also has to be >= 0.
# The sampling options and the duel_* keys are not listed: option_overrides and
# duel_settings below check and report those (they carry range rules), so no key is
# checked in two places.
VALUE_RULES = {
    "system_prompt": (_is_str, "must be a string"),
    "llama_url": (_is_str, "must be a string"),
    "conversations_dir": (_is_str, "must be a string"),
    "projects_dir": (_is_str, "must be a string"),
    "server_log_dir": (_is_str, "must be a string"),
    "server_log": (_is_str, "must be a string"),
    "auto_save": (_is_bool, "must be true or false"),
    "save_thinking": (_is_bool, "must be true or false"),
    "think_start": (_is_str, "must be a string"),
    "think_end": (_is_str, "must be a string"),
    "context_check": (_is_bool, "must be true or false"),
    "reserve_output_tokens": (_is_count, "must be an integer >= 0"),
}


def wrong_values(data, source):
    """(clean, lines): `data` without the entries of VALUE_RULES whose value is the wrong type,
    and one printable line per dropped entry, in file order. Other keys pass through."""
    clean = {}
    lines = []
    for key, value in data.items():
        accepts, description = VALUE_RULES.get(key, (None, None))
        if accepts is None or accepts(value):
            clean[key] = value
        else:
            lines.append(f"{source}: {key} {description}; ignored.")
    return clean, lines


def load_config_checked():
    """(config, problems): the config as load_config() returns it, and one printable line per
    unknown key and per wrong-typed value in the file (each is ignored)."""
    config = dict(DEFAULTS)
    problems = []

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            file_config = tomllib.load(f)
        known, wrong = wrong_values({k: v for k, v in file_config.items() if k in DEFAULTS}, CONFIG_FILE.name)
        config.update(known)
        problems.extend(wrong)
        for key, suggestion in unknown_keys(file_config):
            hint = f'; did you mean "{suggestion}"?' if suggestion else "."
            problems.append(f'{CONFIG_FILE.name}: unknown key "{key}" is ignored{hint}')

    return config, problems


def load_config():
    """Load config from TOML file, merged with defaults.

    Returns a dict with all config keys guaranteed present.
    Missing file or keys silently fall back to defaults; load_config_checked() also
    reports the unknown keys and the wrong-typed values (which fall back too).
    """
    return load_config_checked()[0]


# The sampling options a session can override: the type /set parses and what the server
# accepts. The server clamps most out-of-range values silently and answers a few with
# HTTP 400, so the app checks first and refuses what it would otherwise get wrong.
OPTION_KEYS = {
    "seed": int,
    "temperature": float,
    "top_p": float,
    "min_p": float,
    "repeat_penalty": float,
    "n_predict": int,
}
OPTION_RULES = {
    "seed": (lambda v: -1 <= v <= 0xFFFFFFFF, "an integer from -1 (random) to 4294967295"),
    "temperature": (lambda v: v >= 0, "a number >= 0 (0 is greedy)"),
    "top_p": (lambda v: 0 <= v <= 1, "a number from 0 to 1 (1 disables it)"),
    "min_p": (lambda v: 0 <= v <= 1, "a number from 0 to 1 (0 disables it)"),
    "repeat_penalty": (lambda v: v > 0, "a number > 0 (1 disables it)"),
    "n_predict": (
        lambda v: v >= 1,
        "an integer >= 1 (`/set n_predict default` returns to the server's limit)",
    ),
}


def validated(key, value):
    """`value` as option `key`'s type, or ValueError saying what the option accepts.

    A bool is refused (it is an int in Python), as is a float for an integer option
    (TOML `n_predict = 5.5`); an int is fine for a float option (`temperature = 1`).
    """
    kind = OPTION_KEYS[key]
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok or (kind is int and not isinstance(value, int)) or not math.isfinite(value):
        raise ValueError(f"must be {kind.__name__}")
    value = kind(value)
    accepts, description = OPTION_RULES[key]
    if not accepts(value):
        raise ValueError(f"must be {description}")
    return value


def option_overrides(config, source="config.toml"):
    """(options, problems): the session's overrides from `config`, only the keys that are set
    and valid, and one printable line per rejected key.

    source names the file in a problem line.

    State.options holds nothing else. /props reports the server's launch defaults but
    never what a request sent, so the app must remember what it sends; every other
    option is left out of the request and the server's own default applies.
    """
    options = {}
    problems = []
    for key in OPTION_KEYS:
        if config.get(key) is None:
            continue
        try:
            options[key] = validated(key, config[key])
        except ValueError as exc:
            problems.append(f"{source}: {key} {exc}; ignored.")
    return options, problems


def duel_settings(config, source="config.toml"):
    """(settings, problems): the session's /duel settings from `config`; a bad value is reported
    in `problems` (one printable line each) and falls back to its default.

    Keys are the config names without the `duel_` prefix. `auto` has no config key: it
    starts off in every session. A seed of -1 ("random") counts as unset.
    """
    settings = {"auto": False}
    problems = []

    def bad(key, why):
        problems.append(f"{source}: duel_{key} {why}; ignored.")

    axes = config.get("duel_axes", duel.DEFAULT_AXES)
    if isinstance(axes, list) and axes and all(isinstance(a, str) and a.strip() for a in axes):
        settings["axes"] = [a.strip() for a in axes]
    else:
        bad("axes", "must be a non-empty list of non-empty strings")
        settings["axes"] = list(duel.DEFAULT_AXES)

    for side, default in (("a", duel.DEFAULT_STANCE_A), ("b", duel.DEFAULT_STANCE_B)):
        value = config.get(f"duel_stance_{side}", default)
        if isinstance(value, str) and value.strip():
            settings[f"stance_{side}"] = value.strip()
        else:
            bad(f"stance_{side}", "must be a non-empty string")
            settings[f"stance_{side}"] = default

    context = config.get("duel_context", "shared")
    if context in ("shared", "split"):
        settings["context"] = context
    else:
        bad("context", "must be 'shared' or 'split'")
        settings["context"] = "shared"

    for side in duel.SIDES:
        for option in ("temperature", "seed"):
            key = f"{option}_{side}"
            settings[key] = None
            value = config.get(f"duel_{key}")
            if value is None:
                continue
            try:
                value = validated(option, value)
            except ValueError as exc:
                bad(key, str(exc))
                continue
            settings[key] = None if option == "seed" and value == -1 else value
    return settings, problems


def apply_project(config, settings):
    """A new config: `config` with a project's settings laid over it (`config` is not changed)."""
    merged = dict(config)
    merged.update(settings)
    return merged
