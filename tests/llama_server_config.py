"""Config, argv and port helpers for the real llama-server integration fixture.

Everything here is pure (no pytest import, no process launched) so it can be unit
tested offline. The fixture in conftest.py is only wiring around these functions. See
docs/testing.md.
"""

import os
import re
import shlex
import shutil
import socket
import tomllib
from urllib.parse import urlparse
from dataclasses import dataclass, field, replace
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
COMMITTED_CONFIG = TESTS_DIR / "llama-server.toml"
LOCAL_CONFIG = TESTS_DIR / "llama-server.local.toml"
TEMPLATE_DIR = TESTS_DIR / "chat-templates"

HOST = "127.0.0.1"
LLAMA_TEST_PORT = 8001
DEFAULT_STARTUP_TIMEOUT = 120

# --llama-model all: every profile, one after another (not a valid profile name)
ALL_PROFILES = "all"

# Flags the fixture sets itself (-m, --host, --port); [defaults] and a profile may not.
RESERVED_FLAGS = frozenset({"-m", "--model", "--host", "--port"})

# A flag as written to llama-server: one or two dashes, then a letter. This accepts both
# flavors (-ngl, --n-gpu-layers) and rejects a bare name (ctx-size) and a number (-1).
FLAG_RE = re.compile(r"--?[A-Za-z][^\s]*")


class NotConfigured(Exception):
    """This machine is not set up for the integration tests; the fixture skips."""


class LlamaTestConfigError(Exception):
    """A problem found before launching llama-server; a hard error, never a skip."""


@dataclass
class Profile:
    name: str
    file: str
    args: list = field(default_factory=list)  # defaults merged under the profile's; one token tuple per flag
    chat_template: str | None = None
    chat_template_path: Path | None = None
    vram_mb: int | None = None
    startup_timeout: float | None = None
    # The thinking tags this model is expected to use, as observed on the real model:
    # None = not recorded, () = the model does not think, (start, end) = a thinking model.
    think_tags: tuple | None = None
    # Where the GGUF came from, as an http(s) link to the file. A reference only: nothing
    # downloads it, and the tests use the file already in the models directory.
    source_url: str | None = None


@dataclass
class Config:
    profiles: dict  # name -> Profile, in file order


@dataclass
class LocalConfig:
    models_dir: str | None = None
    binary: str | None = None
    startup_timeout: float | None = None
    budget_vram_mb: int | None = None


@dataclass
class Setup:
    binary: str
    models_dir: Path
    gguf: Path
    profile: Profile
    startup_timeout: float


def _read_toml(path, what):
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise LlamaTestConfigError(f"{what} not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise LlamaTestConfigError(f"{what} is not valid TOML ({path}): {exc}") from None


def _check_keys(table, allowed, where):
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise LlamaTestConfigError(
            f"unknown key {unknown[0]!r} in {where} (allowed: {', '.join(sorted(allowed))})"
        )


def _positive_int(value, where):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LlamaTestConfigError(f"{where} must be a positive integer, got {value!r}")
    return value


def _positive_number(value, where):
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise LlamaTestConfigError(f"{where} must be a positive number, got {value!r}")
    return value


def _flag_name(tokens):
    """The flag exactly as spelled, without a trailing =value (--ctx-size=4096 -> --ctx-size)."""
    first = tokens[0]
    return first.split("=", 1)[0] if first.startswith("--") else first


def _split_flag(item, where):
    """One string written as you would type it to llama-server -> its tokens.

    The first token is the flag and must start with a dash (either flavor); the rest are
    its value(s). Whether llama-server knows the flag is not checked.
    """
    if not isinstance(item, str):
        raise LlamaTestConfigError(f"{where}: every element must be a string, got {item!r}")
    try:
        tokens = shlex.split(item)
    except ValueError as exc:
        raise LlamaTestConfigError(f"{where}: cannot split {item!r}: {exc}") from None
    if not tokens or not FLAG_RE.fullmatch(tokens[0]):
        raise LlamaTestConfigError(
            f"{where}: {item!r} does not start with a flag; write it as you would type it to "
            "llama-server, for example '--ctx-size 4096' or '-c 4096'"
        )
    return tuple(tokens)


def _parse_flags(value, where):
    """A list of strings, one llama-server flag (and its value) each -> a list of token tuples."""
    if not isinstance(value, list):
        raise LlamaTestConfigError(f"{where} must be a list of strings, one flag per element")
    parsed = []
    for item in value:
        tokens = _split_flag(item, where)
        if _flag_name(tokens) in RESERVED_FLAGS:
            raise LlamaTestConfigError(
                f"{item!r} in {where} is set by the fixture (-m/--model, --host and --port are reserved)"
            )
        parsed.append(tokens)
    return parsed


def _parse_remove(value, where, default_names):
    """Flags to drop from [defaults] for one profile, spelled exactly as the default spells them."""
    if not isinstance(value, list):
        raise LlamaTestConfigError(f"{where} must be a list of flags")
    names = set()
    for item in value:
        tokens = _split_flag(item, where)
        if len(tokens) != 1:
            raise LlamaTestConfigError(f"{where}: {item!r} must be just the flag, with no value")
        if tokens[0] not in default_names:
            raise LlamaTestConfigError(
                f"{where}: {item!r} matches no flag in [defaults] args. Spell it exactly as the "
                "default does: aliases such as -c and --ctx-size are not matched."
            )
        names.add(tokens[0])
    return names


def _merge(default_args, own_args, removed):
    """Defaults minus removed ones and ones the profile spells again, then the profile's own."""
    own_names = {_flag_name(tokens) for tokens in own_args}
    kept = [t for t in default_args if _flag_name(t) not in own_names and _flag_name(t) not in removed]
    return kept + own_args


def with_extra_args(profile, extra):
    """The profile with a test's `llama_args` marker flags merged over its args.

    Same rule as a profile over [defaults]: a flag spelled the same way replaces the one
    already there (--ctx-size 2048 replaces --ctx-size 16384; -c 2048 would not), and the
    reserved flags are refused. The profile itself is not changed.
    """
    if not extra:
        return profile
    return replace(profile, args=_merge(profile.args, _parse_flags(list(extra), "llama_args"), set()))


def extras_id(extra):
    """A short test-id suffix for a test's extra flags: ('--ctx-size 2048',) -> 'ctx-size-2048'."""
    return "+".join("-".join(item.split()).lstrip("-") for item in extra)


def _parse_think_tags(value, where):
    """[] (the model does not think) or [start, end]; the tags themselves are not judged."""
    ok = isinstance(value, list) and len(value) in (0, 2)
    ok = ok and all(isinstance(tag, str) and tag.strip() for tag in value)
    if not ok:
        raise LlamaTestConfigError(
            f"think_tags in {where} must be [] (the model does not think) or [start, end], "
            f"two non-empty strings; got {value!r}"
        )
    return tuple(value)


def _parse_source_url(value, where):
    """An http(s) link to the GGUF file. Nothing is fetched; the value is only checked.

    Credentials in the URL are refused (the file is committed) and never echoed back.
    """
    problem = f"source_url in {where} must be an http(s) URL of the GGUF file, got {value!r}"
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise LlamaTestConfigError(problem)
    parts = urlparse(value)
    if parts.scheme not in ("http", "https") or not parts.netloc or not parts.hostname:
        raise LlamaTestConfigError(problem)
    if parts.username is not None or parts.password is not None:
        raise LlamaTestConfigError(
            f"source_url in {where} must not contain credentials: the profiles file is committed"
        )
    return value


def load_config(path=COMMITTED_CONFIG, template_dir=TEMPLATE_DIR):
    """Parse and validate the committed profiles file.

    Unknown keys, a bad type, a reserved flag, a bad chat_template or an empty `models`
    table are errors naming the problem. Whether the flags in `args` make sense for
    llama-server is not checked: that is the developer's responsibility.

    `args` is a list of strings, one flag per element, written as you would type it to
    llama-server. A profile's flags replace a default spelled with the same flag; aliases
    (-c and --ctx-size) are not matched, so `remove` names defaults to drop.
    """
    raw = _read_toml(path, "profiles file")
    where_file = Path(path).name
    _check_keys(raw, {"defaults", "models"}, where_file)

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise LlamaTestConfigError(f"[defaults] in {where_file} must be a table")
    _check_keys(defaults, {"args"}, f"[defaults] in {where_file}")
    default_args = _parse_flags(defaults.get("args", []), f"[defaults] args in {where_file}")
    default_names = {_flag_name(tokens) for tokens in default_args}

    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise LlamaTestConfigError(f"no profiles defined in {where_file}")

    profiles = {}
    for name, table in models.items():
        where = f"profile {name!r}"
        if name == ALL_PROFILES or "," in name or name != name.strip():
            raise LlamaTestConfigError(
                f"{where} cannot be used as a profile name: {ALL_PROFILES!r} selects every profile, "
                "a comma separates names in a selection, and names have no surrounding spaces"
            )
        if not isinstance(table, dict):
            raise LlamaTestConfigError(f"{where} must be a table")
        _check_keys(
            table,
            {
                "file",
                "args",
                "remove",
                "chat_template",
                "vram_mb",
                "startup_timeout",
                "think_tags",
                "source_url",
            },
            where,
        )
        file = table.get("file")
        if not isinstance(file, str) or not file:
            raise LlamaTestConfigError(f"{where} needs a `file` (the GGUF filename)")
        own_args = _parse_flags(table.get("args", []), f"args in {where}")
        removed = _parse_remove(table.get("remove", []), f"remove in {where}", default_names)

        profile = Profile(name=name, file=file, args=_merge(default_args, own_args, removed))
        if "vram_mb" in table:
            profile.vram_mb = _positive_int(table["vram_mb"], f"vram_mb in {where}")
        if "startup_timeout" in table:
            profile.startup_timeout = _positive_number(
                table["startup_timeout"], f"startup_timeout in {where}"
            )
        if "think_tags" in table:
            profile.think_tags = _parse_think_tags(table["think_tags"], where)
        if "source_url" in table:
            profile.source_url = _parse_source_url(table["source_url"], where)
        if "chat_template" in table:
            template = table["chat_template"]
            if not isinstance(template, str) or not template or template != Path(template).name:
                raise LlamaTestConfigError(
                    f"chat_template in {where} must be a plain filename in {template_dir}, got {template!r}"
                )
            template_path = Path(template_dir) / template
            if not template_path.is_file():
                raise LlamaTestConfigError(f"chat_template in {where}: {template_path} does not exist")
            profile.chat_template = template
            profile.chat_template_path = template_path
        profiles[name] = profile
    return Config(profiles=profiles)


def known_think_tags(config):
    """Every thinking tag any profile records: what a non-thinking reply must not contain."""
    return frozenset(tag for profile in config.profiles.values() for tag in profile.think_tags or ())


def select_profile(config, name=None):
    """The named profile, or the first one in the file when no name is given."""
    if not config.profiles:
        raise LlamaTestConfigError("no profiles defined")
    if not name:
        return next(iter(config.profiles.values()))
    try:
        return config.profiles[name]
    except KeyError:
        valid = ", ".join(config.profiles)
        raise LlamaTestConfigError(f"unknown profile {name!r} (valid profiles: {valid})") from None


def select_profile_names(config, spec=None):
    """The profile names a run exercises, in file order or in the order given.

    No spec (the default) is the first profile only, so a plain run loads one model.
    "all" is every profile; "a,b" is those two. Each name becomes one server start, one
    after another, so the cost is one model load per name.
    """
    names = list(config.profiles)
    if not names:
        raise LlamaTestConfigError("no profiles defined")
    if spec is None or not spec.strip():
        return names[:1]
    items = [item.strip() for item in spec.split(",")]
    if "" in items:
        raise LlamaTestConfigError(f"empty profile name in {spec!r}")
    if ALL_PROFILES in items:
        if len(items) > 1:
            raise LlamaTestConfigError(f"{ALL_PROFILES!r} cannot be combined with other names: {spec!r}")
        return names
    for item in items:
        if item not in config.profiles:
            raise LlamaTestConfigError(f"unknown profile {item!r} (valid profiles: {', '.join(names)})")
    return list(dict.fromkeys(items))


def load_local(path=LOCAL_CONFIG):
    """Parse the optional, gitignored machine-specific file. Missing is fine."""
    if not Path(path).exists():
        return LocalConfig()
    raw = _read_toml(path, "local config")
    where_file = Path(path).name
    _check_keys(raw, {"models_dir", "binary", "startup_timeout", "budget"}, where_file)
    local = LocalConfig()
    for key in ("models_dir", "binary"):
        if key in raw:
            if not isinstance(raw[key], str) or not raw[key]:
                raise LlamaTestConfigError(f"{key} in {where_file} must be a non-empty string")
            setattr(local, key, raw[key])
    if "startup_timeout" in raw:
        local.startup_timeout = _positive_number(raw["startup_timeout"], f"startup_timeout in {where_file}")
    if "budget" in raw:
        budget = raw["budget"]
        if not isinstance(budget, dict):
            raise LlamaTestConfigError(f"[budget] in {where_file} must be a table")
        _check_keys(budget, {"vram_mb"}, f"[budget] in {where_file}")
        if "vram_mb" in budget:
            local.budget_vram_mb = _positive_int(budget["vram_mb"], f"vram_mb in [budget] in {where_file}")
    return local


def _absolute(value, what, environ):
    """Expand a leading ~ (using environ's HOME) and require an absolute path."""
    if value == "~" or value.startswith("~/"):
        home = environ.get("HOME") or str(Path.home())
        path = Path(home + value[1:])
    else:
        path = Path(value).expanduser()
    if not path.is_absolute():
        raise LlamaTestConfigError(f"{what} must be an absolute path (after ~ expansion), got {value!r}")
    return path


def _find_binary(configured, environ):
    """The explicit local-file `binary` (checked even when nothing else is configured)."""
    if "/" not in configured and not configured.startswith("~"):
        found = shutil.which(configured, path=environ.get("PATH"))
        if not found:
            raise LlamaTestConfigError(f"binary {configured!r} in the local config is not on PATH")
        return found
    path = _absolute(configured, "binary in the local config", environ)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise LlamaTestConfigError(f"binary in the local config is not an executable file: {path}")
    return str(path)


def resolve_setup(
    cli_model,
    environ,
    committed_path=COMMITTED_CONFIG,
    local_path=LOCAL_CONFIG,
    template_dir=TEMPLATE_DIR,
    extra_args=(),
):
    """Everything the fixture needs, or NotConfigured (skip) / LlamaTestConfigError (error).

    `extra_args` are a test's `llama_args` flags, merged over the profile's own.

    Takes `environ` and the two file paths as arguments so tests can pass a dict and
    files under tmp_path instead of the real machine setup.
    """
    config = load_config(committed_path, template_dir)
    local = load_local(local_path)
    profile = with_extra_args(
        select_profile(config, cli_model or environ.get("LLAMA_TEST_MODEL")), extra_args
    )

    binary = _find_binary(local.binary, environ) if local.binary else None

    from_env = environ.get("LLAMA_TEST_MODEL_DIR")
    if from_env:
        models_dir_value, source = from_env, "$LLAMA_TEST_MODEL_DIR"
    elif local.models_dir:
        models_dir_value, source = local.models_dir, "models_dir in the local config"
    else:
        raise NotConfigured(
            "no models directory configured: set LLAMA_TEST_MODEL_DIR or models_dir in "
            f"{Path(local_path).name} (see docs/testing.md)"
        )

    models_dir = _absolute(models_dir_value, source, environ)
    if not models_dir.is_dir():
        raise LlamaTestConfigError(f"{source} is not a directory: {models_dir}")
    gguf = models_dir / profile.file
    if not gguf.is_file():
        raise LlamaTestConfigError(f"profile {profile.name!r}: model file not found: {gguf}")
    if binary is None:
        binary = shutil.which("llama-server", path=environ.get("PATH"))
        if not binary:
            raise LlamaTestConfigError(
                "llama-server not found on PATH; install it or set binary in the local config"
            )

    startup_timeout = profile.startup_timeout or local.startup_timeout or DEFAULT_STARTUP_TIMEOUT
    return Setup(binary, models_dir, gguf, profile, startup_timeout)


def build_argv(profile, models_dir, port=None):
    """The llama-server arguments (without the binary) for a profile.

    Order: the merged args, the chat template flag if the profile has one, then -m, --host
    and --port, so the result always ends with the host and port. `port=None` reads
    LLAMA_TEST_PORT at call time (a default argument would be fixed at definition).
    """
    if port is None:
        port = LLAMA_TEST_PORT
    argv = [token for tokens in profile.args for token in tokens]
    if profile.chat_template_path:
        argv += ["--chat-template-file", str(profile.chat_template_path)]
    argv += ["-m", str(Path(models_dir) / profile.file), "--host", HOST, "--port", str(port)]
    return argv


def port_is_free(port):
    """True if nothing is listening on HOST:port.

    SO_REUSEADDR is set, as llama-server's own HTTP library does: without it the
    TIME_WAIT connections left by a previous server on this port make the bind fail for a
    while, and a back-to-back run would report a busy port with nothing on it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
        except OSError:
            return False
    return True


def check_port(port=None):
    if port is None:
        port = LLAMA_TEST_PORT
    if not port_is_free(port):
        raise LlamaTestConfigError(
            f"port {port} is already in use; stop whatever is listening there (your own llama-server?) "
            "and run again. The fixture never uses or stops a server it did not start."
        )
