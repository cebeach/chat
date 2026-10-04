"""Config, argv and port helpers for the real llama-server integration fixture.

Everything here is pure (no pytest import, no process launched) so it can be unit
tested offline. The fixture in conftest.py is only wiring around these functions. See
docs/testing.md.
"""

import os
import shutil
import socket
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
COMMITTED_CONFIG = TESTS_DIR / "llama-server.toml"
LOCAL_CONFIG = TESTS_DIR / "llama-server.local.toml"
TEMPLATE_DIR = TESTS_DIR / "chat-templates"

HOST = "127.0.0.1"
LLAMA_TEST_PORT = 8001
DEFAULT_STARTUP_TIMEOUT = 120

# Flags the fixture sets itself; a profile may not set them (short forms included).
RESERVED_ARGS = frozenset({"model", "m", "host", "port"})


class NotConfigured(Exception):
    """This machine is not set up for the integration tests; the fixture skips."""


class LlamaTestConfigError(Exception):
    """A problem found before launching llama-server; a hard error, never a skip."""


@dataclass
class Profile:
    name: str
    file: str
    args: dict = field(default_factory=dict)  # [defaults.args] already merged under the profile's
    chat_template: str | None = None
    chat_template_path: Path | None = None
    vram_mb: int | None = None
    startup_timeout: float | None = None


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


def _check_args(args, where):
    if not isinstance(args, dict):
        raise LlamaTestConfigError(f"{where} must be a table of flag = value")
    for key, value in args.items():
        flag = key.lstrip("-")
        if not flag:
            raise LlamaTestConfigError(f"empty flag name in {where}")
        if flag in RESERVED_ARGS:
            raise LlamaTestConfigError(
                f"{key!r} in {where} is set by the fixture (model, host and port are reserved)"
            )
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, bool) and isinstance(value, list):
                raise LlamaTestConfigError(f"{key!r} in {where}: booleans are not allowed in a list")
            if not isinstance(item, bool | int | float | str):
                raise LlamaTestConfigError(f"{key!r} in {where} has an unsupported value: {value!r}")


def load_config(path=COMMITTED_CONFIG, template_dir=TEMPLATE_DIR):
    """Parse and validate the committed profiles file.

    Unknown keys, a bad type, a reserved flag, a bad chat_template or an empty `models`
    table are errors naming the problem. Whether the flags in `args` make sense for
    llama-server is not checked: that is the developer's responsibility.
    """
    raw = _read_toml(path, "profiles file")
    where_file = Path(path).name
    _check_keys(raw, {"defaults", "models"}, where_file)

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise LlamaTestConfigError(f"[defaults] in {where_file} must be a table")
    _check_keys(defaults, {"args"}, f"[defaults] in {where_file}")
    default_args = defaults.get("args", {})
    _check_args(default_args, f"[defaults.args] in {where_file}")

    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise LlamaTestConfigError(f"no profiles defined in {where_file}")

    profiles = {}
    for name, table in models.items():
        where = f"profile {name!r}"
        if not isinstance(table, dict):
            raise LlamaTestConfigError(f"{where} must be a table")
        _check_keys(table, {"file", "args", "chat_template", "vram_mb", "startup_timeout"}, where)
        file = table.get("file")
        if not isinstance(file, str) or not file:
            raise LlamaTestConfigError(f"{where} needs a `file` (the GGUF filename)")
        args = table.get("args", {})
        _check_args(args, f"{where} args")

        profile = Profile(name=name, file=file, args={**default_args, **args})
        if "vram_mb" in table:
            profile.vram_mb = _positive_int(table["vram_mb"], f"vram_mb in {where}")
        if "startup_timeout" in table:
            profile.startup_timeout = _positive_number(
                table["startup_timeout"], f"startup_timeout in {where}"
            )
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
):
    """Everything the fixture needs, or NotConfigured (skip) / LlamaTestConfigError (error).

    Takes `environ` and the two file paths as arguments so tests can pass a dict and
    files under tmp_path instead of the real machine setup.
    """
    config = load_config(committed_path, template_dir)
    local = load_local(local_path)
    profile = select_profile(config, cli_model or environ.get("LLAMA_TEST_MODEL"))

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


def _flags(args):
    out = []
    for key, value in args.items():
        flag = key if key.startswith("-") else ("-" + key if len(key) == 1 else "--" + key)
        if value is True:
            out.append(flag)
        elif value is False:
            continue
        elif isinstance(value, list):
            for item in value:
                out += [flag, str(item)]
        else:
            out += [flag, str(value)]
    return out


def build_argv(profile, models_dir, port=None):
    """The llama-server arguments (without the binary) for a profile.

    Order: the merged args, the chat template flag if the profile has one, then -m, --host
    and --port, so the result always ends with the host and port. `port=None` reads
    LLAMA_TEST_PORT at call time (a default argument would be fixed at definition).
    """
    if port is None:
        port = LLAMA_TEST_PORT
    argv = _flags(profile.args)
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
