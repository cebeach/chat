"""Tests for the llama-server integration fixture (tests/conftest.py and its helpers).

Everything but the last test is offline, and none of it touches the real machine setup:
no port 8001 (your own llama-server may be listening there), no real
tests/llama-server.local.toml, no real environment variables. Configs live under
tmp_path, the environment is a plain dict, and "a server" is a stand-in script run on an
ephemeral port. See docs/testing.md.
"""

import json
import socket
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest
import requests

from llama_client import LlamaClient
from tests import llama_server_config as cfg
from tests import llama_server_process as proc_mod
from tests.llama_server_config import LlamaTestConfigError, NotConfigured
from tests.llama_server_process import LlamaStartError, RunningServer

PROFILES = """
[defaults]
args = [
    "--ctx-size 4096",
    "--no-webui",
    "-fa on",
    "--log-prefix",
    "-ngl 99",
]

[models.alpha]
file = "alpha.gguf"
vram_mb = 100
args = [
    "--ctx-size 8192",
    "--temp 0.5",
    "--lora a.bin",
    "--lora b.bin",
]
remove = ["--log-prefix"]

[models.beta]
file = "beta.gguf"
chat_template = "beta.jinja"
startup_timeout = 30
"""


@pytest.fixture
def files(tmp_path):
    """A committed-style profiles file, its template dir and a models dir, all under tmp_path."""
    templates = tmp_path / "chat-templates"
    templates.mkdir()
    (templates / "beta.jinja").write_text("{# beta #}\n{{ messages }}\n")
    models = tmp_path / "models"
    models.mkdir()
    for name in ("alpha.gguf", "beta.gguf"):
        (models / name).write_bytes(b"")
    committed = tmp_path / "llama-server.toml"
    committed.write_text(PROFILES)
    return {
        "committed": committed,
        "templates": templates,
        "models": models,
        "local": tmp_path / "llama-server.local.toml",
    }


def load(files, text=None):
    if text is not None:
        files["committed"].write_text(text)
    return cfg.load_config(files["committed"], files["templates"])


def resolve(files, environ=None, cli_model=None):
    return cfg.resolve_setup(
        cli_model,
        {} if environ is None else environ,
        committed_path=files["committed"],
        local_path=files["local"],
        template_dir=files["templates"],
    )


def make_exe(tmp_path, body, name="fake-server"):
    """A stand-in 'llama-server': a Python script. Returns the argv that runs it."""
    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, str(script)]


def free_url():
    """A URL on an ephemeral port nothing listens on (never 8001)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


class TestBuildArgv:
    def test_merge_order_and_both_dash_flavors(self, files):
        profile = cfg.select_profile(load(files), "alpha")
        argv = cfg.build_argv(profile, files["models"], port=9999)
        # kept defaults first (-fa and -ngl are the single-dash flavor), then the profile's own,
        # in order; --ctx-size 4096 was replaced and --log-prefix removed.
        assert argv[:-6] == [
            "--no-webui",
            "-fa",
            "on",
            "-ngl",
            "99",
            "--ctx-size",
            "8192",
            "--temp",
            "0.5",
            "--lora",
            "a.bin",
            "--lora",
            "b.bin",
        ]

    def test_ends_with_model_host_port(self, files):
        profile = cfg.select_profile(load(files), "alpha")
        argv = cfg.build_argv(profile, files["models"])
        assert argv[-6:] == [
            "-m",
            str(files["models"] / "alpha.gguf"),
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
        ]

    def test_template_flag_comes_before_model(self, files):
        config = load(files)
        with_template = cfg.build_argv(config.profiles["beta"], files["models"], port=1)
        flag = with_template.index("--chat-template-file")
        assert with_template[flag + 1] == str(files["templates"] / "beta.jinja")
        assert flag < with_template.index("-m")
        assert "--chat-template-file" not in cfg.build_argv(config.profiles["alpha"], files["models"], port=1)

    def test_port_none_follows_the_constant_at_call_time(self, files, monkeypatch):
        profile = cfg.select_profile(load(files), "alpha")
        monkeypatch.setattr(cfg, "LLAMA_TEST_PORT", 12345)
        assert cfg.build_argv(profile, files["models"])[-1] == "12345"

    @staticmethod
    def argv_for(files, defaults, args=(), remove=()):
        """argv (without the fixture's -m/--host/--port tail) for a one-profile config."""
        # JSON string syntax is valid TOML, so quotes inside a flag survive.
        text = (
            f"[defaults]\nargs = {json.dumps(list(defaults))}\n"
            f'[models.a]\nfile = "a.gguf"\nargs = {json.dumps(list(args))}\n'
            f"remove = {json.dumps(list(remove))}\n"
        )
        return cfg.build_argv(load(files, text).profiles["a"], files["models"], port=1)[:-6]

    def test_a_profile_replaces_a_default_spelled_the_same(self, files, subtests):
        cases = {
            "long flavor": (["--ctx-size 4096"], ["--ctx-size 8192"], ["--ctx-size", "8192"]),
            "short flavor": (["-c 4096"], ["-c 8192"], ["-c", "8192"]),
            "=value form": (["--ctx-size 4096"], ["--ctx-size=8192"], ["--ctx-size=8192"]),
            "bare flag": (["--kv-unified"], ["--kv-unified"], ["--kv-unified"]),
        }
        for name, (defaults, args, expected) in cases.items():
            with subtests.test(name):
                assert self.argv_for(files, defaults, args) == expected

    def test_aliases_and_negatives_are_not_inferred(self, files, subtests):
        """-c and --ctx-size, --x and --no-x, -no-kvu: all different spellings, all kept."""
        cases = {
            "alias": (["--ctx-size 4096"], ["-c 8192"], ["--ctx-size", "4096", "-c", "8192"]),
            "--no- form": (["--log-prefix"], ["--no-log-prefix"], ["--log-prefix", "--no-log-prefix"]),
            "-no- form": (["--kv-unified"], ["-no-kvu"], ["--kv-unified", "-no-kvu"]),
            "--no-host is not --host": (["--ctx-size 1"], ["--no-host"], ["--ctx-size", "1", "--no-host"]),
        }
        for name, (defaults, args, expected) in cases.items():
            with subtests.test(name):
                assert self.argv_for(files, defaults, args) == expected

    def test_remove_drops_a_default_exactly_as_spelled(self, files):
        argv = self.argv_for(
            files, ["--log-prefix", "--ctx-size 4096"], ["--no-log-prefix"], ["--log-prefix"]
        )
        assert argv == ["--ctx-size", "4096", "--no-log-prefix"]

    def test_repeated_flags_and_quoting(self, files):
        argv = self.argv_for(files, ["--lora d.bin"], ["--lora a.bin", "--lora 'my file.bin'", "--seed -1"])
        assert argv == ["--lora", "a.bin", "--lora", "my file.bin", "--seed", "-1"]  # the profile's replace

    def test_template_flags_in_args_pass_through(self, files):
        text = '[models.a]\nfile = "a"\nchat_template = "beta.jinja"\nargs = ["--jinja", "--chat-template chatml"]\n'
        argv = cfg.build_argv(load(files, text).profiles["a"], files["models"], port=1)
        assert "--jinja" in argv and "chatml" in argv  # not judged: the developer owns the flags

    def test_reserved_flags_are_rejected(self, files, subtests):
        for item in ("-m x.gguf", "--model x.gguf", "--host 0.0.0.0", "--port 9", "--port=9"):
            for text in (
                f'[defaults]\nargs = ["{item}"]\n[models.a]\nfile = "a"\n',
                f'[models.a]\nfile = "a"\nargs = ["{item}"]\n',
            ):
                with subtests.test(item=item, where=text.splitlines()[0]):
                    with pytest.raises(LlamaTestConfigError, match="set by the fixture"):
                        load(files, text)


class TestLoadConfig:
    def test_parses_profiles_in_file_order(self, files):
        config = load(files)
        assert list(config.profiles) == ["alpha", "beta"]
        assert config.profiles["alpha"].vram_mb == 100
        assert config.profiles["beta"].startup_timeout == 30
        assert config.profiles["beta"].chat_template_path == files["templates"] / "beta.jinja"

    @pytest.mark.parametrize(
        "text, match",
        [
            ('bogus = 1\n[models.a]\nfile = "a"\n', "unknown key 'bogus'"),
            ('[defaults]\nargz = []\n[models.a]\nfile = "a"\n', "unknown key 'argz'"),
            ('[models.a]\nfile = "a"\nchat_templat = "x"\n', "unknown key 'chat_templat'"),
            ("", "no profiles defined"),
            ("[models]\n", "no profiles defined"),
            ("[models.a]\nvram_mb = 5\n", "needs a `file`"),
            ('[models.a]\nfile = "a"\nvram_mb = 0\n', "positive integer"),
            ('[models.a]\nfile = "a"\nvram_mb = 1.5\n', "positive integer"),
            ('[models.a]\nfile = "a"\nvram_mb = true\n', "positive integer"),
            ('[models.a]\nfile = "a"\nstartup_timeout = -1\n', "positive number"),
            ('[models.a]\nfile = "a"\nchat_template = "../x.jinja"\n', "plain filename"),
            ('[models.a]\nfile = "a"\nchat_template = "missing.jinja"\n', "does not exist"),
            ("[models.a\n", "not valid TOML"),
            # names that would make a selection ambiguous
            ('[models.all]\nfile = "a"\n', "cannot be used as a profile name"),
            ('[models."a,b"]\nfile = "a"\n', "cannot be used as a profile name"),
            ('[models." a"]\nfile = "a"\n', "cannot be used as a profile name"),
            # the args list
            ('[defaults.args]\nctx-size = 4096\n[models.a]\nfile = "a"\n', "must be a list of strings"),
            ('[models.a]\nfile = "a"\nargs = "--ctx-size 1"\n', "must be a list of strings"),
            ('[models.a]\nfile = "a"\nargs = [5]\n', "must be a string, got 5"),
            (
                '[models.a]\nfile = "a"\nargs = ["ctx-size 4096"]\n',
                "'ctx-size 4096' does not start with a flag",
            ),
            ('[models.a]\nfile = "a"\nargs = ["-1"]\n', "does not start with a flag"),
            ('[models.a]\nfile = "a"\nargs = [""]\n', "does not start with a flag"),
            ('[models.a]\nfile = "a"\nargs = ["--lora \'unclosed"]\n', "cannot split"),
            # remove
            (
                '[defaults]\nargs = ["--ctx-size 1"]\n[models.a]\nfile = "a"\nremove = "--ctx-size"\n',
                "must be a list",
            ),
            (
                '[defaults]\nargs = ["--ctx-size 1"]\n[models.a]\nfile = "a"\nremove = ["-c"]\n',
                "matches no flag in \\[defaults\\]",
            ),
            ('[models.a]\nfile = "a"\nremove = ["--ctx-size"]\n', "matches no flag in \\[defaults\\]"),
            (
                '[defaults]\nargs = ["--ctx-size 1"]\n[models.a]\nfile = "a"\nremove = ["--ctx-size 1"]\n',
                "just the flag",
            ),
        ],
    )
    def test_errors_name_the_problem(self, files, text, match):
        with pytest.raises(LlamaTestConfigError, match=match):
            load(files, text)

    def test_missing_file(self, files):
        with pytest.raises(LlamaTestConfigError, match="not found"):
            cfg.load_config(files["committed"].parent / "nope.toml", files["templates"])

    def test_vram_budget_is_not_compared(self, files):
        files["local"].write_text("[budget]\nvram_mb = 10\n")
        setup = resolve(
            files, {"LLAMA_TEST_MODEL_DIR": str(files["models"]), "PATH": _path_with_llama(files)}
        )
        assert setup.profile.vram_mb == 100  # larger than the budget, and not an error


class TestSelectProfile:
    def test_named_first_unknown_and_empty(self, files):
        config = load(files)
        assert cfg.select_profile(config, "beta").name == "beta"
        assert cfg.select_profile(config).name == "alpha"
        with pytest.raises(
            LlamaTestConfigError, match=r"unknown profile 'nope' \(valid profiles: alpha, beta\)"
        ):
            cfg.select_profile(config, "nope")
        with pytest.raises(LlamaTestConfigError, match="no profiles"):
            cfg.select_profile(cfg.Config(profiles={}))


def _path_with_llama(files):
    """A PATH containing an executable called llama-server (under tmp_path)."""
    bin_dir = files["models"].parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "llama-server"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return str(bin_dir)


class TestResolveSetup:
    def test_no_models_dir_is_the_only_skip(self, files):
        with pytest.raises(NotConfigured, match="docs/testing.md"):
            resolve(files, {})
        with pytest.raises(NotConfigured):  # even when llama-server is on PATH
            resolve(files, {"PATH": _path_with_llama(files)})

    def test_models_dir_from_environ_beats_local_file(self, files, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        (other / "alpha.gguf").write_bytes(b"")
        files["local"].write_text(f'models_dir = "{other}"\n')
        path = _path_with_llama(files)
        assert resolve(files, {"PATH": path}).models_dir == other
        assert (
            resolve(files, {"PATH": path, "LLAMA_TEST_MODEL_DIR": str(files["models"])}).models_dir
            == files["models"]
        )

    def test_returns_everything_the_fixture_needs(self, files):
        environ = {"LLAMA_TEST_MODEL_DIR": str(files["models"]), "PATH": _path_with_llama(files)}
        setup = resolve(files, environ)
        assert setup.profile.name == "alpha"
        assert setup.gguf == files["models"] / "alpha.gguf"
        assert setup.binary.endswith("/bin/llama-server")
        assert setup.startup_timeout == cfg.DEFAULT_STARTUP_TIMEOUT

    def test_model_selection_and_precedence(self, files):
        environ = {
            "LLAMA_TEST_MODEL_DIR": str(files["models"]),
            "PATH": _path_with_llama(files),
            "LLAMA_TEST_MODEL": "beta",
        }
        assert resolve(files, environ).profile.name == "beta"
        assert resolve(files, environ, cli_model="alpha").profile.name == "alpha"  # --llama-model wins
        with pytest.raises(LlamaTestConfigError, match="unknown profile"):
            resolve(files, environ, cli_model="nope")

    def test_startup_timeout_precedence(self, files):
        environ = {"LLAMA_TEST_MODEL_DIR": str(files["models"]), "PATH": _path_with_llama(files)}
        assert resolve(files, environ, cli_model="beta").startup_timeout == 30  # profile
        files["local"].write_text("startup_timeout = 77\n")
        assert resolve(files, environ).startup_timeout == 77  # local file
        assert resolve(files, environ, cli_model="beta").startup_timeout == 30  # profile still wins

    def test_errors_once_a_models_dir_is_configured(self, files, tmp_path):
        path = _path_with_llama(files)
        with pytest.raises(LlamaTestConfigError, match="not a directory"):
            resolve(files, {"LLAMA_TEST_MODEL_DIR": str(tmp_path / "nope"), "PATH": path})
        (files["models"] / "alpha.gguf").unlink()
        with pytest.raises(LlamaTestConfigError, match=r"model file not found: .*alpha\.gguf"):
            resolve(files, {"LLAMA_TEST_MODEL_DIR": str(files["models"]), "PATH": path})
        (files["models"] / "alpha.gguf").write_bytes(b"")
        with pytest.raises(LlamaTestConfigError, match="llama-server not found on PATH"):
            resolve(files, {"LLAMA_TEST_MODEL_DIR": str(files["models"]), "PATH": str(tmp_path / "empty")})

    def test_bad_local_binary_is_reported_even_with_no_models_dir(self, files, tmp_path):
        files["local"].write_text(f'binary = "{tmp_path}/nope/llama-server"\n')
        with pytest.raises(LlamaTestConfigError, match="not an executable file"):
            resolve(files, {})  # it would otherwise be a skip

    def test_local_binary_forms(self, files, tmp_path):
        exe = Path(_path_with_llama(files)) / "llama-server"
        files["local"].write_text(f'binary = "{exe}"\n')
        environ = {"LLAMA_TEST_MODEL_DIR": str(files["models"])}
        assert resolve(files, environ).binary == str(exe)
        files["local"].write_text('binary = "llama-server"\n')  # a bare name is looked up on the dict's PATH
        assert resolve(files, {**environ, "PATH": str(exe.parent)}).binary == str(exe)
        with pytest.raises(LlamaTestConfigError, match="not on PATH"):
            resolve(files, {**environ, "PATH": str(tmp_path / "empty")})

    def test_local_file_keys(self, files):
        for text, match in [
            ("bogus = 1\n", "unknown key 'bogus'"),
            ("[budget]\nvram = 1\n", "unknown key 'vram'"),
            ("startup_timeout = 0\n", "positive number"),
            ("[budget]\nvram_mb = 0\n", "positive integer"),
        ]:
            files["local"].write_text(text)
            with pytest.raises(LlamaTestConfigError, match=match):
                resolve(files, {})
        files["local"].write_text("[budget]\nvram_mb = 24000\n")  # valid, and optional files are fine
        with pytest.raises(NotConfigured):
            resolve(files, {})

    def test_path_rules(self, files, tmp_path):
        home = tmp_path / "home"
        (home / "models").mkdir(parents=True)
        (home / "models" / "alpha.gguf").write_bytes(b"")
        path = _path_with_llama(files)
        setup = resolve(files, {"HOME": str(home), "LLAMA_TEST_MODEL_DIR": "~/models", "PATH": path})
        assert setup.models_dir == home / "models"  # ~ expanded from the dict's HOME
        files["local"].write_text('models_dir = "~/models"\n')
        assert resolve(files, {"HOME": str(home), "PATH": path}).models_dir == home / "models"
        for environ, text in [
            ({"LLAMA_TEST_MODEL_DIR": "relative/models"}, ""),
            ({}, 'models_dir = "relative/models"\n'),
            ({"LLAMA_TEST_MODEL_DIR": str(files["models"])}, 'binary = "bin/llama-server"\n'),
        ]:
            files["local"].write_text(text)
            with pytest.raises(LlamaTestConfigError, match="must be an absolute path"):
                resolve(files, {**environ, "PATH": path})


class TestPorts:
    def test_free_then_busy(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        assert cfg.port_is_free(port)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", port))
            listener.listen()  # bound-but-not-listening would not block the guard's SO_REUSEADDR bind
            assert not cfg.port_is_free(port)

    def test_free_again_right_after_the_server_closed_a_connection(self):
        """The previous run's llama-server closed its connections first, leaving TIME_WAIT."""
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        conn, _ = listener.accept()
        conn.close()  # the server side closes first, so TIME_WAIT is on the server's port
        client.close()
        listener.close()
        with socket.socket() as plain:  # without SO_REUSEADDR this bind fails: the case being guarded
            with pytest.raises(OSError):
                plain.bind(("127.0.0.1", port))
        assert cfg.port_is_free(port)

    def test_check_port(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with pytest.raises(LlamaTestConfigError, match=f"port {port} is already in use"):
                cfg.check_port(port)
        cfg.check_port(port)  # free again: returns quietly, launches nothing


class TestLaunch:
    def test_log_dir_holds_both_streams_and_the_command_line(self, tmp_path):
        argv = make_exe(
            tmp_path,
            """
            import sys
            print("to stdout", flush=True)
            print("to stderr", file=sys.stderr, flush=True)
            """,
        )
        server = proc_mod.launch(argv, tmp_path / "run")
        server.proc.wait(timeout=10)
        proc_mod.stop(server)
        log = (tmp_path / "run" / "llama-server.log").read_text()
        assert "to stdout" in log and "to stderr" in log  # one file, both streams
        assert (
            tmp_path / "run" / "argv.txt"
        ).read_text().splitlines() == argv  # the command line, nothing else
        assert (tmp_path / "run" / "env-removed.txt").exists()
        assert (tmp_path / "run").is_dir()  # not deleted by stop()

    def test_environment_scrub(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "99")
        monkeypatch.setenv("LLAMA_ARG_CHAT_TEMPLATE_FILE", "/x.jinja")
        monkeypatch.setenv("LLAMA_API_KEY", "secret-key")
        monkeypatch.setenv("KEEP_ME", "1")
        argv = make_exe(
            tmp_path,
            """
            import os
            for k in sorted(os.environ):
                if k.startswith(("LLAMA", "KEEP")):
                    print(f"{k}={os.environ[k]}")
            """,
        )
        server = proc_mod.launch(argv, tmp_path / "run")
        server.proc.wait(timeout=10)
        proc_mod.stop(server)
        log = (tmp_path / "run" / "llama-server.log").read_text()
        assert log.strip() == "KEEP_ME=1"  # the others never reached the child
        removed = (tmp_path / "run" / "env-removed.txt").read_text()
        assert removed.splitlines() == ["LLAMA_API_KEY", "LLAMA_ARG_CHAT_TEMPLATE_FILE", "LLAMA_ARG_CTX_SIZE"]
        assert "secret-key" not in removed and "99" not in removed  # names only, never values

    def test_stop_is_idempotent(self, tmp_path):
        argv = make_exe(tmp_path, "import time; time.sleep(60)")
        server = proc_mod.launch(argv, tmp_path / "run")
        proc_mod.stop(server)
        assert server.proc.poll() is not None and server.log_file.closed
        proc_mod.stop(server)  # twice, and on an exited process, without raising


class FakePopen:
    """poll() returns the given values in turn, then repeats the last one."""

    def __init__(self, polls):
        self.polls = list(polls)

    def poll(self):
        return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]


def fake_server(tmp_path, polls, text="loading model...\nfatal: out of memory\n"):
    log = tmp_path / "llama-server.log"
    log.write_text(text)
    return RunningServer(FakePopen(polls), open(log, "ab"), log, ["llama-server", "-m", "x.gguf"])


class TestWaitReady:
    def test_early_exit_is_a_fast_hard_failure_that_starts_with_the_log_path(self, tmp_path):
        server = fake_server(tmp_path, [3])
        with mock.patch.object(LlamaClient, "is_available", side_effect=AssertionError("must not be called")):
            with pytest.raises(LlamaStartError) as exc:
                proc_mod.wait_ready(server, free_url(), timeout=30)
        message = str(exc.value)
        assert message.startswith(str(server.log_path))
        assert "exit code 3" in message
        assert "command: llama-server -m x.gguf" in message
        assert "fatal: out of memory" in message  # the log tail

    def test_timeout_has_the_same_message_shape(self, tmp_path):
        server = fake_server(tmp_path, [None])
        with mock.patch.object(LlamaClient, "is_available", return_value=False):
            with pytest.raises(LlamaStartError) as exc:
                proc_mod.wait_ready(server, free_url(), timeout=0.2, interval=0.05)
        message = str(exc.value)
        assert message.startswith(str(server.log_path))
        assert "did not become ready within 0.2 s" in message
        assert "command: " in message and "fatal: out of memory" in message
        assert "exit code" not in message  # still running, so there is none

    def test_health_ok_but_process_gone_is_a_failure(self, tmp_path):
        """A server that lost a bind race has exited; another server's /health must not count."""
        server = fake_server(tmp_path, [None, 1])  # alive at the first poll, gone after /health answers
        with mock.patch.object(LlamaClient, "is_available", return_value=True):
            with pytest.raises(LlamaStartError, match="exited right after /health answered"):
                proc_mod.wait_ready(server, free_url(), timeout=5)

    def test_connection_errors_are_retried(self, tmp_path):
        server = fake_server(tmp_path, [None])
        answers = [requests.ConnectionError("refused"), requests.Timeout("slow"), True]
        with mock.patch.object(LlamaClient, "is_available", side_effect=answers):
            proc_mod.wait_ready(server, free_url(), timeout=5, interval=0.01)


class TestStart:
    @pytest.fixture
    def launched(self, monkeypatch):
        """Record every server launch() makes, so a test can inspect it after start() raised."""
        seen = []
        real = proc_mod.launch

        def recording(*args, **kwargs):
            seen.append(real(*args, **kwargs))
            return seen[-1]

        monkeypatch.setattr(proc_mod, "launch", recording)
        return seen

    def test_failed_start_has_already_stopped_the_process(self, tmp_path, launched):
        argv = make_exe(tmp_path, "import time; time.sleep(60)")  # never becomes healthy
        with mock.patch.object(LlamaClient, "is_available", return_value=False):
            with pytest.raises(LlamaStartError, match="did not become ready"):
                proc_mod.start(argv, tmp_path / "run", free_url(), timeout=0.3, interval=0.05)
        (server,) = launched
        assert server.proc.poll() is not None  # already gone: no finalizer or teardown has run
        assert server.log_file.closed

    def test_interrupt_stops_the_process_too(self, tmp_path, launched):
        argv = make_exe(tmp_path, "import time; time.sleep(60)")
        with mock.patch.object(LlamaClient, "is_available", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                proc_mod.start(argv, tmp_path / "run", free_url(), timeout=30)
        (server,) = launched
        assert server.proc.poll() is not None and server.log_file.closed

    def test_success_returns_a_running_server(self, tmp_path):
        argv = make_exe(tmp_path, "import time; time.sleep(60)")
        with mock.patch.object(LlamaClient, "is_available", return_value=True):
            server = proc_mod.start(argv, tmp_path / "run", free_url(), timeout=5, interval=0.05)
        try:
            assert server.proc.poll() is None
        finally:
            proc_mod.stop(server)


class TestSelectProfileNames:
    """Which profiles a run exercises: the first by default, opt in to more."""

    @pytest.fixture
    def config(self, files):
        return load(files)

    def test_default_is_the_first_profile_only(self, config, subtests):
        for spec in (None, "", "   "):
            with subtests.test(spec=spec):
                assert cfg.select_profile_names(config, spec) == ["alpha"]

    def test_all_is_every_profile_in_file_order(self, config):
        assert cfg.select_profile_names(config, "all") == ["alpha", "beta"]

    def test_a_list_keeps_the_order_given_and_drops_repeats(self, config, subtests):
        cases = {"beta": ["beta"], "beta,alpha": ["beta", "alpha"], " beta , alpha ": ["beta", "alpha"]}
        cases["alpha,beta,alpha"] = ["alpha", "beta"]
        for spec, expected in cases.items():
            with subtests.test(spec=spec):
                assert cfg.select_profile_names(config, spec) == expected

    def test_errors_name_the_problem(self, config, subtests):
        cases = {
            "nope": r"unknown profile 'nope' \(valid profiles: alpha, beta\)",
            "alpha,nope": "unknown profile 'nope'",
            "alpha,,beta": "empty profile name",
            ",": "empty profile name",
            "all,alpha": "cannot be combined",
        }
        for spec, match in cases.items():
            with subtests.test(spec=spec):
                with pytest.raises(LlamaTestConfigError, match=match):
                    cfg.select_profile_names(config, spec)


class FakeConfig:
    def __init__(self, option=None):
        self.option = option
        self.stash = pytest.Stash()

    def getoption(self, name):
        assert name == "--llama-model"
        return self.option


class FakeMetafunc:
    """Just enough of pytest's Metafunc to see what the hook parametrizes."""

    def __init__(self, fixturenames, option=None):
        self.fixturenames = fixturenames
        self.config = FakeConfig(option)
        self.calls = []

    def parametrize(self, argnames, argvalues, **kwargs):
        self.calls.append((argnames, list(argvalues), kwargs))


class TestGenerateTests:
    """tests/conftest.py's pytest_generate_tests: one llama_server per selected profile."""

    @pytest.fixture(autouse=True)
    def profiles(self, files, monkeypatch):
        config = load(files)
        monkeypatch.setattr(cfg, "load_config", lambda *args, **kwargs: config)
        monkeypatch.delenv("LLAMA_TEST_MODEL", raising=False)

    @staticmethod
    def run(fixturenames, option=None):
        from tests import conftest as conftest_mod

        metafunc = FakeMetafunc(fixturenames, option)
        conftest_mod.pytest_generate_tests(metafunc)
        return metafunc, conftest_mod

    def test_tests_that_do_not_need_the_server_are_left_alone(self):
        metafunc, _ = self.run(["tmp_path", "monkeypatch"])
        assert metafunc.calls == []

    def test_default_parametrizes_the_first_profile_only(self):
        metafunc, _ = self.run(["llama_server"])
        assert metafunc.calls == [("llama_server", ["alpha"], {"indirect": True, "scope": "session"})]

    def test_a_test_using_only_llama_client_is_parametrized_too(self):
        """metafunc.fixturenames is the whole closure, so llama_client pulls llama_server in."""
        metafunc, _ = self.run(["llama_client", "llama_server"])
        assert [call[1] for call in metafunc.calls] == [["alpha"]]

    def test_all_and_lists(self, subtests):
        for option, expected in {"all": ["alpha", "beta"], "beta,alpha": ["beta", "alpha"]}.items():
            with subtests.test(option=option):
                metafunc, _ = self.run(["llama_server"], option)
                assert [call[1] for call in metafunc.calls] == [expected]

    def test_the_environment_variable_is_used_and_the_option_wins(self, monkeypatch):
        monkeypatch.setenv("LLAMA_TEST_MODEL", "beta")
        assert [c[1] for c in self.run(["llama_server"])[0].calls] == [["beta"]]
        assert [c[1] for c in self.run(["llama_server"], "alpha")[0].calls] == [["alpha"]]
        monkeypatch.setenv("LLAMA_TEST_MODEL", "all")
        assert [c[1] for c in self.run(["llama_server"])[0].calls] == [["alpha", "beta"]]

    def test_a_bad_selection_is_raised_when_the_server_is_needed_not_while_collecting(self):
        metafunc, conftest_mod = self.run(["llama_server"], "nope")
        assert [call[1] for call in metafunc.calls] == [[conftest_mod.CONFIG_ERROR_ID]]
        error = metafunc.config.stash[conftest_mod.CONFIG_ERROR]
        assert isinstance(error, LlamaTestConfigError) and "unknown profile 'nope'" in str(error)

    def test_an_unreadable_profiles_file_is_handled_the_same_way(self, monkeypatch):
        def broken(*args, **kwargs):
            raise LlamaTestConfigError("profiles file is not valid TOML")

        monkeypatch.setattr(cfg, "load_config", broken)
        metafunc, conftest_mod = self.run(["llama_server"])
        assert [call[1] for call in metafunc.calls] == [[conftest_mod.CONFIG_ERROR_ID]]
        assert "not valid TOML" in str(metafunc.config.stash[conftest_mod.CONFIG_ERROR])


class TestCommittedConfig:
    """tests/llama-server.toml and tests/chat-templates/ as committed in the repository."""

    def test_every_profile_builds_an_argv(self, tmp_path):
        config = cfg.load_config()
        for name, profile in config.profiles.items():
            argv = cfg.build_argv(profile, tmp_path)  # a dummy models dir: no real models needed
            assert argv[-4:-1] == ["--host", "127.0.0.1", "--port"], name
            assert argv[-1] == "8001", name

    def test_templates_and_profiles_match(self):
        config = cfg.load_config()
        used = {p.chat_template for p in config.profiles.values() if p.chat_template}
        on_disk = {p.name for p in cfg.TEMPLATE_DIR.glob("*.jinja")} if cfg.TEMPLATE_DIR.is_dir() else set()
        assert used <= on_disk  # load_config already checks each one exists
        assert on_disk <= used, f"templates no profile uses: {sorted(on_disk - used)}"


@pytest.mark.integration
def test_server_smoke(llama_server, llama_client):
    """Needs a real llama-server: see docs/testing.md. Skipped when no models dir is configured."""
    assert llama_client.is_available()
    assert llama_client.server_model
    assert llama_client.server_n_ctx
    profile = cfg.select_profile(cfg.load_config(), llama_server.profile)
    if profile.chat_template_path:
        served = requests.get(f"{llama_server.url}/props", timeout=10).json()["chat_template"]
        assert served.strip() == profile.chat_template_path.read_text().strip()
