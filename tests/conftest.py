"""Fixtures for the integration tests that need a real llama-server. See docs/testing.md.

All rules live in tests/llama_server_config.py and tests/llama_server_process.py, where
they are unit tested; the fixture below only wires them together.
"""

import os
from dataclasses import dataclass

import pytest

from llama_client import LlamaClient
from tests import llama_server_config as cfg
from tests.llama_server_process import start, stop

LOG_DIRS = pytest.StashKey[list]()
CONFIG_ERROR = pytest.StashKey[Exception]()

# Stands in for a profile name when the profiles file cannot be read at collection time.
# The error is raised when an integration test needs the server, not while collecting, so
# runs that deselect the integration tests (-m "not integration") do not depend on the file.
CONFIG_ERROR_ID = "invalid-llama-server-config"


def pytest_addoption(parser):
    parser.addoption(
        "--llama-model",
        default=None,
        help="which profiles in tests/llama-server.toml the integration tests run against: "
        "a name, a comma-separated list, or 'all' (default: $LLAMA_TEST_MODEL, else the first "
        "profile only). Each profile is one server start, one after another",
    )


def pytest_generate_tests(metafunc):
    """Run every test that needs llama_server once per selected profile.

    The fixture is session-scoped and parametrized, so pytest groups the tests by profile:
    it starts one server, runs everything that needs it, stops it, then starts the next.
    Servers never overlap, so only one model is in memory at a time and port 8001 is reused.
    """
    if "llama_server" not in metafunc.fixturenames:
        return
    spec = metafunc.config.getoption("--llama-model") or os.environ.get("LLAMA_TEST_MODEL")
    try:
        names = cfg.select_profile_names(cfg.load_config(), spec)
    except cfg.LlamaTestConfigError as exc:
        metafunc.config.stash[CONFIG_ERROR] = exc
        names = [CONFIG_ERROR_ID]
    extra = tuple(arg for mark in metafunc.definition.iter_markers("llama_args") for arg in mark.args)
    if extra and names != [CONFIG_ERROR_ID]:
        # A tuple, so equal extras compare equal and pytest groups those tests on one server.
        suffix = cfg.extras_id(extra)
        names = [pytest.param((name, extra), id=f"{name}+{suffix}") for name in names]
    metafunc.parametrize("llama_server", names, indirect=True, scope="session")


@dataclass
class LlamaServer:
    url: str
    port: int
    argv: list
    profile: str
    log_dir: object
    extra_args: tuple = ()


@pytest.fixture(scope="session")
def llama_server(request, tmp_path_factory):
    """A llama-server started from tests/llama-server.toml for one profile.

    Parametrized by the profile (see pytest_generate_tests): by default only the first
    profile, or the ones chosen with --llama-model. A test marked
    `@pytest.mark.llama_args("--ctx-size 2048")` gets the profile's server with those flags
    merged over its args. Skips when no models directory is configured; every other problem is an error that names the cause or points at
    llama-server.log.
    """
    param = getattr(request, "param", None)
    name, extra = param if isinstance(param, tuple) else (param, ())
    if name == CONFIG_ERROR_ID:
        raise request.config.stash[CONFIG_ERROR]
    try:
        setup = cfg.resolve_setup(name, os.environ, extra_args=extra)
    except cfg.NotConfigured as exc:
        pytest.skip(str(exc))

    port = cfg.LLAMA_TEST_PORT
    cfg.check_port(port)
    argv = [setup.binary, *cfg.build_argv(setup.profile, setup.models_dir, port)]
    log_dir = tmp_path_factory.mktemp(f"llama-server-{setup.profile.name}")
    request.config.stash.setdefault(LOG_DIRS, []).append(log_dir)

    url = f"http://{cfg.HOST}:{port}"
    server = start(argv, log_dir, url, setup.startup_timeout)
    request.addfinalizer(lambda: stop(server))
    return LlamaServer(url, port, argv, setup.profile.name, log_dir, extra)


@pytest.fixture
def llama_client(llama_server):
    """A new LlamaClient for the running server, refreshed from /props."""
    client = LlamaClient(llama_server.url)
    client.refresh()
    return client


def pytest_terminal_summary(terminalreporter, config):
    for log_dir in config.stash.get(LOG_DIRS, []):
        terminalreporter.write_line(f"llama-server logs: {log_dir}")
