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


def pytest_addoption(parser):
    parser.addoption(
        "--llama-model",
        default=None,
        help="profile in tests/llama-server.toml to launch for integration tests "
        "(default: $LLAMA_TEST_MODEL, else the first profile)",
    )


@dataclass
class LlamaServer:
    url: str
    port: int
    argv: list
    profile: str
    log_dir: object


@pytest.fixture(scope="session")
def llama_server(request, tmp_path_factory):
    """A llama-server started from tests/llama-server.toml, shared by the whole session.

    Skips when no models directory is configured; every other problem is an error that
    names the cause or points at llama-server.log.
    """
    try:
        setup = cfg.resolve_setup(request.config.getoption("--llama-model"), os.environ)
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
    return LlamaServer(url, port, argv, setup.profile.name, log_dir)


@pytest.fixture
def llama_client(llama_server):
    """A new LlamaClient for the running server, refreshed from /props."""
    client = LlamaClient(llama_server.url)
    client.refresh()
    return client


def pytest_terminal_summary(terminalreporter, config):
    for log_dir in config.stash.get(LOG_DIRS, []):
        terminalreporter.write_line(f"llama-server logs: {log_dir}")
