"""Launching, waiting for and stopping a real llama-server for the integration tests.

Kept out of conftest.py so it can be unit tested with a stand-in executable. See
docs/testing.md.
"""

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from llama_client import LlamaClient

LOG_TAIL_LINES = 40
STOP_TIMEOUT = 10
POLL_INTERVAL = 0.5

# llama-server reads its options from these, which would silently override the
# committed profile (and an API key would make /props refuse LlamaClient).
SCRUBBED_PREFIX = "LLAMA_ARG_"
SCRUBBED_NAMES = frozenset({"LLAMA_API_KEY"})


class LlamaStartError(Exception):
    """llama-server was launched but did not become ready. The message starts with the log path."""


@dataclass
class RunningServer:
    proc: subprocess.Popen
    log_file: object
    log_path: Path
    argv: list


def scrub_env(environ):
    """(environment for the child, sorted names removed)."""
    env, removed = {}, []
    for name, value in environ.items():
        if name.startswith(SCRUBBED_PREFIX) or name in SCRUBBED_NAMES:
            removed.append(name)
        else:
            env[name] = value
    return env, sorted(removed)


def launch(argv, log_dir, environ=None):
    """Start argv with stdout and stderr in one file in log_dir. Returns a RunningServer."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env, removed = scrub_env(os.environ if environ is None else environ)
    (log_dir / "argv.txt").write_text("".join(f"{a}\n" for a in argv))
    (log_dir / "env-removed.txt").write_text("".join(f"{n}\n" for n in removed))
    log_path = log_dir / "llama-server.log"
    log_file = open(log_path, "wb")
    try:
        proc = subprocess.Popen(
            argv, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env
        )
    except BaseException:
        log_file.close()
        raise
    return RunningServer(proc, log_file, log_path, list(argv))


def _log_tail(log_path, lines=LOG_TAIL_LINES):
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError as exc:
        return f"(could not read the log: {exc})"
    tail = text.splitlines()[-lines:]
    return "\n".join(tail) if tail else "(the log is empty)"


def _failure(server, reason, exit_code=None):
    code = f" (exit code {exit_code})" if exit_code is not None else ""
    return LlamaStartError(
        f"{server.log_path}\n"
        f"llama-server {reason}{code}. The file named on the line above is its full output.\n"
        f"command: {shlex.join(server.argv)}\n"
        f"--- last {LOG_TAIL_LINES} lines of the log ---\n"
        f"{_log_tail(server.log_path)}"
    )


def wait_ready(server, url, timeout, interval=POLL_INTERVAL):
    """Poll /health until it returns 200, raising LlamaStartError on exit or timeout.

    proc.poll() is checked before every /health call and once more after the first
    success: a server that lost a bind race has exited by then, so another server's
    /health is not accepted as ours.
    """
    deadline = time.monotonic() + timeout
    while True:
        code = server.proc.poll()
        if code is not None:
            raise _failure(server, "exited before it became ready", code)
        try:
            ready = LlamaClient(url).is_available()
        except requests.RequestException:
            ready = False
        if ready:
            code = server.proc.poll()
            if code is not None:
                raise _failure(server, "exited right after /health answered", code)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _failure(server, f"did not become ready within {timeout:g} s")
        time.sleep(min(interval, remaining))


def stop(server, timeout=STOP_TIMEOUT):
    """Terminate, then kill, then close the log. Safe on an exited process and safe to repeat."""
    proc = server.proc
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if not server.log_file.closed:
        server.log_file.close()


def start(argv, log_dir, url, timeout, environ=None, interval=POLL_INTERVAL):
    """launch + wait_ready. On any failure (including Ctrl-C) the process is stopped first.

    A finalizer on a session-scoped fixture only runs when the whole session ends, and
    code after a fixture's yield is never reached when setup raises, so cleanup of a
    failed start has to happen here.
    """
    server = launch(argv, log_dir, environ)
    try:
        wait_ready(server, url, timeout, interval)
    except BaseException:
        stop(server)
        raise
    return server
