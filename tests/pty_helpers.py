"""Run a child process on a pseudo-terminal, for the tests that need a real tty (docs/testing.md).

The child gets the pty slave as stdin/stdout/stderr and as its controlling terminal (otherwise Ctrl-C and
ISIG do not behave as in a real session), a throwaway HOME (so the history file, config and saved
conversations are never the user's), and the repository root as cwd and PYTHONPATH (a script run from
tests/ would otherwise put tests/ at sys.path[0], and `import ui` would fail).

Synchronisation is by what the child writes, never by sleeping: wait_prompts() counts the exact coloured
prompt sequence readline writes, and wait_message() reads a control pipe the child writes to when it
reaches a named hold and then blocks until release(), so a test can look at the tty at a known point.
"""

import fcntl
import os
import re
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What readline writes for ui.PROMPT: the \x01 \x02 markers only tell it the width and are not written.
PROMPT_SEQ = b"\x1b[1;32m>>> \x1b[0m"


class PtyChild:
    def __init__(self, script, home, args=(), cols=100, rows=30, control=False, env=None):
        self.master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        child_env = {
            **os.environ,
            "HOME": str(home),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
            "TERM": "xterm-256color",
            **(env or {}),
        }
        pass_fds, self._messages, self._release_w = (), None, None
        if control:
            msg_r, msg_w = os.pipe()  # child -> test: "I am at hold <name>"
            rel_r, rel_w = os.pipe()  # test -> child: continue
            child_env["CONTROL_W"], child_env["CONTROL_R"] = str(msg_w), str(rel_r)
            pass_fds = (msg_w, rel_r)
            self._messages, self._release_w = msg_r, rel_w
            self._to_close = (msg_w, rel_r)
        self.out = b""
        self._pending_messages = b""
        self.proc = subprocess.Popen(
            [sys.executable, str(script), *map(str, args)],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            pass_fds=pass_fds,
            start_new_session=True,
            preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0),
            env=child_env,
            cwd=ROOT,
        )
        os.close(slave)
        if control:
            for fd in self._to_close:
                os.close(fd)

    # --- reading -----------------------------------------------------------------------------

    def pump(self, timeout=0.05):
        """Read whatever the child has written, waiting up to `timeout` for the first byte."""
        got = False
        end = time.monotonic() + timeout
        while True:
            ready, _, _ = select.select([self.master], [], [], max(0.0, end - time.monotonic()))
            if not ready:
                return got
            try:
                data = os.read(self.master, 65536)
            except OSError:  # the child closed the tty
                return got
            if not data:
                return got
            self.out += data
            got = True
            end = time.monotonic()  # drain what is already there, do not wait for more

    def _fail(self, what):
        raise AssertionError(f"{what}\nlast output: {self.out[-600:]!r}\nchild exit: {self.proc.poll()}")

    def wait_for(self, predicate, what, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump(0.05)
            if predicate(self.out):
                return
        self._fail(f"timed out waiting for {what}")

    def prompts_seen(self):
        return self.out.count(PROMPT_SEQ)

    def wait_prompts(self, n, timeout=10):
        """Wait until the n-th prompt sequence (counted from the start of the run) has been written."""
        self.wait_for(lambda out: out.count(PROMPT_SEQ) >= n, f"prompt #{n}", timeout)

    def got_lines(self):
        return [m.decode("utf-8", "replace") for m in re.findall(rb"GOT (.*?)\r?\n", self.out)]

    def wait_got(self, n, timeout=10):
        """Wait until n GOT lines have been printed; return them as repr strings."""
        self.wait_for(lambda out: len(re.findall(rb"GOT .*?\r?\n", out)) >= n, f"GOT line #{n}", timeout)
        return self.got_lines()

    def after_prompt(self, n):
        """Everything written after the n-th prompt sequence."""
        parts = self.out.split(PROMPT_SEQ)
        return PROMPT_SEQ.join(parts[n:]) if len(parts) > n else b""

    def settle(self, quiet=0.3):
        """Read until the child has been silent for `quiet` seconds (used to assert that nothing more comes)."""
        while self.pump(quiet):
            pass

    # --- control pipe (holds) -----------------------------------------------------------------

    def wait_message(self, name, timeout=10):
        end = time.monotonic() + timeout
        want = name.encode() + b"\n"
        while time.monotonic() < end:
            self.pump(0.02)
            ready, _, _ = select.select([self._messages], [], [], 0.05)
            if ready:
                self._pending_messages += os.read(self._messages, 4096)
            if want in self._pending_messages:
                self._pending_messages = self._pending_messages.split(want, 1)[1]
                # The child wrote its terminal output before it wrote the message, but the pty and the
                # pipe are different channels, so that output may not have been read yet.
                self.settle(0.05)
                return
        self._fail(f"timed out waiting for hold {name!r}")

    def release(self):
        os.write(self._release_w, b"x")

    # --- acting ------------------------------------------------------------------------------

    def send(self, data):
        os.write(self.master, data if isinstance(data, bytes) else data.encode())

    def flags(self):
        lflag = termios.tcgetattr(self.master)[3]
        return {
            "ECHO": bool(lflag & termios.ECHO),
            "ICANON": bool(lflag & termios.ICANON),
            "ISIG": bool(lflag & termios.ISIG),
        }

    def exit_code(self, timeout=10):
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self._fail("the child did not exit")

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for fd in (self.master, self._messages, self._release_w):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
