"""get_user_input() and echo_suppressed() on a real pseudo-terminal (no model, no network).

The old prompt read the first key itself through sys.stdin and re-injected it into readline; a
fast burst (a paste, an arrow key, type-ahead) lost everything after its first byte and showed the
rest again, one character per later prompt. These tests drive a child process on a pty, so the real
readline and the real tty are involved. See docs/testing.md and tests/pty_helpers.py.
"""

import subprocess
import sys

import pytest

from tests.pty_helpers import PROMPT_SEQ, ROOT, PtyChild

PROMPT_CHILD = ROOT / "tests" / "prompt_child.py"
STREAM_CHILD = ROOT / "tests" / "stream_child.py"


@pytest.fixture
def start(tmp_path):
    children = []

    def factory(script=PROMPT_CHILD, **kwargs):
        child = PtyChild(script, tmp_path, **kwargs)
        children.append(child)
        return child

    yield factory
    for child in children:
        child.close()


class TestThePrompt:
    def test_the_child_starts_and_readline_has_prepared_the_tty_when_the_prompt_appears(self, start):
        # A path or startup mistake fails here, visibly. And the property the other tests rely on:
        # readline sets up the terminal before it draws the prompt, so keys sent once the prompt is
        # visible arrive in a stable state.
        child = start()
        child.wait_prompts(1)
        flags = child.flags()
        assert (flags["ECHO"], flags["ICANON"], flags["ISIG"]) == (False, False, True)
        child.settle()
        assert child.prompts_seen() == 1  # and it does not redraw by itself

    def test_a_typed_line_round_trips_and_readline_draws_it(self, start):
        child = start()
        child.wait_prompts(1)
        child.send("hello\r")
        assert child.wait_got(1) == ["'hello'"]
        assert b"hello" in child.after_prompt(1)

    def test_a_multibyte_first_character(self, start):
        child = start()
        child.wait_prompts(1)
        child.send("éx\r")
        assert child.wait_got(1) == ["'éx'"]

    def test_an_arrow_key_as_the_first_key_does_not_leak_into_the_line(self, start):
        # Up with an empty history does nothing; the old code inserted a literal ESC and swallowed "[A".
        child = start()
        child.wait_prompts(1)
        child.send(b"\x1b[Aabc\r")
        assert child.wait_got(1) == ["'abc'"]

    def test_a_burst_in_one_write_arrives_whole(self, start):
        child = start()
        child.wait_prompts(1)
        child.send("hello\r")
        assert child.wait_got(1) == ["'hello'"]

    def test_two_consecutive_prompts_both_work(self, start):
        child = start()
        child.wait_prompts(1)
        child.send("first\r")
        child.wait_got(1)
        child.wait_prompts(2)
        child.send("second\r")
        assert child.wait_got(2) == ["'first'", "'second'"]

    def test_nothing_from_a_burst_shows_up_at_the_next_prompt(self, start):
        # The old code returned 'h' for "hello\r" and then showed "e" at the next prompt with no keypress.
        child = start()
        child.wait_prompts(1)
        child.send("hello\r")
        child.wait_got(1)
        child.wait_prompts(2)
        child.settle(0.5)
        child.send("\r")
        assert child.wait_got(2) == ["'hello'", "''"]

    def test_ctrl_c_at_an_empty_prompt_reprompts(self, start):
        child = start()
        child.wait_prompts(1)
        child.send(b"\x03")
        child.wait_prompts(2)
        child.settle()
        assert child.prompts_seen() == 2  # exactly one re-prompt, and no GOT line for it
        assert child.got_lines() == []
        child.send("ok\r")
        assert child.wait_got(1) == ["'ok'"]

    def test_ctrl_d_ends_the_child_with_status_zero(self, start):
        child = start()
        child.wait_prompts(1)
        child.send(b"\x04")
        assert child.wait_got(1) == ["None"]
        assert child.exit_code() == 0

    def test_piped_stdin_works(self, tmp_path):
        # The old code called tcgetattr on a non-tty and died with termios.error.
        result = subprocess.run(
            [sys.executable, str(PROMPT_CHILD)],
            input=b"hi\n",
            capture_output=True,
            cwd=ROOT,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path),
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert b"GOT 'hi'" in result.stdout
        assert b"Traceback" not in result.stderr


class TestEchoDuringAReply:
    SENTINEL = "qzxj"  # shares no character with the fake reply, so any occurrence in the output is an echo

    def test_nothing_is_echoed_while_the_reply_streams_and_the_tty_comes_back(self, start):
        child = start(STREAM_CHILD, control=True)
        assert set(self.SENTINEL).isdisjoint("Hello world ")
        child.wait_message("token")  # held mid-stream
        assert child.flags() == {"ECHO": False, "ICANON": True, "ISIG": True}
        assert self.SENTINEL.encode() not in child.out
        child.send(self.SENTINEL)
        child.settle()
        assert self.SENTINEL.encode() not in child.out  # the old code echoed it into the reply
        child.release()
        child.wait_message("token")
        child.release()
        child.wait_message("done")  # reply and `with` finished, readline not entered yet
        assert b"DONE" in child.out
        assert child.flags() == {"ECHO": True, "ICANON": True, "ISIG": True}
        child.release()
        child.wait_prompts(1)
        child.send("ok\r")
        # The typed-ahead sentinel was discarded, not submitted, and readline drew what was typed after it.
        assert child.wait_got(1) == ["'ok'"]
        assert b"ok" in child.after_prompt(1)

    def test_ctrl_c_mid_stream_interrupts_the_reply_and_restores_the_tty(self, start):
        child = start(STREAM_CHILD, control=True)
        child.wait_message("token")
        child.send(b"\x03")
        child.wait_message("done")
        assert child.flags() == {"ECHO": True, "ICANON": True, "ISIG": True}
        child.release()
        child.wait_prompts(1)
        child.send(b"\x04")
        assert child.wait_got(1) == ["None"]
        assert child.exit_code() == 0

    def test_the_prompt_after_a_suppressed_turn_draws_what_is_typed(self, start):
        # If ECHO were still off when readline was entered it would draw none of the line (readline
        # skips redisplay when it finds echo off). This is the guard for that failure.
        child = start(STREAM_CHILD, control=True)
        child.wait_message("token")
        child.release()
        child.wait_message("token")
        child.release()
        child.wait_message("done")
        child.release()
        child.wait_prompts(1)
        assert (
            child.flags()["ECHO"] is False
        )  # readline's own prepared mode: it saved ECHO on, then cleared it
        child.send("visible\r")
        child.wait_got(1)
        assert b"visible" in child.after_prompt(1)
        assert PROMPT_SEQ in child.out
