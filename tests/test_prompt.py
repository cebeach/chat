"""get_user_input() hands every key to readline; echo_suppressed() keeps the tty quiet during a turn.

Before this, get_user_input drew its own prompt, switched the tty to raw and read the first key
itself through sys.stdin, which swallowed the rest of a fast burst and showed it again at later
prompts. These tests pin the replacement. Real-terminal behaviour is in test_prompt_pty.py.
"""

import re
import signal
import termios
from unittest import mock

import pytest

import ui

LFLAG = termios.ECHO | termios.ICANON | termios.ISIG
SAVED = [0, 0, 0, LFLAG, 0, 0, [b"\0"] * 32]


class StrictStdin:
    """A sys.stdin stand-in: a tty if asked, and anything beyond isatty()/fileno() is a test failure."""

    def __init__(self, tty=True):
        self._tty = tty

    def isatty(self):
        return self._tty

    def fileno(self):
        return 99

    def __getattr__(self, name):
        raise AssertionError(f"touched sys.stdin.{name}")


@pytest.fixture
def stdin(monkeypatch):
    fake = StrictStdin()
    monkeypatch.setattr("sys.stdin", fake)
    return fake


@pytest.fixture
def tcflush():
    with mock.patch.object(termios, "tcflush") as m:
        yield m


class TestGetUserInput:
    def test_input_gets_a_colour_prompt_with_the_escapes_enclosed(self, stdin, tcflush):
        with mock.patch("builtins.input", return_value="x") as inp:
            ui.get_user_input()
        assert inp.call_count == 1
        prompt = inp.call_args.args[0]
        # readline measures the prompt width: every escape sequence must sit between \x01 and \x02.
        ignored = re.findall("\x01(.*?)\x02", prompt)
        assert ignored and all(seg.startswith("\x1b[") and seg.endswith("m") for seg in ignored)
        assert re.sub("\x01.*?\x02", "", prompt) == ">>> "
        assert "\x1b" not in re.sub("\x01.*?\x02", "", prompt)

    def test_it_never_reads_sys_stdin(self, stdin, tcflush):
        # The carry-over came from sys.stdin.read(1) buffering bytes that readline never saw. The
        # stand-in raises on every attribute except isatty() and fileno(), so any read would fail.
        with mock.patch("builtins.input", return_value="hello"):
            assert ui.get_user_input() == "hello"

    def test_pending_input_is_flushed_once_and_before_the_prompt(self, stdin):
        order = mock.Mock()
        with (
            mock.patch.object(termios, "tcflush", order.flush),
            mock.patch("builtins.input", order.input),
        ):
            order.input.return_value = "x"
            ui.get_user_input()
        assert [c[0] for c in order.mock_calls] == ["flush", "input"]
        assert order.flush.call_args.args == (99, termios.TCIFLUSH)

    def test_a_reprompt_after_ctrl_c_does_not_flush_again(self, stdin, tcflush, capsys):
        with mock.patch("builtins.input", side_effect=[KeyboardInterrupt, KeyboardInterrupt, "typed"]) as inp:
            assert ui.get_user_input() == "typed"
        assert tcflush.call_count == 1
        assert inp.call_count == 3
        assert capsys.readouterr().out == "\n\n"

    def test_ctrl_d_returns_none_after_a_newline(self, stdin, tcflush, capsys):
        with mock.patch("builtins.input", side_effect=EOFError):
            assert ui.get_user_input() is None
        assert capsys.readouterr().out == "\n"

    def test_an_empty_line_is_returned_as_is(self, stdin, tcflush):
        with mock.patch("builtins.input", return_value=""):
            assert ui.get_user_input() == ""

    def test_a_flush_error_is_swallowed(self, stdin):
        with (
            mock.patch.object(termios, "tcflush", side_effect=termios.error(25, "not a tty")),
            mock.patch("builtins.input", return_value="x"),
        ):
            assert ui.get_user_input() == "x"

    def test_without_a_tty_it_does_not_flush_and_still_reads_a_line(self, monkeypatch, tcflush):
        monkeypatch.setattr("sys.stdin", StrictStdin(tty=False))
        with mock.patch("builtins.input", return_value="piped"):
            assert ui.get_user_input() == "piped"
        tcflush.assert_not_called()


class TestRedrawAfterResume:
    def test_it_writes_the_visible_prompt_and_the_line_on_a_cleared_row(self, capsys, monkeypatch):
        monkeypatch.setattr(ui.readline, "get_line_buffer", lambda: "abc")
        ui._redraw_after_resume(signal.SIGCONT, None)
        # No \x01/\x02 (those are only for readline's width calculation), and nothing but the row.
        assert capsys.readouterr().out == "\r\033[K\033[1;32m>>> \033[0mabc"

    def test_it_is_installed_only_while_waiting_at_the_prompt_and_restored_after(self, stdin, tcflush):
        before = signal.getsignal(signal.SIGCONT)
        seen = []

        def fake_input(prompt):
            seen.append(signal.getsignal(signal.SIGCONT))
            return "x"

        with mock.patch("builtins.input", fake_input):
            ui.get_user_input()
        assert seen == [ui._redraw_after_resume]
        assert signal.getsignal(signal.SIGCONT) == before

    def test_it_is_restored_after_ctrl_c_ctrl_d_and_installed_again_for_each_reprompt(
        self, stdin, tcflush, capsys
    ):
        before = signal.getsignal(signal.SIGCONT)
        seen = []

        def fake_input(prompt):
            seen.append(signal.getsignal(signal.SIGCONT))
            if len(seen) == 1:
                raise KeyboardInterrupt
            raise EOFError

        with mock.patch("builtins.input", fake_input):
            assert ui.get_user_input() is None
        assert seen == [ui._redraw_after_resume, ui._redraw_after_resume]
        assert signal.getsignal(signal.SIGCONT) == before


class FakeTty:
    """tcgetattr/tcsetattr stand-ins that record every tcsetattr call as (when, attrs)."""

    def __init__(self, fail_on=()):
        self.calls = []
        self._fail_on = dict(fail_on)  # call number (1-based) -> exception to raise instead

    def get(self, fd):
        return [list(x) if isinstance(x, list) else x for x in SAVED]

    def set(self, fd, when, attrs):
        n = len(self.calls) + 1
        self.calls.append((when, [list(x) if isinstance(x, list) else x for x in attrs]))
        if n in self._fail_on:
            raise self._fail_on[n]


@pytest.fixture
def tty(stdin):
    def install(fail_on=()):
        fake = FakeTty(fail_on)
        patches = (
            mock.patch.object(termios, "tcgetattr", fake.get),
            mock.patch.object(termios, "tcsetattr", fake.set),
        )
        for p in patches:
            p.start()
        return fake

    installed = []
    yield lambda **kw: installed.append(install(**kw)) or installed[-1]
    mock.patch.stopall()


QUIET = [0, 0, 0, LFLAG & ~termios.ECHO, 0, 0, [b"\0"] * 32]


class TestEchoSuppressed:
    def test_only_the_echo_bit_is_cleared_and_it_is_applied_immediately(self, tty):
        fake = tty()
        with ui.echo_suppressed():
            assert fake.calls == [(termios.TCSANOW, QUIET)]
        assert fake.calls[-1] == (termios.TCSANOW, SAVED)
        assert all(when == termios.TCSANOW for when, _ in fake.calls)

    def test_the_saved_attributes_come_back_when_the_body_raises(self, tty):
        fake = tty()
        with pytest.raises(ValueError):
            with ui.echo_suppressed():
                raise ValueError("boom")
        assert fake.calls[-1] == (termios.TCSANOW, SAVED)

    def test_the_saved_attributes_come_back_on_keyboard_interrupt(self, tty):
        fake = tty()
        with pytest.raises(KeyboardInterrupt):
            with ui.echo_suppressed():
                raise KeyboardInterrupt
        assert fake.calls[-1] == (termios.TCSANOW, SAVED)

    def test_an_interrupt_during_the_restore_is_retried_and_the_restore_completes(self, tty):
        fake = tty(fail_on={2: KeyboardInterrupt()})  # call 1 = quiet, call 2 = the first restore attempt
        with ui.echo_suppressed():
            pass
        assert [a for _, a in fake.calls] == [QUIET, SAVED, SAVED]  # the retry applied SAVED

    def test_the_bodys_exception_still_propagates_when_the_restore_is_interrupted(self, tty):
        fake = tty(fail_on={2: KeyboardInterrupt()})
        with pytest.raises(ValueError):
            with ui.echo_suppressed():
                raise ValueError("boom")
        assert fake.calls[-1] == (termios.TCSANOW, SAVED)

    def test_an_interrupt_during_entry_is_retried_not_propagated(self, tty):
        # Documented side effect: that Ctrl-C is dropped, the quiet attributes still apply, the body runs.
        fake = tty(fail_on={1: KeyboardInterrupt()})
        ran = []
        with ui.echo_suppressed():
            ran.append(fake.calls[-1][1])
        assert ran == [QUIET]
        assert [a for _, a in fake.calls] == [QUIET, QUIET, SAVED]

    def test_a_tty_error_on_restore_is_swallowed_and_hides_nothing(self, tty):
        fake = tty(fail_on={2: termios.error(5, "gone")})
        with ui.echo_suppressed():
            pass  # no exception
        fake = tty(fail_on={2: termios.error(5, "gone")})
        with pytest.raises(ValueError):
            with ui.echo_suppressed():
                raise ValueError("boom")  # not replaced by the termios.error

    def test_without_a_tty_it_touches_nothing_and_runs_the_body(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StrictStdin(tty=False))
        with mock.patch.object(termios, "tcgetattr") as get, mock.patch.object(termios, "tcsetattr") as set_:
            with ui.echo_suppressed():
                ran = True
        assert ran
        get.assert_not_called()
        set_.assert_not_called()

    def test_a_tcgetattr_error_makes_it_a_no_op_that_still_runs_the_body(self, stdin):
        with (
            mock.patch.object(termios, "tcgetattr", side_effect=termios.error(25, "no")),
            mock.patch.object(termios, "tcsetattr") as set_,
        ):
            with ui.echo_suppressed():
                ran = True
        assert ran
        set_.assert_not_called()
