"""echo_suppressed() must cover the whole model turn in chat.main(), and nothing else.

test_prompt_pty.py's child scripts wrap their own `with`, so they would pass even if chat.py wrapped
the wrong span. This runs one turn of the real chat.main() and records, in one ordered log, whether
suppression was active when each step ran: from the request through the stats line and the autosave
file write to the trailing blank line it must be on; at both get_user_input() calls it must be off,
because readline entered with ECHO off draws none of what the user types.

Follows the existing main() test in test_served_model.py: LlamaClient is a mock, not FakeServer,
because main() starts with GET /health, which FakeServer does not answer, and because a test has to be
able to raise from inside the stream.
"""

import sys
from unittest import mock

import pytest
from requests.exceptions import ConnectionError, HTTPError

import chat
import ui
from config import DEFAULTS


class Recorder:
    """Stands in for echo_suppressed: a context manager that logs enter/exit and exposes `active`."""

    def __init__(self, log):
        self.log, self.active = log, False

    def __call__(self):
        return self

    def __enter__(self):
        self.active = True
        self.log.append(("enter", True))

    def __exit__(self, *exc):
        self.log.append(("exit", self.active))
        self.active = False
        return False


class FakeStream:
    def __init__(self, tokens, interrupt_after=None, stats=None):
        self.tokens, self.interrupt_after = tokens, interrupt_after
        self.stats = {"completion_tokens": 1} if stats is None else stats
        self.server_model, self.server_n_ctx, self.think_tags, self.model = (
            "/m/x.gguf",
            4096,
            None,
            "/m/x.gguf",
        )

    def __iter__(self):
        for i, token in enumerate(self.tokens):
            if self.interrupt_after == i:
                raise KeyboardInterrupt
            yield token


def run_turn(tmp_path, chat_behaviour):
    """Run main() for one message, then EOF. Returns (log, the messages display_info was given)."""
    log, infos = [], []
    rec = Recorder(log)

    def logged(name, real=None):
        def wrapper(*args, **kwargs):
            log.append((name, rec.active))
            return real(*args, **kwargs) if real else None

        return wrapper

    def client_chat(**kwargs):
        log.append(("chat", rec.active))
        return chat_behaviour()

    def display_info(msg):
        infos.append(msg)
        log.append(("display_info", rec.active))

    inputs = iter(["hello", None])
    client = mock.Mock(server_model="/m/x.gguf", server_n_ctx=4096, think_tags=None)
    client.is_available.return_value = True
    client.chat.side_effect = client_chat
    config = {**DEFAULTS, "conversations_dir": str(tmp_path)}
    real_print = chat.console.print

    with (
        mock.patch.object(chat, "load_config", return_value=config),
        mock.patch.object(chat, "LlamaClient", return_value=client),
        mock.patch.object(sys, "argv", ["chat.py"]),
        mock.patch.object(chat, "init_readline"),
        mock.patch.object(chat, "save_readline_history"),
        mock.patch.object(chat, "echo_suppressed", rec),
        mock.patch.object(chat, "get_user_input", logged("get_user_input", lambda: next(inputs))),
        mock.patch.object(chat, "display_assistant_stream", logged("display", ui.display_assistant_stream)),
        mock.patch.object(chat, "display_stats", logged("stats", ui.display_stats)),
        mock.patch.object(chat, "_auto_save", logged("auto_save", chat._auto_save)),
        mock.patch.object(chat, "display_error", logged("display_error")),
        mock.patch.object(chat, "display_info", display_info),
        mock.patch.object(chat.console, "print", logged("print", real_print)),
    ):
        chat.main()
    return log, infos


def check_shape(log):
    """The log has exactly one enter/exit; everything between them is active, everything else is not."""
    names = [n for n, _ in log]
    assert names.count("enter") == 1 and names.count("exit") == 1
    i, j = names.index("enter"), names.index("exit")
    assert all(active for _, active in log[i + 1 : j]), log[i + 1 : j]
    outside = [(n, a) for n, a in log[:i] + log[j + 1 :] if n not in ("enter", "exit")]
    assert all(not a for _, a in outside), outside
    # get_user_input ran before the turn and again after it, both with suppression off.
    assert "get_user_input" in names[:i] and "get_user_input" in names[j + 1 :]
    # The trailing blank line is the last thing inside, so ECHO comes back just before the prompt.
    assert names[j - 1] == "print"
    return names[i + 1 : j]


class TestWhichSpanIsSuppressed:
    def test_a_normal_turn_is_suppressed_from_the_request_to_the_trailing_blank_line(self, tmp_path):
        log, infos = run_turn(tmp_path, lambda: FakeStream(["Hello ", "world"]))
        inside = check_shape(log)
        for step in ("chat", "display", "stats", "auto_save"):
            assert step in inside, (step, inside)
        assert (
            inside.index("chat") < inside.index("display") < inside.index("stats") < inside.index("auto_save")
        )
        assert infos == []

    def test_a_connection_error_is_reported_while_still_suppressed(self, tmp_path):
        def behaviour():
            raise ConnectionError("down")

        log, _ = run_turn(tmp_path, behaviour)
        inside = check_shape(log)
        assert "display_error" in inside and "auto_save" not in inside

    def test_an_http_error_is_reported_while_still_suppressed(self, tmp_path):
        def behaviour():
            raise HTTPError("500")

        log, _ = run_turn(tmp_path, behaviour)
        assert "display_error" in check_shape(log)

    def test_an_interrupt_while_waiting_for_the_server_prints_response_interrupted(self, tmp_path):
        def behaviour():
            raise KeyboardInterrupt

        log, infos = run_turn(tmp_path, behaviour)
        inside = check_shape(log)
        assert infos == ["Response interrupted."]
        assert "display_info" in inside and "display" not in inside

    def test_an_interrupt_mid_stream_is_handled_by_the_display_and_the_turn_completes(self, tmp_path):
        # display_assistant_stream catches this one itself: " [interrupted]" is appended and nothing
        # reaches chat.py's handler, so there is no "Response interrupted." message. The stats are
        # probably empty after an interrupted stream; the call still happens (it prints nothing).
        log, infos = run_turn(tmp_path, lambda: FakeStream(["Hello ", "world"], interrupt_after=1, stats={}))
        inside = check_shape(log)
        assert infos == []
        for step in ("chat", "display", "stats", "auto_save"):
            assert step in inside, (step, inside)
