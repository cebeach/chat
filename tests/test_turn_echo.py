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
    def __init__(self, tokens, interrupt_after=None, stats=None, truncated=False):
        self.tokens, self.interrupt_after, self.truncated = tokens, interrupt_after, truncated
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


class Session:
    """What a run of main() left behind: the ordered log, the messages shown, the conversation."""

    def __init__(self):
        self.log, self.infos, self.errors, self.conversation, self.client = [], [], [], None, None
        self.chat_calls = []


def run_turn(tmp_path, chat_behaviour):
    """Run main() for one message, then EOF. Returns (log, the messages display_info was given)."""
    session = run_session(tmp_path, chat_behaviour)
    return session.log, session.infos


def run_session(
    tmp_path,
    chat_behaviour,
    inputs=("hello", None),
    check_fit=(100, 4096),
    config=None,
    argv=(),
    setup=None,
    answers=(),
):
    """Run main() over `inputs` (None is EOF). check_fit is what client.check_fit returns,
    or an exception to raise from inside the check; a list gives one result per call.
    argv adds command-line arguments; setup(client) runs before main() to script the mock
    client; answers feed input() (the /remember approval prompt): a string, or an exception to raise."""
    session = Session()
    log, infos = session.log, session.infos
    rec = Recorder(log)

    def logged(name, real=None):
        def wrapper(*args, **kwargs):
            log.append((name, rec.active))
            return real(*args, **kwargs) if real else None

        return wrapper

    def client_chat(**kwargs):
        log.append(("chat", rec.active))
        session.chat_calls.append(kwargs)
        return chat_behaviour()

    def client_check(messages, model=None):
        log.append(("check_fit", rec.active))
        session.check_messages = messages
        result = check_fit.pop(0) if isinstance(check_fit, list) else check_fit
        if isinstance(result, BaseException):
            raise result
        return result

    def display_error(msg):
        session.errors.append(msg)
        log.append(("display_error", rec.active))

    def make_conversation(*args, **kwargs):
        session.conversation = real_conversation(*args, **kwargs)
        return session.conversation

    def display_info(msg):
        infos.append(msg)
        log.append(("display_info", rec.active))

    inputs = iter(inputs)
    client = mock.Mock(server_model="/m/x.gguf", server_n_ctx=4096, think_tags=None)
    client.is_available.return_value = True
    client.chat.side_effect = client_chat
    client.check_fit.side_effect = client_check
    session.client = client
    config = {
        **DEFAULTS,
        "conversations_dir": str(tmp_path),
        "projects_dir": str(tmp_path / "projects"),
        **(config or {}),
    }
    answers = iter(answers)

    def fake_input(prompt=""):
        answer = next(answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    if setup:
        setup(client)
    real_print = chat.console.print
    real_conversation = chat.Conversation
    make_conversation.load = real_conversation.load  # /load and /conversations use the class
    make_conversation.list_saved = real_conversation.list_saved

    with (
        mock.patch.object(chat, "load_config", return_value=config),
        mock.patch.object(chat, "LlamaClient", return_value=client),
        mock.patch.object(sys, "argv", ["chat.py", *argv]),
        mock.patch("builtins.input", fake_input),
        mock.patch.object(chat, "init_readline"),
        mock.patch.object(chat, "save_readline_history"),
        mock.patch.object(chat, "echo_suppressed", rec),
        mock.patch.object(chat, "get_user_input", logged("get_user_input", lambda: next(inputs))),
        mock.patch.object(chat, "display_assistant_stream", logged("display", ui.display_assistant_stream)),
        mock.patch.object(chat, "display_stats", logged("stats", ui.display_stats)),
        mock.patch.object(
            chat, "display_truncated_warning", logged("truncated_warning", ui.display_truncated_warning)
        ),
        mock.patch.object(chat, "_auto_save", logged("auto_save", chat._auto_save)),
        mock.patch.object(chat, "display_error", display_error),
        mock.patch.object(chat, "Conversation", make_conversation),
        mock.patch.object(chat, "display_info", display_info),
        mock.patch.object(chat.console, "print", logged("print", real_print)),
    ):
        chat.main()
    return session


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
        for step in ("check_fit", "chat", "display", "stats", "auto_save"):
            assert step in inside, (step, inside)
        assert (
            inside.index("check_fit")
            < inside.index("chat")
            < inside.index("display")
            < inside.index("stats")
            < inside.index("auto_save")
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


def roles(conversation):
    return [m["role"] for m in conversation.messages]


class TestPreSendCheck:
    """The token check runs inside the suppressed span, before the message joins the conversation."""

    def test_a_prompt_that_does_not_fit_is_refused_and_nothing_is_sent_or_stored(self, tmp_path):
        session = run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=(5000, 4096))
        inside = check_shape(session.log)
        assert "check_fit" in inside and "chat" not in inside and "display_error" in inside
        assert "5,000 tokens" in session.errors[0] and "4,096" in session.errors[0]
        assert session.conversation.messages == []

    def test_the_candidate_is_the_history_plus_the_new_message(self, tmp_path):
        session = run_session(tmp_path, lambda: FakeStream(["r"]), inputs=("one", "two", None))
        assert [m["content"] for m in session.check_messages] == ["one", "r", "two"]

    def test_an_accepted_prompt_skips_the_second_refresh(self, tmp_path):
        session = run_session(tmp_path, lambda: FakeStream(["x"]))
        assert session.chat_calls[0]["refreshed"] is True

    def test_a_prompt_over_80_percent_is_sent_with_a_warning(self, tmp_path):
        with mock.patch.object(chat, "display_context_warning") as warn:
            session = run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=(3400, 4096))
        warn.assert_called_once_with(3400, 4096)
        assert roles(session.conversation) == ["user", "assistant"]

    def test_a_prompt_at_80_percent_or_less_is_sent_without_one(self, tmp_path):
        with mock.patch.object(chat, "display_context_warning") as warn:
            run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=(3276, 4096))
        warn.assert_not_called()

    def test_the_reserve_moves_the_boundary(self, tmp_path):
        config = {"reserve_output_tokens": 1000}
        refused = run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=(3100, 4096), config=config)
        assert refused.conversation.messages == [] and "1,000 kept free" in refused.errors[0]
        sent = run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=(3095, 4096), config=config)
        assert roles(sent.conversation) == ["user", "assistant"]

    def test_context_check_off_counts_nothing_refuses_nothing_warns_nothing(self, tmp_path):
        with mock.patch.object(chat, "display_context_warning") as warn:
            session = run_session(
                tmp_path, lambda: FakeStream(["x"]), check_fit=(10**6, 4096), config={"context_check": False}
            )
        session.client.check_fit.assert_not_called()
        warn.assert_not_called()
        assert roles(session.conversation) == ["user", "assistant"]
        assert session.chat_calls[0]["refreshed"] is False

    def test_an_unknown_window_skips_the_check(self, tmp_path):
        session = run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=(10**6, None))
        assert roles(session.conversation) == ["user", "assistant"]
        assert session.errors == []

    @pytest.mark.parametrize("failure", [ConnectionError("down"), HTTPError("500")])
    def test_a_count_that_cannot_be_made_lets_the_send_meet_the_same_failure(self, tmp_path, failure):
        def behaviour():
            raise failure

        session = run_session(tmp_path, behaviour, check_fit=failure)
        # chat() was called and failed as it does without a check, taking the unanswered message back
        assert len(session.chat_calls) == 1 and session.chat_calls[0]["refreshed"] is False
        assert session.conversation.messages == []

    def test_an_unreadable_token_count_lets_the_send_proceed_too(self, tmp_path):
        session = run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=ValueError("not json"))
        assert roles(session.conversation) == ["user", "assistant"] and session.errors == []

    def test_a_count_that_cannot_be_made_on_a_later_turn_leaves_the_earlier_messages_intact(self, tmp_path):
        replies = iter([FakeStream(["r"]), ConnectionError("down")])

        def behaviour():
            reply = next(replies)
            if isinstance(reply, BaseException):
                raise reply
            return reply

        failure = ConnectionError("down")
        session = run_session(
            tmp_path, behaviour, inputs=("one", "two", None), check_fit=[(10, 4096), failure]
        )
        assert [m["content"] for m in session.conversation.messages] == ["one", "r"]

    def test_ctrl_c_during_the_check_cancels_the_send_and_keeps_the_program_running(self, tmp_path):
        session = run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=KeyboardInterrupt())
        inside = check_shape(session.log)
        assert "chat" not in inside and "Cancelled." in session.infos
        assert session.conversation.messages == []


class TestRefusedCommands:
    def test_a_refused_retry_restores_the_removed_messages(self, tmp_path):
        session = run_session(
            tmp_path,
            lambda: FakeStream(["reply"]),
            inputs=("hello", "/retry", None),
            check_fit=[(10, 4096), (5000, 4096)],
        )
        assert [m["content"] for m in session.conversation.messages] == ["hello", "reply"]
        assert len(session.chat_calls) == 1 and len(session.errors) == 1

    def test_an_accepted_retry_sends_again(self, tmp_path):
        session = run_session(
            tmp_path, lambda: FakeStream(["reply"]), inputs=("hello", "/retry", None), check_fit=(10, 4096)
        )
        assert [m["content"] for m in session.conversation.messages] == ["hello", "reply"]
        assert len(session.chat_calls) == 2

    def test_a_refused_read_leaves_the_conversation_unchanged(self, tmp_path, monkeypatch):
        (tmp_path / "big.txt").write_text("lots of text")
        monkeypatch.chdir(tmp_path)
        session = run_session(
            tmp_path,
            lambda: FakeStream(["x"]),
            inputs=("/read big.txt", None),
            check_fit=(5000, 4096),
        )
        assert session.conversation.messages == [] and session.chat_calls == []
        assert "5,000 tokens" in session.errors[0]

    def test_the_prompt_size_is_shown_after_a_file_is_included(self, tmp_path, monkeypatch):
        (tmp_path / "f.txt").write_text("some text")
        monkeypatch.chdir(tmp_path)
        with mock.patch.object(chat, "display_prompt_size") as shown:
            run_session(
                tmp_path, lambda: FakeStream(["x"]), inputs=("see @@<f.txt>", None), check_fit=(1000, 4096)
            )
        shown.assert_called_once_with(1000, 4096)

    def test_no_prompt_size_without_an_included_file(self, tmp_path):
        with mock.patch.object(chat, "display_prompt_size") as shown:
            run_session(tmp_path, lambda: FakeStream(["x"]), check_fit=(1000, 4096))
        shown.assert_not_called()


class TestTruncatedReply:
    """A reply the server stopped because the context window filled is flagged, not shown as finished."""

    def test_the_warning_follows_the_reply_inside_the_suppressed_span(self, tmp_path):
        log, _ = run_turn(tmp_path, lambda: FakeStream(["Half a sentence"], truncated=True))
        inside = check_shape(log)
        assert "truncated_warning" in inside
        assert inside.index("display") < inside.index("truncated_warning") < inside.index("auto_save")

    def test_no_warning_for_a_reply_that_finished(self, tmp_path):
        log, _ = run_turn(tmp_path, lambda: FakeStream(["Done."]))
        assert "truncated_warning" not in [n for n, _ in log]

    def test_the_reply_is_still_stored(self, tmp_path):
        session = run_session(tmp_path, lambda: FakeStream(["Half a sentence"], truncated=True))
        assert [m["content"] for m in session.conversation.messages] == ["hello", "Half a sentence"]

    def test_it_says_when_nothing_of_the_answer_was_written(self, tmp_path, capsys):
        def cut_off_while_thinking():
            stream = FakeStream(["<think>still working it out"], truncated=True)
            stream.think_tags = ("<think>", "</think>", "detected")
            return stream

        session = run_session(tmp_path, cut_off_while_thinking)
        out = " ".join(capsys.readouterr().out.split())
        assert "cut off because the context window is full" in out
        assert "still thinking, so there is no answer" in out
        assert session.conversation.messages[-1]["content"] == ""

    def test_an_ordinary_cut_off_reply_does_not_claim_the_answer_is_missing(self, tmp_path, capsys):
        run_session(tmp_path, lambda: FakeStream(["Half a sentence"], truncated=True))
        out = " ".join(capsys.readouterr().out.split())
        assert "cut off because the context window is full" in out and "no answer" not in out
