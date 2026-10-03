"""/info shows the context window, usage and ratio; stale usage is reset."""

import contextlib
import io
import re

import pytest

from chat import State, handle_command
from config import DEFAULTS
from conversation import Conversation
from ui import display_conversation_info

SUMMARY = {"messages": 2, "user_messages": 1, "assistant_messages": 1, "words": 4, "characters": 20}
STATS = {"prompt_tokens": 1203, "completion_tokens": 142, "context_tokens": 1345}


def render_fn(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    # Rich colors numbers and draws table borders; compare the text only.
    text = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
    return " ".join(re.sub(r"[│┃┏┓┡┩└┘━─┳╇┴]", " ", text).split())


def info(stats=None, context_length=None):
    return render_fn(lambda: display_conversation_info(SUMMARY, stats, context_length))


class TestInfoTable:
    def test_shows_window_used_and_ratio(self):
        out = info(STATS, 8192)
        assert "Context window 8,192 tokens" in out
        assert "Context used 1,345 tokens" in out
        assert "Context usage 16.4%" in out

    def test_no_reply_yet_shows_only_the_window(self):
        out = info({}, 8192)
        assert "Context window 8,192 tokens" in out
        assert "Context used" not in out
        assert "Context usage" not in out

    def test_unknown_window_shows_used_but_no_ratio(self):
        out = info(STATS, None)
        assert "Context window unknown" in out
        assert "Context used 1,345 tokens" in out
        assert "Context usage" not in out

    def test_usage_above_the_window_renders(self):
        assert "Context usage 109.9%" in info({"context_tokens": 9000}, 8192)


class TestStaleStats:
    def make_state(self):
        config = {**DEFAULTS, "conversations_dir": self.tmp}
        return State(model="/m/x.gguf", config=config, context_length=8192, last_stats=dict(STATS))

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)

    def run_cmd(self, cmd, args, conversation, state):
        render_fn(lambda: handle_command(cmd, args, None, conversation, state))

    def test_clear_resets_last_stats(self):
        state = self.make_state()
        self.run_cmd("/clear", "", Conversation(), state)
        assert state.last_stats == {}

    def test_retry_resets_last_stats(self):
        state = self.make_state()
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("a")
        self.run_cmd("/retry", "", conv, state)
        assert state.last_stats == {}

    def test_successful_load_resets_last_stats(self):
        saved = Conversation()
        saved.add_user("q")
        saved.add_assistant("a")
        saved.save(self.tmp, name="c")
        state = self.make_state()
        self.run_cmd("/load", "c", Conversation(), state)
        assert state.last_stats == {}

    def test_failed_load_keeps_last_stats(self):
        state = self.make_state()
        self.run_cmd("/load", "missing", Conversation(), state)
        assert state.last_stats == STATS

    def test_retry_with_nothing_to_retry_keeps_last_stats(self):
        state = self.make_state()
        self.run_cmd("/retry", "", Conversation(), state)
        assert state.last_stats == STATS
