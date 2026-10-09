"""/info: the token breakdown, the window and the usage ratio."""

import contextlib
import io
import re

from ui import display_conversation_info

SUMMARY = {"messages": 2, "user_messages": 1, "assistant_messages": 1, "words": 4, "characters": 20}
COUNTS = {"system": 10, "user": 100, "assistant": 200, "overhead": 14, "total": 324, "n_ctx": 8192}


def render_fn(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    # Rich colors numbers and draws table borders; compare the text only.
    text = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
    return " ".join(re.sub(r"[│┃┏┓┡┩└┘━─┳╇┴]", " ", text).split())


def info(counts=None, context_length=None):
    return render_fn(lambda: display_conversation_info(SUMMARY, counts, context_length))


class TestInfoTable:
    def test_shows_each_share_the_total_and_the_ratio(self):
        out = info(COUNTS, 8192)
        assert "Tokens: system prompt 10" in out
        assert "Tokens: your messages 100" in out
        assert "Tokens: AI replies 200" in out
        assert "Tokens: template ≈ 14" in out
        assert "Prompt tokens 324" in out
        assert "Context window 8,192 tokens" in out
        assert "Window used 4.0%" in out

    def test_without_counts_only_the_window_shows(self):
        out = info(None, 8192)
        assert "Context window 8,192 tokens" in out
        assert "Prompt tokens" not in out and "Window used" not in out

    def test_the_window_the_server_reported_with_the_counts_wins(self):
        out = info({**COUNTS, "n_ctx": 2048}, 8192)
        assert "Context window 2,048 tokens" in out and "Window used 15.8%" in out

    def test_unknown_window_shows_the_counts_but_no_ratio(self):
        out = info({**COUNTS, "n_ctx": None}, None)
        assert "Context window unknown" in out
        assert "Prompt tokens 324" in out and "Window used" not in out

    def test_usage_above_the_window_renders(self):
        assert "Window used 109.9%" in info({**COUNTS, "total": 9000}, None)
