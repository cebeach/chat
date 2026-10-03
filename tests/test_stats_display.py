"""The per-turn stats line shows context usage when the context length is known."""

import contextlib
import io
import re

from ui import display_stats

STATS = {"completion_tokens": 142, "tokens_per_second": 38.2, "prompt_tokens": 1203, "context_tokens": 1345}


def render(stats, context_length=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        display_stats(stats, context_length)
    # Rich colors numbers; compare the text only.
    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


class TestStatsLine:
    def test_shows_used_total_and_ratio(self):
        assert "ctx 1,345 / 8,192 (16.4%)" in render(STATS, 8192)

    def test_unknown_context_length_omits_segment(self):
        for length in (None, 0):
            out = render(STATS, length)
            assert "ctx" not in out
            assert "142 tokens" in out

    def test_usage_above_the_limit_renders(self):
        assert "(109.9%)" in render({**STATS, "context_tokens": 9000}, 8192)

    def test_missing_context_tokens_omits_segment(self):
        stats = {k: v for k, v in STATS.items() if k != "context_tokens"}
        assert "ctx" not in render(stats, 8192)

    def test_empty_stats_print_nothing(self):
        assert render({}, 8192) == ""
