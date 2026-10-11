"""The per-turn stats line: generated tokens, speed, the prompt count and the window's contents (ctx)."""

import contextlib
import io
import re

import pytest

from ui import display_stats

STATS = {"completion_tokens": 142, "tokens_per_second": 38.2, "prompt_tokens": 1203, "context_tokens": 1345}


def render(stats, n_ctx=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        display_stats(stats, n_ctx)
    # Rich colors numbers and wraps the line at the console width; compare the text only.
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()).split())


def render_raw(stats, n_ctx=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        display_stats(stats, n_ctx)
    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


class TestStatsLine:
    def test_the_line_has_the_agreed_shape(self):
        stats = {
            "completion_tokens": 251,
            "tokens_per_second": 52.3,
            "prompt_tokens": 21,
            "context_tokens": 271,
        }
        line = render_raw(stats, 262144)
        assert line == "  251 generated | 52.3 tok/s | 21 prompt tok | ctx: 271 / 262,144 (0.1%)\n"
        assert len(line.rstrip()) <= 80  # it fits a standard terminal without wrapping

    def test_the_generated_count_is_the_servers_thinking_plus_answer(self):
        out = render(STATS)
        assert "142 generated |" in out
        assert "38.2 tok/s" in out and "1,203 prompt tok" in out

    def test_large_counts_have_thousands_separators(self):
        stats = {"completion_tokens": 12345, "prompt_tokens": 1234567, "context_tokens": 1246911}
        out = render(stats, 2000000)
        assert "12,345 generated | 1,234,567 prompt tok | ctx: 1,246,911 / 2,000,000 (62.3%)" in out

    def test_window_use_is_shown_out_of_the_window_with_its_share(self):
        # 1203 prompt + 142 generated: what the window holds now, thinking included
        assert "ctx: 1,345 / 8,192 (16.4%)" in render(STATS, 8192)

    @pytest.mark.parametrize(
        "used, window, shown",
        [(512, 4096, "12.5%"), (1530, 8192, "18.7%"), (2048, 4096, "50.0%")],
    )
    def test_the_share_has_one_decimal(self, used, window, shown):
        assert f"ctx: {used:,} / {window:,} ({shown})" in render({"context_tokens": used}, window)

    def test_without_a_window_size_only_the_count_is_shown(self):
        out = render(STATS)
        assert out.endswith("ctx: 1,345") and "%" not in out

    def test_without_a_count_there_is_no_ctx_segment(self):
        assert "ctx" not in render({"completion_tokens": 5}, 8192)

    def test_a_missing_field_leaves_its_segment_out(self):
        out = render({"completion_tokens": 5})
        assert "5 generated" in out and "prompt tok" not in out and "tok/s" not in out

    def test_empty_stats_print_nothing(self):
        assert render({}) == ""
