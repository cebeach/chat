"""The per-turn stats line: generated tokens (thinking + answer), speed and the server's prompt count."""

import contextlib
import io
import re

from ui import display_stats

STATS = {"completion_tokens": 142, "tokens_per_second": 38.2, "prompt_tokens": 1203, "context_tokens": 1345}


def render(stats):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        display_stats(stats)
    # Rich colors numbers; compare the text only.
    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


class TestStatsLine:
    def test_labels_the_server_count_as_thinking_plus_answer(self):
        out = render(STATS)
        assert "142 generated (thinking + answer)" in out
        assert "38.2 tok/s" in out and "1203 prompt tokens" in out

    def test_context_use_is_not_shown(self):
        # prompt + generated overstates the next prompt for a thinking model, so nothing shows it
        assert "ctx" not in render(STATS) and "1,345" not in render(STATS)

    def test_a_missing_field_leaves_its_segment_out(self):
        out = render({"completion_tokens": 5})
        assert "5 generated" in out and "prompt tokens" not in out and "tok/s" not in out

    def test_empty_stats_print_nothing(self):
        assert render({}) == ""
