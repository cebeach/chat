"""The per-turn stats line shows context usage when the context length is known."""

import contextlib
import io
import re
import unittest

from ui import display_stats

STATS = {"completion_tokens": 142, "tokens_per_second": 38.2, "prompt_tokens": 1203, "context_tokens": 1345}


def render(stats, context_length=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        display_stats(stats, context_length)
    # Rich colors numbers; compare the text only.
    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


class StatsLineTests(unittest.TestCase):
    def test_shows_used_total_and_ratio(self):
        self.assertIn("ctx 1,345 / 8,192 (16.4%)", render(STATS, 8192))

    def test_unknown_context_length_omits_segment(self):
        for length in (None, 0):
            out = render(STATS, length)
            self.assertNotIn("ctx", out)
            self.assertIn("142 tokens", out)

    def test_usage_above_the_limit_renders(self):
        self.assertIn("(109.9%)", render({**STATS, "context_tokens": 9000}, 8192))

    def test_missing_context_tokens_omits_segment(self):
        stats = {k: v for k, v in STATS.items() if k != "context_tokens"}
        self.assertNotIn("ctx", render(stats, 8192))

    def test_empty_stats_print_nothing(self):
        self.assertEqual(render({}, 8192), "")


if __name__ == "__main__":
    unittest.main()
