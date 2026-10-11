"""The stats line under a reply, over the real main(): window use as a share of the window.

Uses run_session from test_turn_echo.py (a mock LlamaClient). Each turn re-reads the window size
from the stream, so FakeStream's 4096 is the denominator here.
"""

from tests.test_turn_echo import FakeStream, run_session

STATS = {"completion_tokens": 800, "prompt_tokens": 1530, "context_tokens": 2330}


def stats_out(capsys):
    return " ".join(capsys.readouterr().out.split())


def test_the_line_shows_what_the_window_holds_and_its_share(tmp_path, capsys):
    run_session(tmp_path, lambda: FakeStream(["Hi"], stats=dict(STATS)))
    out = stats_out(capsys)
    assert "1,530 prompt tok | ctx: 2,330 / 4,096 (56.9%)" in out


def test_the_sum_of_prompt_and_generated_is_not_assumed(tmp_path, capsys):
    # A context shift leaves the slot with fewer tokens than prompt + generated.
    run_session(tmp_path, lambda: FakeStream(["Hi"], stats={**STATS, "context_tokens": 1900}))
    assert "ctx: 1,900 / 4,096 (46.4%)" in stats_out(capsys)


def test_stats_off_prints_no_line(tmp_path, capsys):
    run_session(tmp_path, lambda: FakeStream(["Hi"], stats=dict(STATS)), inputs=("/stats", "hello", None))
    assert "ctx" not in stats_out(capsys)


def test_without_a_window_size_the_count_is_shown_alone(tmp_path, capsys):
    def no_window(client):
        client.server_n_ctx = None  # startup sets state.context_length from the client

    def stream():
        s = FakeStream(["Hi"], stats=dict(STATS))
        s.server_n_ctx = None  # a stream reporting None leaves the earlier value alone
        return s

    run_session(tmp_path, stream, setup=no_window, check_fit=(100, None))
    out = stats_out(capsys)
    assert "ctx: 2,330" in out and "ctx: 2,330 /" not in out
