"""The sampling options against a real llama-server started with known launch flags.

Every launch value is passed with `llama_args` (docs/testing.md), so there is nothing to start by
hand. /props reports only those launch defaults, never what a request sent, which is why the app
keeps the user's overrides itself; these tests check both halves against a real server: the
defaults the command line sets are what /set shows, and an override actually reaches /completion.
"""

import re

import pytest
from pytest import approx

from chat import State, handle_command
from config import DEFAULTS
from conversation import Conversation
from tests.pty_helpers import ROOT, PtyChild
from tests.test_conversation_info import render_fn

# option -> (launch flag, value); /props names the options like /completion does.
LAUNCH = {
    "temperature": ("--temp", 0.6),
    "top_p": ("--top-p", 0.9),
    "min_p": ("--min-p", 0.1),
    "repeat_penalty": ("--repeat-penalty", 1.3),
    "seed": ("--seed", 42),
}
# --predict is launched too, but /props cannot report it (see the test below).
PREDICT = 64
FLAGS = (*(f"{flag} {value}" for flag, value in LAUNCH.values()), f"--predict {PREDICT}")
OTHER = {"temperature": ("--temp", 0.2), "min_p": ("--min-p", 0.02)}
OTHER_FLAGS = tuple(f"{flag} {value}" for flag, value in OTHER.values())

STORY = "Write a short story about a lighthouse keeper."


def make_state(client):
    return State(model=client.server_model, config={**DEFAULTS}, context_length=client.server_n_ctx)


def run_set(client, state, args=""):
    return render_fn(lambda: handle_command("/set", args, client, Conversation(), state))


def reply(client, options):
    stream = client.chat(client.server_model, [{"role": "user", "content": STORY}], options)
    return "".join(stream), stream.stats


@pytest.mark.integration
@pytest.mark.llama_args(*FLAGS)
class TestLaunchFlagsReachSet:
    def test_props_reports_every_launch_value(self, llama_client):
        defaults = llama_client.sampling_defaults()
        for option, (_, value) in LAUNCH.items():
            assert defaults[option] == approx(value), option  # float32 is widened by the server

    def test_props_does_not_report_the_launch_n_predict(self, llama_client):
        """llama-server builds the /props params from a default task, so n_predict is always -1.

        If this fails, llama.cpp now reports it: drop ui._NOT_REPORTED and show the value.
        """
        assert llama_client.sampling_defaults()["n_predict"] == -1

    def test_set_shows_the_launch_values_as_the_server_column(self, llama_client):
        out = run_set(llama_client, make_state(llama_client))
        for option, value in [("temperature", "0.6"), ("top_p", "0.9"), ("min_p", "0.1")]:
            assert f"{option} {value} -" in out
        assert "repeat_penalty 1.3 -" in out and "seed 42 -" in out
        assert "n_predict not reported -" in out

    def test_an_override_is_shown_beside_the_launch_value(self, llama_client):
        state = make_state(llama_client)
        run_set(llama_client, state, "temperature 0.2")
        assert "temperature 0.6 0.2" in run_set(llama_client, state)

    def test_an_n_predict_override_caps_the_reply_below_the_launch_value(self, llama_client):
        _, stats = reply(llama_client, {"seed": 1, "temperature": 0})
        assert 5 < stats["completion_tokens"] <= PREDICT  # the launch --predict applies
        _, stats = reply(llama_client, {"seed": 1, "temperature": 0, "n_predict": 5})
        assert stats["completion_tokens"] <= 5

    def test_a_request_n_predict_of_minus_one_does_not_lift_the_launch_limit(self, llama_client):
        """llama-server reads -1 as "use --predict", so the app must not offer -1 as an override.

        If this fails, a request can now lift the limit: allow -1 in chat._OPTION_RULES again.
        """
        _, stats = reply(llama_client, {"seed": 1, "temperature": 0, "n_predict": -1})
        assert stats["completion_tokens"] <= PREDICT

    def test_min_p_and_repeat_penalty_overrides_are_accepted(self, llama_client):
        text, stats = reply(llama_client, {"seed": 1, "min_p": 0.05, "repeat_penalty": 1.1, "n_predict": 8})
        assert text and stats["completion_tokens"] > 0  # a 400 would have raised

    def test_the_seed_override_is_sent_and_wins_over_the_launch_seed(self, llama_client):
        # Two identical runs alone would prove nothing: the launch --seed 42 is deterministic too.
        free = {"temperature": 1.0, "top_p": 1.0, "min_p": 0.0, "n_predict": 48}
        one, _ = reply(llama_client, {**free, "seed": 1})
        again, _ = reply(llama_client, {**free, "seed": 1})
        two, _ = reply(llama_client, {**free, "seed": 2})
        assert one == again
        assert one != two

    def test_the_real_repl_shows_and_applies_them(self, llama_server, tmp_path):
        child = PtyChild(ROOT / "chat.py", tmp_path, args=["--url", llama_server.url])
        try:
            child.wait_prompts(1, timeout=60)
            child.send("/set\r")
            child.wait_prompts(2, timeout=30)
            table = plain(child.after_prompt(1))
            assert "temperature 0.6 -" in table and "n_predict not reported -" in table
            child.send("/set n_predict 5\r")
            child.wait_prompts(3, timeout=30)
            assert "n_predict set to 5." in plain(child.after_prompt(2))
            child.send("Reply with the single word: ok\r")
            child.wait_prompts(4, timeout=180)
            child.send(b"\x04")
            assert child.exit_code(timeout=30) == 0
        finally:
            child.close()


@pytest.mark.integration
@pytest.mark.llama_args(*OTHER_FLAGS)
class TestAnotherLaunch:
    """Other launch flags give other server values: /set reads them, nothing is remembered."""

    def test_set_shows_the_other_values(self, llama_client):
        out = run_set(llama_client, make_state(llama_client))
        assert "temperature 0.2 -" in out and "min_p 0.02 -" in out
        assert "temperature 0.6" not in out


def plain(raw):
    """Terminal output as plain text: no colors, no table borders, one space between words."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    return " ".join(re.sub(r"[│┃┏┓┡┩└┘━─┳╇┴]", " ", text).split())
