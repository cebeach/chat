"""The pre-send token check against a real llama-server with a 2048-token window.

`llama_args("--ctx-size 2048")` starts the profile's server with a small window, so a prompt
can be made to fit, to nearly fill it, or to overflow it with a few thousand words. Each test
runs once per selected profile (docs/testing.md); they are skipped unless a models directory
is configured. Prompts are sized with the server's own /tokenize, never estimated.
"""

from unittest import mock

import pytest
import requests

import chat
import tokens
from chat import State
from config import DEFAULTS
from conversation import Conversation, split_think
from tests import llama_server_config as cfg

WINDOW = 2048
# Short, deterministic replies: at this window a thinking model could otherwise fill it mid-reply.
OPTIONS = {"seed": 1, "temperature": 0, "n_predict": 16}
THINK_PROMPT = "What is 17 + 25? Answer with just the number."

pytestmark = [pytest.mark.integration, pytest.mark.llama_args(f"--ctx-size {WINDOW}")]


def user(text):
    return [{"role": "user", "content": text}]


def text_with_prompt_tokens(client, target, prefix="Count: "):
    """Text that makes the one-message prompt exactly `target` tokens, found by adjusting a word count."""
    words = max(1, target)
    for _ in range(40):
        needed = client.prompt_tokens(user(prefix + "word " * words))
        if needed == target:
            return prefix + "word " * words
        words = max(1, words + (target - needed))
    pytest.fail(f"could not size a prompt to {target} tokens (stopped at {needed})")


def make_state(client, **config):
    return State(
        model=client.server_model,
        config={**DEFAULTS, "conversations_dir": "/nonexistent", **config},
        context_length=client.server_n_ctx,
    )


def server_accepts(client, messages):
    """Whether llama-server itself takes the rendered prompt (it rejects one of n_ctx tokens or more)."""
    prompt = client._apply_template(messages, model=client.server_model)
    resp = requests.post(
        f"{client.base_url}/completion", json={"prompt": prompt, "n_predict": 1}, timeout=120
    )
    assert resp.status_code in (200, 400), resp.text
    return resp.status_code == 200


def test_the_window_is_the_one_the_marker_set(llama_client):
    assert llama_client.server_n_ctx == WINDOW


def test_the_counted_prompt_equals_what_the_server_evaluates(llama_client):
    """check_fit on the pending prompt is the reply's prompt_tokens; this is /tokenize == tokens_evaluated."""
    conv = Conversation(system_prompt="You are terse.")
    conv.add_user("Say hi.")
    conv.add_assistant("hi")
    pending = conv.get_messages() + user("Now say bye.")
    needed, n_ctx = llama_client.check_fit(pending, model=llama_client.server_model)
    stream = llama_client.chat(llama_client.server_model, pending, OPTIONS, refreshed=True)
    "".join(stream)
    assert n_ctx == WINDOW
    assert needed == stream.stats["prompt_tokens"]


def test_the_counted_prompt_with_project_notes_equals_what_the_server_evaluates(llama_client):
    """The notes block rides in the system message, so it is priced and evaluated with it."""
    conv = Conversation(system_prompt="You are terse.")
    conv.project_notes = "- Anna is left-handed\n- Ben is her brother\n"
    conv.add_user("Say hi.")
    conv.add_assistant("hi")
    pending = conv.get_messages() + user("Now say bye.")
    assert pending[0]["content"].endswith("- Ben is her brother\n")
    needed, _ = llama_client.check_fit(pending, model=llama_client.server_model)
    stream = llama_client.chat(llama_client.server_model, pending, OPTIONS, refreshed=True)
    "".join(stream)
    assert needed == stream.stats["prompt_tokens"]
    counts = tokens.breakdown(llama_client, conv, model=llama_client.server_model)
    assert counts["notes"] > 0
    assert (
        counts["system"] + counts["notes"] + counts["user"] + counts["assistant"] + counts["overhead"]
        == counts["total"]
    )


def test_info_totals_match_the_counted_history(llama_client):
    conv = Conversation(system_prompt="You are terse.")
    conv.add_user("Say hi.")
    conv.add_assistant("hi")
    counts = tokens.breakdown(llama_client, conv, model=llama_client.server_model)
    needed, _ = llama_client.check_fit(conv.get_messages(), model=llama_client.server_model)
    assert counts["total"] == needed
    assert counts["system"] + counts["user"] + counts["assistant"] + counts["overhead"] == needed
    assert counts["overhead"] > 0 and counts["n_ctx"] == WINDOW


def test_the_boundary_is_where_the_server_draws_it(llama_client):
    """fits() accepts exactly the prompts llama-server accepts, for n_ctx - 3 .. n_ctx tokens."""
    for target in range(WINDOW - 3, WINDOW + 1):
        messages = user(text_with_prompt_tokens(llama_client, target))
        needed, n_ctx = llama_client.check_fit(messages, model=llama_client.server_model)
        assert needed == target
        assert tokens.fits(needed, n_ctx) == server_accepts(llama_client, messages), target


def test_a_prompt_far_over_the_window_is_refused_and_nothing_is_stored(llama_client):
    state = make_state(llama_client)
    big = user("word " * 6000)
    with mock.patch.object(chat, "display_error") as error:
        ok, priced = chat._fits_window(llama_client, big, state)
    assert ok is False and priced[0] > WINDOW
    assert "context window is 2,048" in error.call_args.args[0]


def test_sixty_percent_is_allowed_quietly_and_eighty_five_warns(llama_client):
    state = make_state(llama_client)
    with mock.patch.object(chat, "display_context_warning") as warn:
        ok, priced = chat._fits_window(llama_client, user(text_with_prompt_tokens(llama_client, 1230)), state)
        assert ok and priced == (1230, WINDOW)
        warn.assert_not_called()
        ok, priced = chat._fits_window(llama_client, user(text_with_prompt_tokens(llama_client, 1740)), state)
        assert ok and priced == (1740, WINDOW)
        warn.assert_called_once_with(1740, WINDOW)


def test_with_the_check_off_the_oversize_send_reaches_the_servers_own_error(llama_client):
    state = make_state(llama_client, context_check=False)
    big = user("word " * 6000)
    with mock.patch.object(chat, "display_error") as error:
        ok, priced = chat._fits_window(llama_client, big, state)
    assert (ok, priced) == (True, None)
    error.assert_not_called()
    with pytest.raises(requests.HTTPError):
        llama_client.chat(llama_client.server_model, big, OPTIONS)


def test_the_reserve_moves_the_boundary_on_the_real_window(llama_client):
    state = make_state(llama_client, reserve_output_tokens=500)
    with mock.patch.object(chat, "display_error"):
        ok, _ = chat._fits_window(llama_client, user(text_with_prompt_tokens(llama_client, 1600)), state)
    assert ok is False
    ok, _ = chat._fits_window(llama_client, user(text_with_prompt_tokens(llama_client, 1500)), state)
    assert ok is True


def test_thinking_is_generated_but_not_part_of_the_next_prompt(llama_server, llama_client):
    profile = cfg.select_profile(cfg.load_config(), llama_server.profile)
    if not profile.think_tags:
        pytest.skip("this profile's model does not think")
    stream = llama_client.chat(
        llama_client.server_model, user(THINK_PROMPT), {"seed": 1, "temperature": 0, "n_predict": 1000}
    )
    reply = "".join(stream)
    thinking, answer = split_think(reply, stream.think_tags)
    assert thinking and answer.strip()
    conv = Conversation()
    conv.add_user(THINK_PROMPT)
    conv.add_assistant(answer, thinking=thinking)
    counts = tokens.breakdown(llama_client, conv, model=llama_client.server_model)
    assert counts["assistant"] == llama_client.count_tokens(answer, add_special=False)
    assert stream.stats["completion_tokens"] > counts["assistant"]


def test_a_reply_that_runs_out_of_window_is_flagged_truncated(llama_client):
    """A prompt that leaves room for only a few tokens: the server stops the reply and says so."""
    text = text_with_prompt_tokens(llama_client, WINDOW - 10, prefix="Repeat the word 'word' forever. ")
    stream = llama_client.chat(
        llama_client.server_model, user(text), {"seed": 1, "temperature": 0}, refreshed=False
    )
    "".join(stream)
    stats = stream.stats
    assert stream.truncated is True
    assert stats["prompt_tokens"] + stats["completion_tokens"] >= WINDOW - 2


def test_a_reply_stopped_by_a_token_limit_is_not_flagged_truncated(llama_client):
    """n_predict also ends a reply early (stop_type limit), but the window did not fill."""
    stream = llama_client.chat(
        llama_client.server_model, user("Count from 1 to 100, one number per line."), OPTIONS
    )
    "".join(stream)
    assert stream.stats["completion_tokens"] == OPTIONS["n_predict"]
    assert stream.truncated is False
