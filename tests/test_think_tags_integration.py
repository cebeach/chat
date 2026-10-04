"""Thinking-tag detection checked against real models.

Models mark their reasoning with different delimiters, and the app infers them from the
server (docs/thinking-tags.md). The offline tests check that inference against captured
template excerpts; these tests check it against the models themselves. They run once per
selected profile (see docs/testing.md), compare with the `think_tags` recorded for the
profile in tests/llama-server.toml, and need a running llama-server, so they are skipped
unless a models directory is configured.
"""

import pytest

from conversation import strip_think
from tests import llama_server_config as cfg

PROMPT = "What is 17 + 25? Answer with just the number."
ANSWER = "42"
# Deterministic on the machine where the tags were recorded, and generous enough for these
# models to finish thinking: they used 48 to 170 tokens for this prompt.
OPTIONS = {"seed": 1, "temperature": 0, "n_predict": 1500}


def recorded_tags(llama_server):
    """(tags recorded for the profile under test, the whole config). Fails if none are recorded.

    A tuple of two tags, or () for a model that does not think.
    """
    config = cfg.load_config()
    profile = cfg.select_profile(config, llama_server.profile)
    if profile.think_tags is None:
        pytest.fail(
            f"profile {profile.name!r} records no think_tags in tests/llama-server.toml. Run the model "
            "and read what it emits (docs/thinking-tags.md, 'How to check a new model'), then record "
            "it there: [start, end] for a thinking model, [] for one that does not think.",
            pytrace=False,
        )
    return profile.think_tags, config


@pytest.mark.integration
def test_detected_tags_match_the_model(llama_server, llama_client):
    expected, _ = recorded_tags(llama_server)
    detected = llama_client.think_tags
    if not expected:
        assert detected is None, f"recorded as a model that does not think, but the app detected {detected}"
        return
    assert detected is not None, f"recorded {list(expected)} but the app detected no tags"
    assert detected[:2] == expected
    assert detected[2] == "detected"  # from the server, not from a config override


@pytest.mark.integration
def test_a_real_reply_uses_the_tags(llama_server, llama_client):
    expected, config = recorded_tags(llama_server)
    messages = [{"role": "user", "content": PROMPT}]

    stream = llama_client.chat(llama_client.server_model, messages, OPTIONS)
    reply = "".join(stream)
    assert ANSWER in reply

    if not expected:
        assert stream.think_tags is None
        leaked = sorted(tag for tag in cfg.known_think_tags(config) if tag in reply)
        assert not leaked, f"a non-thinking model's reply contains thinking tags {leaked}: {reply!r}"
        assert reply.strip() == ANSWER
        return

    start, end = expected
    assert stream.think_tags[:2] == expected
    assert end in reply, f"no {end!r} in the reply (cut off at n_predict={OPTIONS['n_predict']}?): {reply!r}"
    # The reply opens with the start tag. For a forced-open template (Qwen's prompt already
    # ends with it) the app re-emits it; for the others the model writes it.
    assert reply.startswith(start), reply[:80]
    assert reply.index(start) < reply.index(end)
    # What the app saves with save_thinking off is the answer and nothing else.
    assert strip_think(reply, [expected]) == ANSWER
