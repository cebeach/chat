"""/duel against a real llama-server with two slots (profiles whose --parallel is 2 or more).

Asserts structure, never text: what a model writes is not under test. A real model can be cut off by
the shared KV pool or by its own reasoning, which is an abnormal end the design allows, so the legal
outcomes are read from the duel record's `ended` and `choice` fields instead of assuming completion.
The full happy path is pinned offline (test_duel_session.py), where the replies are scripted.
Run with --llama-model qwen35-2B-Q8_0-parallel2. See docs/testing.md.
"""

import json
import re

import pytest

import duel
from conversation import Conversation
from tests.pty_helpers import ROOT, PtyChild

# Thinking is switched off for these tests: a small reasoning model can spend the whole shared pool
# thinking about the draft prompt, which would make every run an abnormal end (a legal outcome, but
# one that exercises nothing past the first phase). Both tests share one server start.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llama_parallel(2),
    pytest.mark.llama_args("--reasoning off"),
]

BRIEF = "Write a story of about 60 words about a lighthouse keeper who finds a door."
# `processing task` is logged by the server for every request a slot takes (SLT_INF in
# tools/server/server-context.cpp): "slot launch_slot_: id  1 | task 5 | processing task, ...".
PROCESSING = re.compile(r"slot\s+\S+:\s+id\s+(\d+)\s+\|\s+task\s+\d+\s+\|\s+processing task")


def flag_value(argv, *names):
    """The value of the last occurrence of any of `names` in the server's argv."""
    value = None
    for i, arg in enumerate(argv[:-1]):
        if arg in names:
            value = argv[i + 1]
    return value


def test_the_server_reports_its_slots_and_the_pool_it_was_started_with(llama_server, llama_client):
    parallel = int(flag_value(llama_server.argv, "--parallel", "-np"))
    ctx_size = int(flag_value(llama_server.argv, "--ctx-size", "-c"))
    assert parallel >= 2
    assert llama_client.server_slots == parallel
    # --kv-unified (in the default args) makes each slot's reported n_ctx the whole pool, which is
    # why /duel assumes `shared`; this is the check of that assumption.
    assert "--kv-unified" in llama_server.argv
    assert llama_client.server_n_ctx == ctx_size


def test_a_real_duel_runs_on_both_slots_and_leaves_a_legal_record(llama_server, tmp_path):
    child = PtyChild(ROOT / "chat.py", tmp_path, args=["--url", llama_server.url])
    try:
        child.wait_prompts(1, timeout=60)
        child.send(f"/duel --auto {BRIEF}\r")
        # A failed DRAFT never shows the menu, so wait for either outcome.
        child.wait_for(
            lambda out: b"Use which?" in out or b"The drafts could not be completed" in out,
            "the final choice or a failed draft",
            timeout=900,
        )
        menu = b"Use which?" in child.out
        if menu:
            child.send("a\r")
        child.wait_prompts(2, timeout=60)
        child.send(b"\x04")
        assert child.exit_code(timeout=30) == 0
    finally:
        child.close()

    data = tmp_path / ".local" / "share" / "chat" / "conversations"
    (path,) = list((data / "duels").glob("duel-*.json"))
    record = json.loads(path.read_text())
    ended, choice = record["ended"], record["choice"]
    where = f"ended={ended!r}, choice={choice!r}"

    assert ended == "normal" or ended.startswith("abnormal:"), where
    assert all(
        i["status"] in duel.STANDING | duel.TERMINAL | {duel.RAISED, duel.REBUTTED}
        for i in record["ledger"]["issues"]
    ), where

    saved = [Conversation.load(data, name)[0] for name, _ in Conversation.list_saved(data)]
    messages = [m for conversation in saved for m in conversation.messages]
    if ended.startswith("abnormal:DRAFT:"):
        # (b) no drafts: nothing reached the conversation
        assert choice == "none" and messages == [], where
    else:
        # (a) the drafts or revisions were offered and the child chose side A
        assert choice == "a", where
        assert [m["role"] for m in messages] == ["user", "assistant"], where
        assert messages[1]["content"].strip(), where

    # Both slots took requests; side A is pinned to slot 0 and side B to slot 1.
    log = (llama_server.log_dir / "llama-server.log").read_text(errors="replace")
    assert {0, 1} <= {int(n) for n in PROCESSING.findall(log)}, where
