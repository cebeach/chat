"""chat._run_phase: one phase of a duel over scripted slots (no server, no main loop).

Assertions use the per-slot sequences of SlotScript, never the interleaving of the two sides.
"""

import threading
from unittest import mock

import pytest
from requests.exceptions import ConnectionError, HTTPError

import chat
import duel
from chat import State, _Job, _run_phase
from config import DEFAULTS
from conversation import Conversation
from duel import Phase
from tests.helpers import Reply, SlotScript
from tests.test_turn_echo import Recorder


class Env:
    def __init__(self, script, check_fit=(100, 4096), config=None, context="shared", show_stats=False):
        self.script = script
        self.client = mock.Mock(server_model="/m/x.gguf", server_n_ctx=4096)
        self.client.chat.side_effect = lambda **kw: script.behaviour()
        script.setup(self.client)
        fits = check_fit
        self.client.check_fit.side_effect = lambda messages, model=None: (
            fits.pop(0) if isinstance(fits, list) else fits
        )
        self.state = State(model="/m/x.gguf", config={**DEFAULTS, **(config or {})}, context_length=4096)
        self.state.duel["context"] = context
        self.state.show_stats = show_stats
        self.log = []
        self.held = {0: 0, 1: 0}
        self.a, self.b = Conversation("sys A"), Conversation("sys B")

    def jobs(self, *keys, text="go"):
        conversations = {"a": self.a, "b": self.b}
        slots = {"a": 0, "b": 1, duel.JUDGE: 0}
        judge = Conversation("judge")
        conversations[duel.JUDGE] = judge
        self.judge = judge
        return [_Job(k, conversations[k], text, slots[k], {"temperature": 0.5}) for k in keys]

    def run(self, phase, *keys, text="go"):
        with mock.patch.object(chat, "echo_suppressed", Recorder(self.log)):
            return _run_phase(phase, self.jobs(*keys, text=text), self.client, self.state, self.held)


def roles(conversation):
    return [m["role"] for m in conversation.messages]


class TestNormalPhase:
    def test_each_participant_gets_one_user_and_one_assistant_message(self):
        env = Env(SlotScript({0: [Reply("Story A.")], 1: [Reply("Story B.")]}))
        result = env.run(Phase.DRAFT, "a", "b")
        assert result.ok and result.cause is None
        assert {k: t.answer for k, t in result.turns.items()} == {"a": "Story A.", "b": "Story B."}
        assert roles(env.a) == roles(env.b) == ["user", "assistant"]
        assert env.a.messages[1]["content"] == "Story A."

    def test_each_side_is_pinned_to_its_slot_with_its_options(self):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B")]}))
        env.run(Phase.DRAFT, "a", "b")
        (call_a,), (call_b,) = env.script.calls[0], env.script.calls[1]
        assert (call_a["id_slot"], call_b["id_slot"]) == (0, 1)
        assert call_a["options"] == call_b["options"] == {"temperature": 0.5}
        assert call_a["refreshed"] is True and call_a["model"] == "/m/x.gguf"

    def test_the_prompt_is_the_conversation_plus_the_new_user_message(self):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B")]}))
        env.run(Phase.DRAFT, "a", "b", text="the brief")
        (call,) = env.script.calls[0]
        assert call["messages"] == [
            {"role": "system", "content": "sys A"},
            {"role": "user", "content": "the brief"},
        ]

    def test_a_non_participant_is_untouched(self):
        env = Env(SlotScript({1: [Reply("B")]}))
        result = env.run(Phase.DEFEND, "b")
        assert result.ok and env.a.messages == []
        assert roles(env.b) == ["user", "assistant"]
        assert env.script.calls.get(0) is None

    def test_the_two_streams_overlap(self):
        # Each stream waits for the other to start; a sequential implementation would time out.
        env = Env(SlotScript({0: [Reply("A one")], 1: [Reply("B one")]}, barrier=2))
        assert env.run(Phase.DRAFT, "a", "b").ok

    def test_one_echo_suppression_pair_per_phase(self):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B")]}))
        env.run(Phase.DRAFT, "a", "b")
        assert [n for n, _ in env.log] == ["enter", "exit"]

    def test_thinking_is_split_off_and_stored_apart(self):
        tags = ("<think>", "</think>", "config")
        env = Env(
            SlotScript({0: [Reply("<think>hmm ACCEPT 1</think>Answer.", think_tags=tags)], 1: [Reply("B")]})
        )
        result = env.run(Phase.DRAFT, "a", "b")
        assert result.turns["a"].answer == "Answer." and result.turns["a"].thinking == "hmm ACCEPT 1"
        message = env.a.messages[1]
        assert message["content"] == "Answer." and message["thinking"] == "hmm ACCEPT 1"
        assert message["model"] == "/m/x.gguf"

    def test_held_tokens_follow_the_last_completed_turn(self):
        env = Env(SlotScript({0: [Reply("A", context_tokens=300)], 1: [Reply("B", context_tokens=450)]}))
        env.run(Phase.DRAFT, "a", "b")
        assert env.held == {0: 300, 1: 450}

    def test_stats_are_shown_when_on(self, capsys):
        env = Env(SlotScript({0: [Reply("A word")], 1: [Reply("B word")]}), show_stats=True)
        env.run(Phase.DRAFT, "a", "b")
        assert capsys.readouterr().out.count("generated (thinking + answer)") == 2

    def test_both_texts_are_displayed_under_their_labels(self, capsys):
        env = Env(SlotScript({0: [Reply("alpha text")], 1: [Reply("beta text")]}))
        env.run(Phase.DRAFT, "a", "b")
        out = capsys.readouterr().out
        assert "Side A:" in out and "alpha text" in out and "Side B:" in out and "beta text" in out
        assert out.index("alpha text") < out.index("Side B:")

    def test_the_judge_runs_alone_on_slot_zero(self):
        env = Env(SlotScript({0: [Reply("Reasons.\nRECOMMEND: Text 1")]}))
        result = env.run(Phase.JUDGE, duel.JUDGE)
        assert result.ok and list(result.turns) == [duel.JUDGE]
        assert env.script.calls[0][0]["id_slot"] == 0


class TestEmptyAnswers:
    def test_empty_draft_is_abnormal(self):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("   ")]}))
        result = env.run(Phase.DRAFT, "a", "b")
        assert not result.ok and result.cause == "empty_answer"

    def test_only_thinking_is_empty_too(self):
        tags = ("<think>", "</think>", "config")
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("<think>only this</think>", think_tags=tags)]}))
        assert env.run(Phase.REVISE, "a", "b").cause == "empty_answer"

    @pytest.mark.parametrize("phase", [Phase.CROSS, Phase.DEFEND, Phase.REPLY])
    def test_empty_in_the_other_phases_is_stored_as_no_answer(self, phase):
        env = Env(SlotScript({0: [Reply("")], 1: [Reply("fine")]}))
        result = env.run(phase, "a", "b")
        assert result.ok
        assert env.a.messages[1]["content"] == duel.NO_ANSWER
        assert result.turns["a"].answer == duel.NO_ANSWER and result.turns["b"].answer == "fine"

    @pytest.mark.parametrize("phase", [Phase.JUDGE, Phase.MERGE])
    def test_empty_judge_or_merge_is_abnormal(self, phase):
        env = Env(SlotScript({0: [Reply("")]}))
        assert env.run(phase, duel.JUDGE).cause == "empty_answer"


class TestAbnormalEnd:
    def assert_discarded(self, env, result, cause):
        assert not result.ok and result.cause == cause
        assert env.a.messages == [] and env.b.messages == []  # whole phase rolled back
        assert result.turns == {}

    def test_a_cut_off_reply_discards_the_whole_phase_even_the_complete_turn(self):
        env = Env(SlotScript({0: [Reply("A fine")], 1: [Reply("B cut", truncated=True)]}))
        self.assert_discarded(env, env.run(Phase.DRAFT, "a", "b"), "cut_off")

    def test_a_token_limit_is_a_cut_off(self):
        env = Env(SlotScript({0: [Reply("A", limited=True)], 1: [Reply("B")]}))
        self.assert_discarded(env, env.run(Phase.DRAFT, "a", "b"), "cut_off")

    def test_a_stream_without_a_final_chunk_is_a_cut_off(self):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B", finished=False)]}))
        self.assert_discarded(env, env.run(Phase.DRAFT, "a", "b"), "cut_off")

    @pytest.mark.parametrize(
        "error, text",
        [
            (ConnectionError("down"), "Lost connection to llama-server"),
            (HTTPError("503"), "llama-server error: 503"),
            (chat.ThinkTagsError("tags unknown"), "tags unknown"),
        ],
    )
    def test_server_errors_from_the_request(self, error, text, capsys):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B", fail_on_chat=error)]}))
        self.assert_discarded(env, env.run(Phase.DRAFT, "a", "b"), "server_error")
        assert text in capsys.readouterr().out

    def test_a_server_error_in_the_middle_of_a_stream(self):
        env = Env(
            SlotScript({0: [Reply("A")], 1: [Reply("one two three", fail_in_stream=ConnectionError("x"))]})
        )
        self.assert_discarded(env, env.run(Phase.DRAFT, "a", "b"), "server_error")

    def test_an_unexpected_exception_in_a_worker_is_reported_not_lost(self, capsys):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B", fail_on_chat=RuntimeError("boom"))]}))
        self.assert_discarded(env, env.run(Phase.DRAFT, "a", "b"), "server_error")
        assert "boom" in capsys.readouterr().out

    def test_ctrl_c_mid_stream_discards_the_phase_and_closes_the_streams(self):
        env = Env(
            SlotScript({0: [Reply("one two three four", interrupt_at=2)], 1: [Reply("B", finished=True)]})
        )
        result = env.run(Phase.DRAFT, "a", "b")
        self.assert_discarded(env, result, "interrupted")
        assert all(s.closed for streams in env.script.streams.values() for s in streams)

    def test_the_partial_text_is_not_stored(self):
        env = Env(SlotScript({0: [Reply("one two three four", interrupt_at=2)], 1: [Reply("B")]}))
        env.run(Phase.DRAFT, "a", "b")
        assert all("[interrupted]" not in m["content"] for m in env.a.messages)

    def test_an_abnormal_end_is_still_one_suppression_pair(self):
        env = Env(SlotScript({0: [Reply("A", truncated=True)], 1: [Reply("B")]}))
        env.run(Phase.DRAFT, "a", "b")
        assert [n for n, _ in env.log] == ["enter", "exit"]

    def test_no_thread_is_left_running_after_a_normal_phase(self):
        before = threading.active_count()
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B")]}))
        env.run(Phase.DRAFT, "a", "b")
        assert threading.active_count() == before


class TestInterruptWhileReplaying:
    def test_an_interrupt_after_every_turn_finished_does_not_discard_the_phase(self):
        # Both texts are complete when the Ctrl-C arrives during B's replay. The phase is kept
        # (only the replay is cut short) and B's text is whole.
        script = SlotScript({0: [Reply("alpha beta")], 1: [Reply("gamma delta epsilon")]})
        env = Env(script)
        original = chat.display_assistant_stream
        calls = []

        def display(gen, **kw):
            calls.append(kw["label"])
            if kw["label"] == "Side B:":
                # every turn is finished by now; the Ctrl-C only cuts the replay short
                for thread in threading.enumerate():
                    if thread.daemon:
                        thread.join(timeout=5)
                kw["interrupted"].set()
                return ""
            return original(gen, **kw)

        with mock.patch.object(chat, "display_assistant_stream", display):
            result = env.run(Phase.DRAFT, "a", "b")
        assert calls == ["Side A:", "Side B:"]
        assert result.ok and result.turns["b"].answer == "gamma delta epsilon"
        assert roles(env.b) == ["user", "assistant"]


class TestContextBudget:
    def test_a_participant_that_does_not_fit_alone_is_refused_with_one_message(self, capsys):
        env = Env(SlotScript({}), check_fit=[(100, 4096), (5000, 4096)])
        result = env.run(Phase.DRAFT, "a", "b")
        assert not result.ok and result.cause == "context"
        assert env.a.messages == [] and env.b.messages == []
        assert env.script.calls == {}
        out = capsys.readouterr().out
        assert out.count("Not sent") == 1 and "Side B" in out

    def test_each_fits_alone_but_not_together_under_shared(self, capsys):
        env = Env(SlotScript({}), check_fit=(3000, 4096), context="shared")
        result = env.run(Phase.DRAFT, "a", "b")
        assert result.cause == "context" and env.script.calls == {}
        out = capsys.readouterr().out
        assert "together" in out and "/duel set context split" in out

    def test_split_only_asks_that_each_fits_alone(self):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B")]}), check_fit=(3000, 4096), context="split")
        assert env.run(Phase.DRAFT, "a", "b").ok

    def test_the_reserve_is_per_participant(self):
        # 2 * (1900 + 100 + 1) = 4002 <= 4096, but with a reserve of 200: 2 * 2101 = 4202 > 4096
        script = SlotScript({0: [Reply("A")], 1: [Reply("B")]})
        env = Env(script, check_fit=(1900, 4096), config={"reserve_output_tokens": 100})
        assert env.run(Phase.DRAFT, "a", "b").ok
        env = Env(SlotScript({}), check_fit=(1900, 4096), config={"reserve_output_tokens": 200})
        assert env.run(Phase.DRAFT, "a", "b").cause == "context"

    def test_the_judge_counts_the_tokens_the_idle_slot_still_holds(self):
        env = Env(SlotScript({}), check_fit=(1000, 4096))
        env.held = {0: 500, 1: 3200}  # slot 1 keeps 3200; judge needs 1001
        assert env.run(Phase.JUDGE, duel.JUDGE).cause == "context"
        env.held = {0: 3200, 1: 500}  # slot 0 is the judge's own and is replaced
        env.script.replies = {0: [Reply("fine\nRECOMMEND: NONE")]}
        assert env.run(Phase.JUDGE, duel.JUDGE).ok

    def test_a_phase_without_one_side_counts_that_sides_retained_tokens(self):
        env = Env(SlotScript({}), check_fit=(1000, 4096))
        env.held = {0: 3500, 1: 0}
        assert env.run(Phase.DEFEND, "b").cause == "context"

    def test_a_discarded_phase_keeps_the_larger_held_count(self):
        env = Env(SlotScript({0: [Reply("A", truncated=True)], 1: [Reply("B")]}), check_fit=(800, 4096))
        env.held = {0: 100, 1: 2000}
        env.run(Phase.DRAFT, "a", "b")
        assert env.held == {0: 800, 1: 2000}  # the priced prompt (reserve 0) beats the 100 held before
        env = Env(
            SlotScript({0: [Reply("A", truncated=True)], 1: [Reply("B")]}),
            check_fit=(800, 4096),
            config={"reserve_output_tokens": 50},
        )
        env.held = {0: 100, 1: 2000}
        env.run(Phase.DRAFT, "a", "b")
        assert env.held[0] == 850

    def test_pricing_unavailable_refuses_nothing_and_refreshes_itself(self):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B")]}), config={"context_check": False})
        assert env.run(Phase.DRAFT, "a", "b").ok
        env.client.check_fit.assert_not_called()
        env.client.refresh.assert_called()

    def test_with_pricing_the_workers_rely_on_the_check_to_refresh(self):
        env = Env(SlotScript({0: [Reply("A")], 1: [Reply("B")]}))
        env.run(Phase.DRAFT, "a", "b")
        env.client.refresh.assert_not_called()

    def test_ctrl_c_while_pricing_is_cancelled(self, capsys):
        env = Env(SlotScript({}), check_fit=(100, 4096))
        env.client.check_fit.side_effect = KeyboardInterrupt
        result = env.run(Phase.DRAFT, "a", "b")
        assert result.cause == "interrupted" and "Cancelled." in capsys.readouterr().out
        assert env.a.messages == []
