"""/duel over the real main(): a mock client with scripted slots, scripted input() answers.

The replies are scripted per slot (SlotScript), so these tests pin the flow and the bookkeeping,
never the quality of anything a model wrote.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

import chat
import duel
from conversation import Conversation
from tests.helpers import Reply, SlotScript
from tests.test_turn_echo import run_session


class NoShuffle:
    """Text 1 is Side A, Text 2 is Side B."""

    def shuffle(self, items):
        pass


@pytest.fixture(autouse=True)
def fixed_judge_order():
    with mock.patch.object(chat, "_duel_rng", NoShuffle()):
        yield


def full_replies(judge="A is tighter.\nRECOMMEND: Text 1"):
    """A duel that runs all six phases.

    A raises issue 1 against B (B accepts it); B raises issue 2 against A (A rebuts it, B maintains
    it). Both sides revise, A declining issue 2. Slot 0 also answers the judge.
    """
    slot0 = [
        Reply("Draft A.", context_tokens=200),
        Reply('ISSUE | plot | "x" | too slow | speed up', context_tokens=500),  # issue 1, against B
        Reply("REBUT 2: deliberate", context_tokens=600),  # A defends against issue 2
        Reply("Revised A.\nDECLINE 2: deliberate", context_tokens=800),
        Reply(judge, context_tokens=900),
    ]
    slot1 = [
        Reply("Draft B.", context_tokens=210),
        Reply('ISSUE | voice | "y" | flat | warmer', context_tokens=510),  # issue 2, against A
        Reply("ACCEPT 1", context_tokens=610),  # B defends against issue 1
        Reply("MAINTAIN 2", context_tokens=710),  # B answers A's rebuttal
        Reply("Revised B.", context_tokens=810),
    ]
    return {0: slot0, 1: slot1}


def duel_session(tmp_path, replies=None, inputs=("/duel --auto write a story", None), answers=("a",), **kw):
    script = SlotScript(replies if replies is not None else full_replies(), barrier=kw.pop("barrier", None))
    setup = kw.pop("setup", None)

    def both(client):
        script.setup(client)
        if setup:
            setup(client)

    session = run_session(tmp_path, script.behaviour, inputs=inputs, answers=answers, setup=both, **kw)
    session.script = script
    return session


def records(tmp_path):
    return sorted((Path(tmp_path) / "duels").glob("duel-*.json"))


def record(tmp_path):
    (path,) = records(tmp_path)
    return json.loads(path.read_text())


class TestHappyPath:
    def test_all_six_phases_run_and_only_the_chosen_text_enters_the_conversation(self, tmp_path):
        session = duel_session(tmp_path)
        messages = session.conversation.messages
        assert [(m["role"], m["content"]) for m in messages] == [
            ("user", "write a story"),
            ("assistant", "Revised A."),
        ]
        assert session.errors == []

    def test_ten_turns_on_the_right_slots(self, tmp_path):
        session = duel_session(tmp_path)
        assert {slot: len(calls) for slot, calls in session.script.calls.items()} == {0: 5, 1: 5}
        assert session.script.exhausted()

    def test_the_record_says_normal_and_a(self, tmp_path):
        duel_session(tmp_path)
        rec = record(tmp_path)
        assert (rec["ended"], rec["choice"]) == ("normal", "a")
        assert [p["phase"] for p in rec["phases"]] == ["DRAFT", "CROSS", "DEFEND", "REPLY", "REVISE", "JUDGE"]
        assert rec["drafts"] == {"a": "Draft A.", "b": "Draft B."}
        assert rec["texts"] == {"a": "Revised A.", "b": "Revised B."}
        assert rec["judge"]["order"] == ["a", "b"] and rec["judge"]["recommendation"] == "a"
        assert [i["status"] for i in rec["ledger"]["issues"]] == ["addressed", "declined"]

    def test_the_record_is_outside_the_conversation_listing(self, tmp_path):
        duel_session(tmp_path)
        assert all(not name.startswith("duel-") for name, _ in Conversation.list_saved(tmp_path))
        assert len(records(tmp_path)) == 1

    def test_phases_are_announced_in_order_with_running_numbers(self, tmp_path, capsys):
        duel_session(tmp_path)
        out = capsys.readouterr().out
        names = ["draft", "cross-examine", "defend", "reply", "revise", "judge"]
        positions = [out.index(f"Phase {i} · {name}") for i, name in enumerate(names, 1)]
        assert positions == sorted(positions)

    def test_the_conversation_is_autosaved_after_the_final_text(self, tmp_path):
        session = duel_session(tmp_path)
        autosaves = [e for e in session.log if e[0] == "auto_save"]
        assert autosaves
        names = [n for n, _ in Conversation.list_saved(tmp_path)]
        assert names

    def test_the_reply_is_attributed_to_the_model(self, tmp_path):
        session = duel_session(tmp_path)
        assert session.conversation.messages[1]["model"] == "/m/x.gguf"

    def test_the_sides_do_not_see_the_main_conversations_earlier_messages(self, tmp_path):
        replies = full_replies()
        replies[None] = [Reply("Hi there.")]
        session = duel_session(tmp_path, replies, inputs=("hello", "/duel --auto go", None))
        first = session.script.calls[0][0]["messages"]
        assert not any("hello" in m["content"] or "Hi there." in m["content"] for m in first)
        assert [m["content"] for m in session.conversation.messages][:2] == ["hello", "Hi there."]
        assert [m["content"] for m in session.conversation.messages][2:] == ["go", "Revised A."]

    def test_each_side_sees_its_own_stance_and_the_brief(self, tmp_path):
        session = duel_session(tmp_path)
        first_a = session.script.calls[0][0]["messages"]
        first_b = session.script.calls[1][0]["messages"]
        assert duel.DEFAULT_STANCE_A in first_a[0]["content"]
        assert duel.DEFAULT_STANCE_B in first_b[0]["content"]
        assert "write a story" in first_a[1]["content"]

    def test_cross_shows_each_side_the_opponents_draft_only(self, tmp_path):
        session = duel_session(tmp_path)
        cross_a = session.script.calls[0][1]["messages"][-1]["content"]
        cross_b = session.script.calls[1][1]["messages"][-1]["content"]
        assert "Draft B." in cross_a and "Draft A." not in cross_a
        assert "Draft A." in cross_b and "Draft B." not in cross_b

    def test_revise_carries_only_the_issues_against_that_side(self, tmp_path):
        session = duel_session(tmp_path)
        revise_a = session.script.calls[0][3]["messages"][-1]["content"]
        revise_b = session.script.calls[1][4]["messages"][-1]["content"]
        assert "flat" in revise_a and "too slow" not in revise_a
        assert "too slow" in revise_b and "flat" not in revise_b

    def test_roles_alternate_in_each_side(self, tmp_path):
        session = duel_session(tmp_path)
        for slot, index in ((0, 4), (1, 4)):
            roles = [
                m["role"] for m in session.script.calls[slot][index]["messages"] if m["role"] != "system"
            ]
            assert roles == ["user", "assistant"] * (len(roles) // 2) + ["user"]


class TestChoice:
    def test_the_final_choice_prompt_is_never_read_with_the_echo_off(self, tmp_path):
        prompts = []
        duel_session(tmp_path, on_input=lambda prompt, active: prompts.append((prompt, active)))
        assert len(prompts) == 1 and prompts[0][0].startswith("Use which?") and prompts[0][1] is False

    def test_b(self, tmp_path):
        session = duel_session(tmp_path, answers=("b",))
        assert session.conversation.messages[1]["content"] == "Revised B."
        assert record(tmp_path)["choice"] == "b"

    def test_q_discards_everything(self, tmp_path):
        session = duel_session(tmp_path, answers=("q",))
        assert session.conversation.messages == []
        assert record(tmp_path)["choice"] == "none"

    def test_end_of_input_at_the_final_choice_is_q(self, tmp_path):
        session = duel_session(tmp_path, answers=(EOFError(),))
        assert session.conversation.messages == []
        assert record(tmp_path)["choice"] == "none"

    def test_ctrl_c_at_the_final_choice_is_q(self, tmp_path):
        session = duel_session(tmp_path, answers=(KeyboardInterrupt(),))
        assert session.conversation.messages == [] and record(tmp_path)["choice"] == "none"

    def test_unknown_input_is_refused_and_asked_again(self, tmp_path):
        session = duel_session(tmp_path, answers=("what", "a"))
        assert session.errors and session.conversation.messages[1]["content"] == "Revised A."


class TestPauses:
    def test_without_auto_there_is_a_pause_between_every_two_phases(self, tmp_path):
        prompts = []
        duel_session(
            tmp_path,
            inputs=("/duel write a story", None),
            answers=("", "", "", "", "", "a"),
            on_input=lambda prompt, active: prompts.append((prompt, active)),
        )
        assert [p for p, _ in prompts].count("duel> ") == 5  # not after the last phase
        assert all(active is False for _, active in prompts)  # never read with the echo off

    def test_stop_goes_to_the_final_choice_with_the_current_texts_and_no_judge(self, tmp_path):
        session = duel_session(tmp_path, answers=("", "stop", "a"), inputs=("/duel write a story", None))
        assert session.conversation.messages[1]["content"] == "Draft A."
        rec = record(tmp_path)
        assert (rec["ended"], rec["choice"]) == ("stopped", "a")
        assert rec["judge"] is None

    def test_end_of_input_at_a_pause_is_stop(self, tmp_path):
        session = duel_session(tmp_path, answers=(EOFError(), "b"), inputs=("/duel write a story", None))
        assert session.conversation.messages[1]["content"] == "Draft B."
        assert record(tmp_path)["ended"] == "stopped"

    def test_ctrl_c_at_a_pause_is_stop(self, tmp_path):
        duel_session(tmp_path, answers=(KeyboardInterrupt(), "q"), inputs=("/duel write a story", None))
        assert record(tmp_path)["ended"] == "stopped"

    def test_auto_at_a_pause_stops_the_pausing(self, tmp_path):
        prompts = []
        duel_session(
            tmp_path,
            inputs=("/duel write a story", None),
            answers=("auto", "a"),
            on_input=lambda prompt, active: prompts.append(prompt),
        )
        assert prompts.count("duel> ") == 1
        assert record(tmp_path)["ended"] == "normal"

    def test_set_auto_on_makes_the_next_duel_automatic(self, tmp_path):
        prompts = []
        duel_session(
            tmp_path,
            inputs=("/duel set auto on", "/duel write a story", None),
            answers=("a",),
            on_input=lambda prompt, active: prompts.append(prompt),
        )
        assert "duel> " not in prompts

    def test_a_steer_makes_a_side_a_revise_participant_and_reaches_only_that_side(self, tmp_path):
        replies = {
            0: [Reply("Draft A."), Reply("NO ISSUES"), Reply("Revised A."), Reply("Judge.\nRECOMMEND: NONE")],
            1: [Reply("Draft B."), Reply("NO ISSUES")],
        }
        session = duel_session(
            tmp_path,
            replies,
            inputs=("/duel write a story", None),
            answers=("", "a: make it shorter", "", "", "a"),
        )
        assert [len(c) for c in (session.script.calls[0], session.script.calls[1])] == [4, 2]
        revise_a = session.script.calls[0][2]["messages"][-1]["content"]
        assert "make it shorter" in revise_a
        assert session.conversation.messages[1]["content"] == "Revised A."
        assert record(tmp_path)["ledger"]["steers"] == {"a": ["make it shorter"], "b": []}

    def test_both_reaches_both_sides(self, tmp_path):
        replies = {
            0: [Reply("Draft A."), Reply("NO ISSUES"), Reply("Revised A."), Reply("J\nRECOMMEND: NONE")],
            1: [Reply("Draft B."), Reply("NO ISSUES"), Reply("Revised B.")],
        }
        duel_session(
            tmp_path,
            replies,
            inputs=("/duel write a story", None),
            answers=("", "both: tighter", "", "", "a"),
        )
        assert record(tmp_path)["ledger"]["steers"] == {"a": ["tighter"], "b": ["tighter"]}

    def test_a_steer_after_revise_is_refused(self, tmp_path):
        # the last pause (before JUDGE) comes after REVISE has run
        session = duel_session(
            tmp_path,
            inputs=("/duel write a story", None),
            answers=("", "", "", "", "a: too late", "", "a"),
        )
        assert any("Too late" in e for e in session.errors)
        assert record(tmp_path)["ledger"]["steers"] == {"a": [], "b": []}

    def test_dropping_the_only_issue_cancels_defend_reply_and_revise(self, tmp_path):
        replies = {
            0: [
                Reply("Draft A."),
                Reply('ISSUE | plot | "x" | slow | faster'),
                Reply("Judge\nRECOMMEND: NONE"),
            ],
            1: [Reply("Draft B."), Reply("NO ISSUES")],
        }
        session = duel_session(
            tmp_path, replies, inputs=("/duel write a story", None), answers=("", "drop 1", "", "a")
        )
        assert session.script.exhausted()  # no defend, reply or revise turns were needed
        issue = record(tmp_path)["ledger"]["issues"][0]
        assert issue["status"] == "withdrawn" and issue["by_user"]

    def test_accepting_an_issue_at_the_pause_brings_revise_back(self, tmp_path):
        replies = {
            0: [
                Reply("Draft A."),
                Reply('ISSUE | plot | "x" | slow | faster'),
                Reply("Judge\nRECOMMEND: NONE"),
            ],
            1: [Reply("Draft B."), Reply("NO ISSUES"), Reply("Revised B.")],
        }
        session = duel_session(
            tmp_path, replies, inputs=("/duel write a story", None), answers=("", "accept 1", "", "", "a")
        )
        assert session.script.exhausted()  # B revised without defending
        issue = record(tmp_path)["ledger"]["issues"][0]
        assert issue["status"] == "addressed" and issue["by_user"]
        assert "slow" in session.script.calls[1][2]["messages"][-1]["content"]

    def test_skipped_phases_are_noted_without_a_number(self, tmp_path, capsys):
        replies = {
            0: [Reply("Draft A."), Reply("NO ISSUES"), Reply("J\nRECOMMEND: NONE")],
            1: [Reply("Draft B."), Reply("NO ISSUES")],
        }
        session = duel_session(tmp_path, replies)
        out = capsys.readouterr().out
        assert "No issues raised; skipping defend, reply and revise." in session.infos
        assert "Phase 3 · judge" in out
        assert "Phase 3 · defend" not in out
        assert [p["phase"] for p in record(tmp_path)["phases"]] == ["DRAFT", "CROSS", "JUDGE"]

    def test_unknown_pause_command_is_an_error(self, tmp_path):
        session = duel_session(tmp_path, answers=("what?", "stop", "q"), inputs=("/duel write a story", None))
        assert session.errors

    def test_show_prints_the_current_text(self, tmp_path, capsys):
        duel_session(tmp_path, answers=("show a", "stop", "q"), inputs=("/duel write a story", None))
        assert "Draft A." in capsys.readouterr().out


class TestRefusals:
    def test_one_slot(self, tmp_path):
        session = duel_session(tmp_path, replies={}, setup=lambda c: setattr(c, "server_slots", 1))
        assert session.errors == ["Duel needs two slots. Start llama-server with --parallel 2."]
        assert session.script.calls == {} and session.conversation.messages == []
        assert records(tmp_path) == []

    def test_unreadable_slot_count(self, tmp_path):
        session = duel_session(tmp_path, replies={}, setup=lambda c: setattr(c, "server_slots", None))
        assert session.errors == ["Could not read the slot count from llama-server."]
        assert records(tmp_path) == []

    def test_a_bad_include_aborts_before_anything_starts(self, tmp_path):
        session = duel_session(tmp_path, replies={}, inputs=("/duel read @@<nothing.txt>", None))
        assert session.errors and records(tmp_path) == []
        assert session.script.calls == {} and session.conversation.messages == []

    def test_an_included_file_is_part_of_the_brief_and_is_recorded(self, tmp_path):
        notes = tmp_path / "setting.txt"
        notes.write_text("A tall white lighthouse.")
        session = duel_session(tmp_path, inputs=(f"/duel --auto use @@<{notes}>", None))
        assert "A tall white lighthouse." in session.script.calls[0][0]["messages"][-1]["content"]
        user = session.conversation.messages[0]
        assert "A tall white lighthouse." in user["content"] and user["includes"]
        assert record(tmp_path)["includes"]

    def test_a_flag_without_a_brief(self, tmp_path):
        session = duel_session(tmp_path, replies={}, inputs=("/duel --auto", None))
        assert session.errors == ["A duel needs a brief."]

    def test_context_refusal_ends_the_duel_with_one_message(self, tmp_path):
        session = duel_session(tmp_path, check_fit=(5000, 4096), answers=())
        assert len([e for e in session.errors if "Not sent" in e]) == 1
        rec = record(tmp_path)
        assert rec["ended"] == "abnormal:DRAFT:context" and rec["choice"] == "none"
        assert session.conversation.messages == []

    def test_the_pool_rule_refuses_when_each_side_fits_alone(self, tmp_path):
        session = duel_session(tmp_path, check_fit=(3000, 4096), answers=())
        assert record(tmp_path)["ended"] == "abnormal:DRAFT:context"
        assert any("together" in e for e in session.errors)

    def test_the_pool_rule_is_skipped_under_split(self, tmp_path):
        session = duel_session(
            tmp_path,
            check_fit=(3000, 4096),
            inputs=("/duel set context split", "/duel --auto write a story", None),
            replies={
                0: [Reply("Draft A."), Reply("NO ISSUES"), Reply("J\nRECOMMEND: NONE")],
                1: [Reply("Draft B."), Reply("NO ISSUES")],
            },
        )
        assert record(tmp_path)["ended"] == "normal"

    def test_the_pool_rule_is_skipped_when_pricing_is_off(self, tmp_path):
        session = duel_session(
            tmp_path,
            config={"context_check": False},
            replies={
                0: [Reply("Draft A."), Reply("NO ISSUES"), Reply("J\nRECOMMEND: NONE")],
                1: [Reply("Draft B."), Reply("NO ISSUES")],
            },
        )
        assert record(tmp_path)["ended"] == "normal"
        session.client.check_fit.assert_not_called()

    def test_judge_counts_the_retained_tokens_of_the_idle_slot(self, tmp_path):
        # each side holds ~2400 after cross; the judge (1200 + 1) plus slot 1's 2400 exceeds 3500
        replies = {
            0: [
                Reply("Draft A.", context_tokens=1000),
                Reply("NO ISSUES", context_tokens=1100),
                Reply("J\nRECOMMEND: NONE"),
            ],
            1: [Reply("Draft B.", context_tokens=1000), Reply("NO ISSUES", context_tokens=2400)],
        }
        fits = [(200, 3500)] * 4 + [(1200, 3500)]
        session = duel_session(tmp_path, replies, check_fit=fits)
        rec = record(tmp_path)
        assert rec["ended"] == "abnormal:JUDGE:context"
        assert session.script.calls[0][-1] is not None and len(session.script.calls[0]) == 2


class TestAbnormalEnds:
    def test_failed_draft_adds_nothing_but_writes_the_record(self, tmp_path):
        replies = {0: [Reply("Draft A.")], 1: [Reply("Draft B.", truncated=True)]}
        session = duel_session(tmp_path, replies, answers=())
        assert session.conversation.messages == []
        rec = record(tmp_path)
        assert (rec["ended"], rec["choice"]) == ("abnormal:DRAFT:cut_off", "none")

    def test_a_cut_off_later_phase_goes_to_the_final_choice_with_the_drafts(self, tmp_path):
        replies = {
            0: [Reply("Draft A."), Reply("NO ISSUES", truncated=True)],
            1: [Reply("Draft B."), Reply("NO ISSUES")],
        }
        session = duel_session(tmp_path, replies)
        assert session.conversation.messages[1]["content"] == "Draft A."
        rec = record(tmp_path)
        assert (rec["ended"], rec["choice"]) == ("abnormal:CROSS:cut_off", "a")
        assert rec["ledger"]["issues"] == []  # the discarded phase changed nothing

    def test_ctrl_c_mid_phase_goes_to_the_final_choice(self, tmp_path):
        replies = {
            0: [Reply("Draft A."), Reply("one two three four five", interrupt_at=2)],
            1: [Reply("Draft B."), Reply("NO ISSUES")],
        }
        session = duel_session(tmp_path, replies)
        assert record(tmp_path)["ended"] == "abnormal:CROSS:interrupted"
        assert session.conversation.messages[1]["content"] == "Draft A."

    def test_a_server_error(self, tmp_path):
        from requests.exceptions import ConnectionError

        replies = {
            0: [Reply("Draft A."), Reply("x", fail_on_chat=ConnectionError("down"))],
            1: [Reply("Draft B."), Reply("NO ISSUES")],
        }
        duel_session(tmp_path, replies)
        assert record(tmp_path)["ended"] == "abnormal:CROSS:server_error"

    def test_an_empty_revision_is_abnormal(self, tmp_path):
        replies = full_replies()
        replies[0][3] = Reply("DECLINE 2: no")
        duel_session(tmp_path, replies)
        assert record(tmp_path)["ended"] == "abnormal:REVISE:empty_answer"
        assert record(tmp_path)["texts"] == {"a": "Draft A.", "b": "Draft B."}

    def test_a_failed_judge_leaves_the_menu_without_a_recommendation_and_j_retries(self, tmp_path):
        replies = full_replies()
        replies[0][4] = Reply("", context_tokens=1)  # empty judge
        replies[0].append(Reply("Retry.\nRECOMMEND: Text 2"))
        session = duel_session(tmp_path, replies, answers=("j", "b"))
        rec = record(tmp_path)
        assert rec["ended"] == "abnormal:JUDGE:empty_answer"
        assert rec["judge"]["recommendation"] == "b"
        assert session.conversation.messages[1]["content"] == "Revised B."

    def test_a_judge_without_a_recommend_line_is_not_abnormal(self, tmp_path):
        replies = full_replies(judge="Only reasons, no line.")
        duel_session(tmp_path, replies)
        rec = record(tmp_path)
        assert rec["ended"] == "normal" and rec["judge"]["recommendation"] is None


class TestMerge:
    def merge_replies(self, merge):
        replies = full_replies()
        replies[0].append(Reply(merge))
        return replies

    def test_m_a_merges_and_only_c_adds_it(self, tmp_path):
        replies = self.merge_replies("- warmer\n---REWRITE---\nMerged story.")
        session = duel_session(tmp_path, replies, answers=("m a", "c"))
        assert session.conversation.messages[1]["content"] == "Merged story."
        rec = record(tmp_path)
        assert rec["choice"] == "merged:a"
        assert rec["merges"] == [{"base": "a", "strengths": "- warmer", "text": "Merged story."}]

    def test_bare_m_uses_the_recommended_side(self, tmp_path):
        replies = self.merge_replies("s\n---REWRITE---\nM.")
        duel_session(tmp_path, replies, answers=("m", "c"))
        assert record(tmp_path)["merges"][0]["base"] == "a"

    def test_m_alone_is_refused_without_a_recommendation(self, tmp_path):
        replies = full_replies(judge="No line.")
        session = duel_session(tmp_path, replies, answers=("m", "q"))
        assert (
            any("no recommendation" in e.lower() or "no recommendation" in e for e in session.errors)
            or session.errors
        )
        assert record(tmp_path)["merges"] == []

    def test_m_does_not_add_the_merged_text_by_itself(self, tmp_path):
        replies = self.merge_replies("s\n---REWRITE---\nM.")
        session = duel_session(tmp_path, replies, answers=("m a", "q"))
        assert session.conversation.messages == []

    def test_a_second_merge_replaces_the_candidate(self, tmp_path):
        replies = self.merge_replies("s\n---REWRITE---\nFirst.")
        replies[0].append(Reply("s2\n---REWRITE---\nSecond."))
        session = duel_session(tmp_path, replies, answers=("m a", "m a", "c"))
        assert session.conversation.messages[1]["content"] == "Second."
        assert len(record(tmp_path)["merges"]) == 2

    def test_a_merge_without_the_rewrite_line_shows_the_menu_again_without_c(self, tmp_path):
        replies = self.merge_replies("no marker here")
        session = duel_session(tmp_path, replies, answers=("m a", "c", "a"))
        assert session.errors  # the format error, and c refused
        assert session.conversation.messages[1]["content"] == "Revised A."
        assert record(tmp_path)["merges"] == []

    def test_the_merge_runs_on_slot_zero_alone(self, tmp_path):
        replies = self.merge_replies("s\n---REWRITE---\nM.")
        session = duel_session(tmp_path, replies, answers=("m b", "q"))
        assert len(session.script.calls[0]) == 6 and len(session.script.calls[1]) == 5
        merge_prompt = session.script.calls[0][5]["messages"][-1]["content"]
        assert "Revised B." in merge_prompt and merge_prompt.index("Revised B.") < merge_prompt.index(
            "Revised A."
        )


class TestSettingsAndOptions:
    def test_options_reach_the_calls_with_precedence(self, tmp_path):
        session = duel_session(
            tmp_path,
            inputs=(
                "/set temperature 0.3",
                "/duel set temperature_b 1.2",
                "/duel set seed_a 5",
                "/duel --auto go",
                None,
            ),
        )
        a, b = session.script.calls[0][0]["options"], session.script.calls[1][0]["options"]
        assert a == {"temperature": 0.3, "seed": 5}
        assert b == {"temperature": 1.2}

    def test_set_temperature_reaches_both_sides(self, tmp_path):
        session = duel_session(tmp_path, inputs=("/set temperature 0.3", "/duel --auto go", None))
        assert (
            session.script.calls[0][0]["options"]
            == session.script.calls[1][0]["options"]
            == {"temperature": 0.3}
        )

    def test_slot_ids_are_pinned(self, tmp_path):
        session = duel_session(tmp_path)
        assert all(c["id_slot"] == 0 for c in session.script.calls[0])
        assert all(c["id_slot"] == 1 for c in session.script.calls[1])

    def test_duel_set_validation(self, tmp_path):
        session = duel_session(
            tmp_path,
            replies={},
            inputs=(
                "/duel set temperature_a -3",
                "/duel set seed_a x",
                "/duel set context nope",
                "/duel set auto maybe",
                "/duel set bogus 1",
                "/duel set axes plot,,voice",
                None,
            ),
        )
        assert len(session.errors) == 6

    def test_duel_set_values_and_default(self, tmp_path):
        session = duel_session(
            tmp_path,
            replies={},
            inputs=(
                "/duel set axes plot, voice",
                "/duel set stance_a Be brief.",
                "/duel set axes default",
                None,
            ),
        )
        assert session.errors == []
        assert session.infos[-1].startswith("axes reset")

    def test_set_axes_reach_the_cross_prompt(self, tmp_path):
        session = duel_session(tmp_path, inputs=("/duel set axes plot,voice", "/duel --auto go", None))
        cross = session.script.calls[0][1]["messages"][-1]["content"]
        assert "plot, voice" in cross

    def test_usage_lists_slots_and_the_assumption(self, tmp_path):
        session = duel_session(tmp_path, replies={}, inputs=("/duel", None))
        text = "\n".join(session.infos)
        assert "Server slots: 2" in text and "shared" in text and "Usage: /duel" in text

    def test_ledger_with_no_duel_yet(self, tmp_path):
        session = duel_session(tmp_path, replies={}, inputs=("/duel ledger", None))
        assert session.infos[-1] == "No duel yet."

    def test_ledger_shows_the_last_duel(self, tmp_path, capsys):
        duel_session(tmp_path, inputs=("/duel --auto go", "/duel ledger", None))
        out = capsys.readouterr().out
        assert "too slow" in out and "flat" in out

    def test_double_dash_makes_set_a_brief(self, tmp_path):
        session = duel_session(tmp_path, inputs=("/duel --auto -- set the scene in a lighthouse", None))
        assert session.conversation.messages[0]["content"] == "set the scene in a lighthouse"


class TestMultilineAndNotes:
    def test_a_multiline_brief(self, tmp_path):
        with mock.patch.object(chat, "get_multiline_input", return_value="line one\nline two"):
            session = duel_session(tmp_path, inputs=('/duel --auto """', None))
        assert session.conversation.messages[0]["content"] == "line one\nline two"

    def test_end_of_input_in_a_multiline_brief_cancels(self, tmp_path):
        with mock.patch.object(chat, "get_multiline_input", return_value=None):
            session = duel_session(tmp_path, replies={}, inputs=('/duel --auto """', None))
        assert records(tmp_path) == [] and session.conversation.messages == []

    def test_the_project_notes_reach_both_sides_and_an_outside_edit_is_picked_up(self, tmp_path):
        project = tmp_path / "projects" / "novel"
        project.mkdir(parents=True)
        (project / "project.md").write_text("The keeper is named Ada.")

        def edit_notes(client):
            (project / "project.md").write_text("The keeper is named Bea.")

        # the edit lands after startup (setup runs before main), so main loads the edited file;
        # an edit between the prompt and /duel is the same read path (_sync_notes)
        session = duel_session(
            tmp_path, inputs=("/duel --auto go", None), argv=("--project", "novel"), setup=edit_notes
        )
        for slot in (0, 1):
            assert "Bea" in session.script.calls[slot][0]["messages"][0]["content"]


class TestRetry:
    def test_retry_regenerates_a_single_normal_draft_replacing_the_duel_text(self, tmp_path):
        replies = full_replies()
        replies[None] = [Reply("A plain draft.")]
        session = duel_session(tmp_path, replies, inputs=("/duel --auto go", "/retry", None))
        assert session.conversation.messages[1]["content"] == "A plain draft."
        assert (
            "id_slot" not in session.script.calls[None][0] or session.script.calls[None][0]["id_slot"] is None
        )


class TestEchoShape:
    def test_every_phase_is_one_suppression_pair_and_the_prompts_are_never_inside(self, tmp_path):
        session = duel_session(tmp_path, inputs=("/duel --auto go", None))
        names = [n for n, _ in session.log]
        assert names.count("enter") == names.count("exit")
        depth = 0
        for name, active in session.log:
            if name == "enter":
                depth += 1
                assert depth == 1  # never nested
            elif name == "exit":
                depth -= 1
            elif name == "get_user_input":
                assert not active
        pairs = names.count("enter")
        # 6 phases, plus the _sync_notes block, the main prompt checks and the final autosave
        assert pairs >= 6
