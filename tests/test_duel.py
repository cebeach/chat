"""duel.py: phases and participants, the issue ledger, prompts, parsers, options, arguments, records."""

import json
import random
from datetime import datetime

import pytest

import duel
from duel import (
    ACCEPTED,
    ADDRESSED,
    DECLINED,
    MAINTAINED,
    RAISED,
    REBUTTED,
    WITHDRAWN,
    Ledger,
    Phase,
)

AXES = duel.DEFAULT_AXES


def ledger_with(*specs):
    """A ledger with one issue per (by, status)."""
    ledger = Ledger()
    for by, status in specs:
        issue = ledger.raise_issue(by, "plot", "q", "o", "alt")
        issue.status = status
    return ledger


def run_sequence(ledger, answers):
    """Drive the phases like chat.py does; `answers` maps a Phase to a function(ledger, side)."""
    ran = []
    after = None
    while (step := duel.next_phase(after, ledger)) is not None:
        phase, who, skipped = step
        ran.append((phase, tuple(who)))
        answers.get(phase, lambda *_: None)(ledger, who)
        if phase == Phase.REVISE or (Phase.REVISE in skipped):
            ledger.close()
        after = phase
    return ran


class TestParticipants:
    def test_draft_and_cross_always_have_both_sides(self):
        ledger = Ledger()
        assert duel.participants(Phase.DRAFT, ledger) == ["a", "b"]
        assert duel.participants(Phase.CROSS, ledger) == ["a", "b"]

    def test_defend_is_each_side_with_a_raised_issue_against_its_draft(self):
        ledger = ledger_with(("a", RAISED))  # raised by A, so against B
        assert duel.participants(Phase.DEFEND, ledger) == ["b"]

    def test_reply_is_each_side_that_raised_a_rebutted_issue(self):
        ledger = ledger_with(("a", REBUTTED), ("b", ACCEPTED))
        assert duel.participants(Phase.REPLY, ledger) == ["a"]

    def test_revise_is_a_side_with_a_standing_issue_or_a_steer(self):
        ledger = ledger_with(("a", ACCEPTED))
        assert duel.participants(Phase.REVISE, ledger) == ["b"]
        ledger.add_steer("a", "shorter")
        assert duel.participants(Phase.REVISE, ledger) == ["a", "b"]

    def test_withdrawn_declined_and_addressed_issues_do_not_stand(self, subtests):
        for status in (WITHDRAWN, DECLINED, ADDRESSED, RAISED, REBUTTED):
            with subtests.test(status=status):
                assert duel.participants(Phase.REVISE, ledger_with(("a", status))) == []

    def test_judge_and_merge_are_one_neutral_participant(self):
        assert duel.participants(Phase.JUDGE, Ledger()) == [duel.JUDGE]
        assert duel.participants(Phase.MERGE, Ledger()) == [duel.JUDGE]


class TestPhasePlan:
    def test_next_phase_skips_phases_without_participants_and_names_them(self):
        phase, who, skipped = duel.next_phase(Phase.CROSS, Ledger())
        assert phase == Phase.JUDGE and who == [duel.JUDGE]
        assert skipped == [Phase.DEFEND, Phase.REPLY, Phase.REVISE]

    def test_after_the_last_phase_there_is_nothing(self):
        assert duel.next_phase(Phase.JUDGE, Ledger()) is None

    def test_minimum_is_three_phases_and_five_turns(self):
        ran = run_sequence(Ledger(), {})
        assert [p for p, _ in ran] == [Phase.DRAFT, Phase.CROSS, Phase.JUDGE]
        assert sum(len(w) for _, w in ran) == 5

    def test_maximum_is_six_phases_and_eleven_turns(self):
        def cross(ledger, who):
            for side in who:
                duel.apply_cross(ledger, side, 'ISSUE | plot | "q" | bad | fix', AXES)

        def defend(ledger, who):
            for side in who:
                duel.apply_defend(
                    ledger,
                    side,
                    "\n".join(f"REBUT {i.n}: no" for i in ledger.with_status(RAISED, against=side)),
                )

        def reply(ledger, who):
            for side in who:
                duel.apply_reply(
                    ledger, side, "\n".join(f"MAINTAIN {i.n}" for i in ledger.with_status(REBUTTED, by=side))
                )

        def revise(ledger, who):
            for side in who:
                duel.apply_revise(ledger, side, "new text")

        ran = run_sequence(
            Ledger(), {Phase.CROSS: cross, Phase.DEFEND: defend, Phase.REPLY: reply, Phase.REVISE: revise}
        )
        assert [p for p, _ in ran] == list(duel.SEQUENCE)
        assert sum(len(w) for _, w in ran) == 11

    def test_merge_adds_one_phase_and_one_turn(self):
        assert duel.participants(Phase.MERGE, Ledger()) == [duel.JUDGE]
        assert Phase.MERGE not in duel.SEQUENCE

    def test_phase_plan_lists_what_would_run_from_the_current_state(self):
        plan = duel.phase_plan(Ledger())
        assert [p for p, _ in plan] == [Phase.DRAFT, Phase.CROSS, Phase.JUDGE]

    def test_a_steer_alone_brings_revise_back_after_a_skip_note(self):
        ledger = Ledger()
        ledger.add_steer("both", "x")
        plan = duel.phase_plan(ledger, after=Phase.CROSS)
        assert [p for p, _ in plan] == [Phase.REVISE, Phase.JUDGE]

    def test_skip_note_names_what_is_skipped_and_why(self, subtests):
        cases = [
            (
                [Phase.DEFEND, Phase.REPLY, Phase.REVISE],
                "No issues raised; skipping defend, reply and revise.",
            ),
            ([Phase.REPLY, Phase.REVISE], "No rebuttals to answer; skipping reply and revise."),
            ([Phase.REPLY], "No rebuttals to answer; skipping reply."),
            ([Phase.REVISE], "Nothing to revise; skipping revise."),
        ]
        for skipped, note in cases:
            with subtests.test(note=note):
                assert duel.skip_note(skipped) == note
        assert duel.skip_note([]) == ""

    def test_usable_text_ignores_decline_lines_only_in_revise(self):
        assert duel.usable_text(Phase.REVISE, "DECLINE 1: no") == ""
        assert duel.usable_text(Phase.DRAFT, "DECLINE 1: no") == "DECLINE 1: no"
        assert duel.usable_text(Phase.CROSS, "  ") == ""

    def test_new_conversation_carries_the_prompt_and_notes(self):
        conversation = duel.new_conversation("sys", "notes")
        assert conversation.system_content() == "sys\n\n# Project notes\nnotes"


ALL = [RAISED, ACCEPTED, REBUTTED, MAINTAINED, WITHDRAWN, ADDRESSED, DECLINED]
LEGAL = {
    (RAISED, ACCEPTED),
    (RAISED, REBUTTED),
    (REBUTTED, MAINTAINED),
    (REBUTTED, WITHDRAWN),
    (ACCEPTED, ADDRESSED),
    (ACCEPTED, DECLINED),
    (MAINTAINED, ADDRESSED),
    (MAINTAINED, DECLINED),
}
MOVES = {
    ACCEPTED: lambda ledger: ledger.accept(1),
    REBUTTED: lambda ledger: ledger.rebut(1, "r"),
    MAINTAINED: lambda ledger: ledger.maintain(1),
    WITHDRAWN: lambda ledger: ledger.withdraw(1),
    ADDRESSED: lambda ledger: ledger.address(1),
    DECLINED: lambda ledger: ledger.decline(1, "r"),
}


class TestLedger:
    def test_every_pair_is_legal_or_raises(self, subtests):
        for start in ALL:
            for to, move in MOVES.items():
                with subtests.test(start=start, to=to):
                    ledger = ledger_with(("a", start))
                    if (start, to) in LEGAL:
                        move(ledger)
                        assert ledger.get(1).status == to
                    else:
                        with pytest.raises(duel.IllegalTransition):
                            move(ledger)
                        assert ledger.get(1).status == start

    def test_standing_and_terminal_sets(self):
        assert duel.STANDING == {ACCEPTED, MAINTAINED}
        assert duel.TERMINAL == {ADDRESSED, DECLINED, WITHDRAWN}

    def test_decline_and_rebut_keep_the_reason(self):
        ledger = ledger_with(("a", REBUTTED))
        ledger.get(1).status = RAISED
        ledger.rebut(1, "it is intended")
        assert ledger.get(1).rebuttal == "it is intended"
        ledger.maintain(1)
        ledger.decline(1, "kept")
        assert ledger.get(1).declined_reason == "kept"

    def test_issues_are_numbered_from_one_and_point_at_the_opponent(self):
        ledger = Ledger()
        first = ledger.raise_issue("a", "plot")
        second = ledger.raise_issue("b", "voice")
        assert (first.n, first.against, first.by) == (1, "b", "a")
        assert (second.n, second.against, second.by) == (2, "a", "b")

    def test_overrides_follow_the_table(self, subtests):
        for start in ALL:
            with subtests.test(accept_from=start):
                ledger = ledger_with(("a", start))
                if start in {RAISED, REBUTTED, MAINTAINED, WITHDRAWN}:
                    ledger.override_accept(1)
                    assert ledger.get(1).status == ACCEPTED and ledger.get(1).by_user
                else:
                    with pytest.raises(duel.Refused):
                        ledger.override_accept(1)
            with subtests.test(drop_from=start):
                ledger = ledger_with(("a", start))
                if start in {RAISED, ACCEPTED, REBUTTED, MAINTAINED}:
                    ledger.override_drop(1)
                    assert ledger.get(1).status == WITHDRAWN and ledger.get(1).by_user
                else:
                    with pytest.raises(duel.Refused):
                        ledger.override_drop(1)

    def test_unknown_issue_number_is_refused(self):
        with pytest.raises(duel.Refused):
            Ledger().override_accept(3)

    def test_overrides_and_steers_are_refused_once_revise_is_closed(self):
        ledger = ledger_with(("a", RAISED))
        ledger.close()
        with pytest.raises(duel.Refused):
            ledger.override_accept(1)
        with pytest.raises(duel.Refused):
            ledger.override_drop(1)
        with pytest.raises(duel.Refused):
            ledger.add_steer("a", "x")
        assert ledger.get(1).status == RAISED and ledger.steers == {"a": [], "b": []}

    def test_an_overridden_issue_leaves_defend_and_reply_input(self):
        ledger = ledger_with(("a", RAISED), ("b", REBUTTED))
        ledger.override_accept(1)
        ledger.override_drop(2)
        assert duel.participants(Phase.DEFEND, ledger) == []
        assert duel.participants(Phase.REPLY, ledger) == []

    def test_accept_at_the_pause_brings_revise_back_and_drop_cancels_it(self):
        ledger = ledger_with(("a", RAISED))
        assert duel.participants(Phase.REVISE, ledger) == []
        ledger.override_accept(1)
        assert duel.participants(Phase.REVISE, ledger) == ["b"]
        ledger.override_drop(1)
        assert duel.participants(Phase.REVISE, ledger) == []

    def test_steers_route_to_one_side_or_both(self):
        ledger = Ledger()
        ledger.add_steer("a", "one")
        ledger.add_steer("both", "two")
        assert ledger.steers == {"a": ["one", "two"], "b": ["two"]}

    def test_to_dict_is_json(self):
        ledger = ledger_with(("a", RAISED))
        assert json.loads(json.dumps(ledger.to_dict()))["issues"][0]["status"] == RAISED


class TestParsers:
    def test_cross_well_formed(self):
        text = 'ISSUE | plot | "He leaves." | too abrupt | add a scene\nISSUE | voice | "Hi" | flat | warmer | really'
        first, second = duel.parse_cross(text, AXES)
        assert first == {
            "axis": "plot",
            "quote": "He leaves.",
            "objection": "too abrupt",
            "alternative": "add a scene",
            "axis_known": True,
            "unparsed": False,
        }
        assert second["alternative"] == "warmer | really"

    def test_cross_no_issues_and_empty_mean_none(self, subtests):
        for text in ("NO ISSUES", "no issues.", "  \n", duel.NO_ANSWER):
            with subtests.test(text=text):
                assert duel.parse_cross(text, AXES) == []

    def test_cross_unparseable_becomes_one_general_unparsed_issue(self):
        (issue,) = duel.parse_cross("The ending feels rushed to me.", AXES)
        assert issue["axis"] == "general" and issue["unparsed"]
        assert "rushed" in issue["objection"]

    def test_cross_unknown_axis_is_kept_but_flagged(self):
        (issue,) = duel.parse_cross('ISSUE | grammar | "x" | bad | fix', AXES)
        assert issue["axis"] == "grammar" and issue["axis_known"] is False

    def test_cross_tolerates_bullets_and_case(self):
        (issue,) = duel.parse_cross('- issue | PLOT | "x" | bad | fix', AXES)
        assert issue["axis"] == "plot" and issue["axis_known"]

    def test_defend_answers_and_missing_answers_accept(self):
        got = duel.parse_defend("ACCEPT 1\nREBUT 2: it is intended\nREBUT 9: stray", [1, 2, 3])
        assert got == {
            1: ("accept", "", False),
            2: ("rebut", "it is intended", False),
            3: ("accept", "", True),
        }

    def test_reply_answers_and_missing_answers_maintain(self):
        got = duel.parse_reply("WITHDRAW 1\nsome chatter", [1, 2])
        assert got == {1: ("withdraw", False), 2: ("maintain", True)}

    def test_revise_strips_trailing_decline_lines(self):
        story, declines = duel.parse_revise(
            "Once upon a time.\nThe end.\n\nDECLINE 2: needed\nDECLINE 5: also"
        )
        assert story == "Once upon a time.\nThe end."
        assert declines == {2: "needed", 5: "also"}

    def test_revise_keeps_a_decline_line_inside_the_text(self):
        story, declines = duel.parse_revise("DECLINE 1: in a quote\nmore text")
        assert story == "DECLINE 1: in a quote\nmore text" and declines == {}

    def test_judge_maps_text_numbers_back_to_sides(self):
        reasons, side = duel.parse_judge("Text 1 is tighter.\nRECOMMEND: Text 2", ("b", "a"))
        assert side == "a" and reasons == "Text 1 is tighter."

    def test_judge_none_missing_or_garbled_means_no_recommendation(self, subtests):
        for text in ("Reasons.\nRECOMMEND: NONE", "Reasons only.", "Reasons.\nRECOMMEND: both"):
            with subtests.test(text=text):
                assert duel.parse_judge(text, ("a", "b"))[1] is None
        assert duel.parse_judge("Only reasons.", ("a", "b"))[0] == "Only reasons."

    def test_merge_needs_the_rewrite_line_and_a_text(self):
        assert duel.parse_merge("- warmer dialogue\n---REWRITE---\nNew story.") == (
            "- warmer dialogue",
            "New story.",
        )
        assert duel.parse_merge("- strengths only") is None
        assert duel.parse_merge("- s\n---REWRITE---\n  \n") is None

    def test_parsing_only_sees_what_it_is_given(self):
        # chat.py passes the answer part only, so a thinking block never reaches a parser.
        assert duel.parse_defend("", [1]) == {1: ("accept", "", True)}


class TestApply:
    def test_cross_creates_issues_against_the_other_side(self):
        ledger = Ledger()
        duel.apply_cross(ledger, "a", 'ISSUE | plot | "x" | bad | fix', AXES)
        assert [(i.by, i.against, i.status) for i in ledger.issues] == [("a", "b", RAISED)]

    def test_defend_then_reply_then_revise(self):
        ledger = Ledger()
        duel.apply_cross(
            ledger, "a", 'ISSUE | plot | "x" | bad | fix\nISSUE | voice | "y" | flat | warm', AXES
        )
        duel.apply_defend(ledger, "b", "ACCEPT 1\nREBUT 2: deliberate")
        assert [i.status for i in ledger.issues] == [ACCEPTED, REBUTTED]
        duel.apply_reply(ledger, "a", "MAINTAIN 2")
        assert ledger.get(2).status == MAINTAINED
        story = duel.apply_revise(ledger, "b", "Better.\nDECLINE 2: deliberate")
        assert story == "Better."
        assert [i.status for i in ledger.issues] == [ADDRESSED, DECLINED]

    def test_unanswered_issues_stand(self):
        ledger = Ledger()
        duel.apply_cross(ledger, "a", 'ISSUE | plot | "x" | bad | fix', AXES)
        duel.apply_defend(ledger, "b", "")
        assert ledger.get(1).status == ACCEPTED and ledger.get(1).unparsed
        duel.apply_revise(ledger, "b", "A text with no decline line.")
        assert ledger.get(1).status == ADDRESSED

    def test_unanswered_rebuttal_is_maintained(self):
        ledger = Ledger()
        duel.apply_cross(ledger, "a", 'ISSUE | plot | "x" | bad | fix', AXES)
        duel.apply_defend(ledger, "b", "REBUT 1: no")
        duel.apply_reply(ledger, "a", duel.NO_ANSWER)
        assert ledger.get(1).status == MAINTAINED and ledger.get(1).unparsed


class TestPrompts:
    BRIEF = "A lighthouse keeper finds a door."

    def test_every_builder_restates_the_brief(self):
        issue = ledger_with(("a", RAISED)).issues[0]
        prompts = [
            duel.draft_prompt(self.BRIEF),
            duel.cross_prompt(self.BRIEF, "text", AXES),
            duel.defend_prompt(self.BRIEF, [issue]),
            duel.reply_prompt(self.BRIEF, [issue]),
            duel.revise_prompt(self.BRIEF, "text", [issue], []),
            duel.judge_prompt(self.BRIEF, ("t1", "t2")),
            duel.merge_prompt(self.BRIEF, "base", "donor"),
        ]
        assert all(self.BRIEF in p for p in prompts)

    def test_each_builder_states_its_format(self):
        issue = ledger_with(("a", RAISED)).issues[0]
        assert "ISSUE | axis" in duel.cross_prompt(
            self.BRIEF, "t", AXES
        ) and "NO ISSUES" in duel.cross_prompt(self.BRIEF, "t", AXES)
        assert "ACCEPT n" in duel.defend_prompt(self.BRIEF, [issue]) and "REBUT n" in duel.defend_prompt(
            self.BRIEF, [issue]
        )
        assert "MAINTAIN n" in duel.reply_prompt(self.BRIEF, [issue]) and "WITHDRAW n" in duel.reply_prompt(
            self.BRIEF, [issue]
        )
        assert "DECLINE n" in duel.revise_prompt(self.BRIEF, "t", [issue], [])
        assert "RECOMMEND: Text 1" in duel.judge_prompt(self.BRIEF, ("a", "b"))
        assert duel.REWRITE_MARK in duel.merge_prompt(self.BRIEF, "a", "b")

    def test_cross_lists_the_allowed_axes(self):
        assert "plot, voice" in duel.cross_prompt(self.BRIEF, "t", ["plot", "voice"])

    def test_draft_default_length_and_revise_ten_percent(self):
        assert duel.DEFAULT_DRAFT_LENGTH in duel.draft_prompt(self.BRIEF)
        prompt = duel.revise_prompt(self.BRIEF, "one two three", [], ["tighter"])
        assert "10%" in prompt and "(3 words)" in prompt and "- tighter" in prompt

    def test_a_defend_prompt_shows_only_the_given_issues_with_quote_and_alternative(self):
        issue = ledger_with(("a", RAISED)).issues[0]
        prompt = duel.defend_prompt(self.BRIEF, [issue])
        assert "Issue 1 (plot): o" in prompt and 'Quote: "q"' in prompt and "Suggested: alt" in prompt

    def test_reply_prompt_carries_the_rebuttal(self):
        ledger = Ledger()
        ledger.raise_issue("a", "plot", "q", "o")
        ledger.rebut(1, "because")
        assert "because" in duel.reply_prompt(self.BRIEF, ledger.issues)

    def test_side_system_adds_the_stance(self):
        assert duel.side_system("Be brief.", "Stance.") == "Be brief.\n\nStance."
        assert duel.side_system("", "Stance.") == "Stance."

    def test_defaults_are_opposed_non_empty_stances(self):
        assert duel.DEFAULT_STANCE_A and duel.DEFAULT_STANCE_B
        assert duel.DEFAULT_STANCE_A != duel.DEFAULT_STANCE_B

    def test_judge_order_is_a_permutation_from_the_injected_rng(self):
        orders = {duel.judge_order(random.Random(i)) for i in range(20)}
        assert orders == {("a", "b"), ("b", "a")}
        assert duel.order_note(("b", "a")) == "Text 1 = Side B, Text 2 = Side A"


class TestSideOptions:
    def test_the_set_override_reaches_both_sides(self):
        got = duel.side_options({"temperature": 0.3}, {}, {})
        assert got["a"] == got["b"] == {"temperature": 0.3}

    def test_duel_setting_beats_set_override(self):
        got = duel.side_options({"temperature": 0.3}, {"temperature_b": 1.1}, {})
        assert got["a"]["temperature"] == 0.3 and got["b"]["temperature"] == 1.1

    def test_seed_precedence_duel_then_override_then_fixed_server_seed(self):
        assert duel.side_options({"seed": 5}, {"seed_a": 9}, {"seed": 77})["a"]["seed"] == 9
        assert duel.side_options({"seed": 5}, {}, {"seed": 77})["a"]["seed"] == 5
        got = duel.side_options({}, {}, {"seed": 77})
        assert got["a"].get("seed") is None and got["b"]["seed"] == 78  # equal fixed seeds: B gets +1

    def test_the_random_server_seed_counts_as_unset(self):
        got = duel.side_options({}, {}, {"seed": 4294967295})
        assert "seed" not in got["a"] and "seed" not in got["b"]

    def test_equal_override_seeds_make_b_plus_one(self):
        got = duel.side_options({"seed": 42}, {}, {})
        assert (got["a"]["seed"], got["b"]["seed"]) == (42, 43)

    def test_different_seeds_are_left_alone(self):
        got = duel.side_options({}, {"seed_a": 1, "seed_b": 2}, {})
        assert (got["a"]["seed"], got["b"]["seed"]) == (1, 2)

    def test_a_random_override_seed_is_sent_as_given(self):
        got = duel.side_options({"seed": -1}, {}, {})
        assert got["a"]["seed"] == -1 and got["b"]["seed"] == -1

    def test_options_do_not_alias_the_input(self):
        options = {"temperature": 0.3}
        duel.side_options(options, {"temperature_a": 1}, {})["a"]["x"] = 1
        assert options == {"temperature": 0.3}


class TestParseArgs:
    def test_nothing_is_usage(self):
        assert duel.parse_args("  ").kind == "usage"

    def test_a_brief_with_and_without_the_flag(self):
        assert duel.parse_args("a lighthouse") == duel.Command("run", brief="a lighthouse")
        assert duel.parse_args("--auto a lighthouse") == duel.Command("run", brief="a lighthouse", auto=True)

    def test_subcommands_only_when_exact(self):
        assert duel.parse_args("ledger").kind == "ledger"
        cmd = duel.parse_args("set axes plot, voice")
        assert (cmd.kind, cmd.key, cmd.value) == ("set", "axes", "plot, voice")
        assert duel.parse_args("settle a score").kind == "run"
        assert duel.parse_args("ledgers of a bank").kind == "run"

    def test_double_dash_ends_flag_and_subcommand_parsing(self):
        cmd = duel.parse_args("-- set the scene in a lighthouse")
        assert cmd.kind == "run" and cmd.brief == "set the scene in a lighthouse"
        cmd = duel.parse_args("--auto -- --odd brief")
        assert cmd.auto and cmd.brief == "--odd brief"

    def test_a_flag_hides_a_following_subcommand_word(self):
        assert duel.parse_args("--auto set the scene").brief == "set the scene"

    def test_errors(self, subtests):
        for text in ("set", "--bogus brief", "--auto", "--"):
            with subtests.test(text=text):
                assert duel.parse_args(text).kind == "error"


class TestRecord:
    def test_written_under_the_directory_with_the_start_time(self, tmp_path):
        started = datetime(2026, 10, 10, 9, 8, 7)
        path = duel.write_record(tmp_path / "duels", {"ended": "normal"}, started)
        assert path.name == "duel-20261010-090807.json"
        assert json.loads(path.read_text()) == {"ended": "normal"}

    def test_a_taken_name_gets_a_counter(self, tmp_path):
        started = datetime(2026, 10, 10, 9, 8, 7)
        first = duel.write_record(tmp_path, {}, started)
        second = duel.write_record(tmp_path, {}, started)
        third = duel.write_record(tmp_path, {}, started)
        assert [p.name for p in (first, second, third)] == [
            "duel-20261010-090807.json",
            "duel-20261010-090807-2.json",
            "duel-20261010-090807-3.json",
        ]

    def test_rewrite_replaces_atomically_and_leaves_no_temporary_file(self, tmp_path):
        path = duel.write_record(tmp_path, {"ended": None}, datetime(2026, 1, 1))
        duel.rewrite_record(path, {"ended": "stopped"})
        assert json.loads(path.read_text()) == {"ended": "stopped"}
        assert [p.name for p in tmp_path.iterdir()] == [path.name]

    def test_a_failed_write_keeps_the_old_record(self, tmp_path):
        path = duel.write_record(tmp_path, {"ended": "normal"}, datetime(2026, 1, 1))
        with pytest.raises(TypeError):
            duel.rewrite_record(path, {"bad": object()})
        assert json.loads(path.read_text()) == {"ended": "normal"}
        assert [p.name for p in tmp_path.iterdir()] == [path.name]

    def test_a_record_is_not_listed_as_a_conversation(self, tmp_path):
        from conversation import Conversation

        duel.write_record(tmp_path / "duels", {}, datetime(2026, 1, 1))
        assert Conversation.list_saved(tmp_path) == []
