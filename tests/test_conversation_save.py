import json
import re
from pathlib import Path
from unittest import mock

import pytest

import chat
import ui
from chat import State, _auto_save, handle_command
from config import DEFAULTS
from conversation import MAX_THINK_PAIRS, Conversation
from conversation import strip_think as _strip_think

THINK = "<think>\nreasoning here\n</think>\nThe answer."
PAIRS = [("<think>", "</think>")]


def strip_think(text):
    return _strip_think(text, PAIRS)


def make_state(tmp, **config):
    return State(
        model="m",
        config={"conversations_dir": str(tmp), "auto_save": True, **config},
        context_length=None,
        auto_save_name="auto_test",
        think_tags=("<think>", "</think>", "detected"),
        think_pairs_seen=list(PAIRS),
    )


def saved_messages(tmp, name):
    return json.loads((Path(tmp) / f"{name}.json").read_text())["messages"]


class TestStripThink:
    def test_balanced_blocks_removed(self):
        assert strip_think(THINK) == "The answer."
        assert strip_think("<think>a</think>X<think>b</think>Y") == "XY"

    def test_unbalanced_left_byte_identical(self, subtests):
        for text in [
            "reasoning\n</think>\nanswer",  # lone closing tag
            "<think>\nreasoning cut off [interrupted]",  # lone opening tag
            "</think>a<think>",  # reversed
            "no tags at all",
        ]:
            with subtests.test(text=text):
                assert strip_think(text) == text

    def test_pair_plus_stray_tag_only_removes_pair(self):
        assert strip_think("<think>a</think> x </think>") == "x </think>"

    def test_nested_only_the_innermost_balanced_pair_is_removed(self):
        # The inner pair is balanced; the outer tags are strays and stay as evidence.
        assert strip_think("<think><think>a</think>b") == "<think>b"
        assert strip_think("<think>x<think>a</think>y</think>") == "<think>xy</think>"


class TestThinkPairsRecording:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
        self.conv = Conversation()
        self.conv.add_user("q")
        self.conv.add_assistant(THINK)

    def raw(self, name):
        return json.loads((Path(self.tmp) / f"{name}.json").read_text())

    def test_pairs_are_recorded_whenever_given_even_without_omit_think(self):
        self.conv.save(self.tmp, name="x", think_pairs=PAIRS)  # omit_think left False
        data = self.raw("x")
        assert data["think_pairs"] == [["<think>", "</think>"]]
        assert list(data)[-1] == "messages"  # messages stay last
        assert data["messages"][1]["content"] == THINK  # nothing stripped

    def test_the_key_is_omitted_when_there_are_no_pairs(self):
        self.conv.save(self.tmp, name="none")
        self.conv.save(self.tmp, name="empty", think_pairs=[])
        assert "think_pairs" not in self.raw("none")
        assert "think_pairs" not in self.raw("empty")

    def test_at_most_the_most_recent_pairs_are_written(self):
        pairs = [(f"<t{i}>", f"</t{i}>") for i in range(MAX_THINK_PAIRS + 4)]
        self.conv.save(self.tmp, name="many", think_pairs=pairs)
        written = self.raw("many")["think_pairs"]
        assert len(written) == MAX_THINK_PAIRS
        assert written[0] == ["<t4>", "</t4>"]  # the 4 oldest were dropped
        assert written[-1] == [f"<t{MAX_THINK_PAIRS + 3}>", f"</t{MAX_THINK_PAIRS + 3}>"]

    def test_load_returns_tuples_and_a_resave_round_trips_them(self):
        self.conv.save(self.tmp, name="x", think_pairs=PAIRS)
        loaded, _ = Conversation.load(self.tmp, "x")
        assert loaded.think_pairs == [("<think>", "</think>")]
        assert isinstance(loaded.think_pairs[0], tuple)
        loaded.save(self.tmp, name="again", think_pairs=loaded.think_pairs)
        assert self.raw("again")["think_pairs"] == [["<think>", "</think>"]]

    def test_malformed_entries_are_dropped_structurally(self, subtests):
        for name, raw in {
            "notalist": "<think>",
            "dict": {"a": "b"},
            "null": None,
            "short": [["<think>"]],
            "long": [["<a>", "</a>", "<b>"]],
            "nonstr": [[1, 2], ["<a>", None]],
            "mixed": ["x", ["<ok>", "</ok>"], 5, ["<a>"]],
        }.items():
            (Path(self.tmp) / f"{name}.json").write_text(
                json.dumps({"model": "m", "system_prompt": "", "think_pairs": raw, "messages": []})
            )
            with subtests.test(case=name):
                loaded, _ = Conversation.load(self.tmp, name)
                expected = [("<ok>", "</ok>")] if name == "mixed" else []
                assert loaded.think_pairs == expected

    def test_a_file_without_the_key_loads_exactly_as_before(self):
        (Path(self.tmp) / "old.json").write_text(
            json.dumps({"model": "m", "system_prompt": "", "messages": [{"role": "user", "content": "q"}]})
        )
        loaded, model = Conversation.load(self.tmp, "old")
        assert (loaded.think_pairs, model, len(loaded.messages)) == ([], "m", 1)

    def test_a_loaded_file_never_yields_more_than_the_cap(self):
        raw = [[f"<t{i}>", f"</t{i}>"] for i in range(1000)]
        (Path(self.tmp) / "big.json").write_text(
            json.dumps({"model": "m", "system_prompt": "", "think_pairs": raw, "messages": []})
        )
        loaded, _ = Conversation.load(self.tmp, "big")
        assert len(loaded.think_pairs) == MAX_THINK_PAIRS


class TestSave:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
        self.conv = Conversation()
        self.conv.add_user(f"keep this {THINK}")
        self.conv.add_assistant(THINK)
        self.conv.add_user("again")
        self.conv.add_assistant("lone\n</think>\nstray")

    def test_default_save_is_unchanged(self):
        self.conv.save(self.tmp, name="plain")
        assert [m["content"] for m in saved_messages(self.tmp, "plain")] == [
            m["content"] for m in self.conv.messages
        ]

    def test_omit_think_strips_assistant_only_and_keeps_memory(self):
        before = json.dumps(self.conv.messages)
        self.conv.save(self.tmp, name="omit", omit_think=True, think_pairs=PAIRS)
        saved = saved_messages(self.tmp, "omit")
        assert saved[0]["content"] == f"keep this {THINK}"  # user message untouched
        assert saved[1]["content"] == "The answer."
        assert saved[3]["content"] == "lone\n</think>\nstray"  # unbalanced untouched
        assert saved[1]["role"] == "assistant"
        assert "timestamp" in saved[1]
        assert json.dumps(self.conv.messages) == before  # memory not mutated

    def test_thinking_only_reply_is_kept_as_empty(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("<think>only thoughts</think>")
        conv.save(self.tmp, name="empty", omit_think=True, think_pairs=PAIRS)
        assert saved_messages(self.tmp, "empty")[1]["content"] == ""

    def test_load_round_trip(self):
        self.conv.save(self.tmp, name="rt", omit_think=True, think_pairs=PAIRS)
        loaded, _ = Conversation.load(self.tmp, "rt")
        assert loaded.messages[1]["content"] == "The answer."


class TestConfigCommand:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
        self.conv = Conversation()
        self.conv.add_user("q")
        self.conv.add_assistant(THINK)

    def run_config(self, state, args):
        with mock.patch.object(chat, "display_error") as err, mock.patch.object(chat, "display_info") as info:
            handle_command("/config", args, None, self.conv, state)
        return info, err

    def test_default_is_on(self):
        assert DEFAULTS["save_thinking"] is True

    def test_toggle_and_explicit_values(self):
        state = make_state(self.tmp, save_thinking=True)
        info, _ = self.run_config(state, "save_thinking")
        assert state.config["save_thinking"] is False
        info.assert_called_once_with("save_thinking: off")
        self.run_config(state, "save_thinking")
        assert state.config["save_thinking"] is True
        self.run_config(state, "save_thinking off")
        assert state.config["save_thinking"] is False
        self.run_config(state, "save_thinking on")
        assert state.config["save_thinking"] is True

    def test_missing_key_counts_as_on(self):
        state = make_state(self.tmp)
        self.run_config(state, "save_thinking")
        assert state.config["save_thinking"] is False

    def test_bad_value_and_unknown_key(self):
        state = make_state(self.tmp, save_thinking=True)
        _, err = self.run_config(state, "save_thinking maybe")
        err.assert_called_once()
        assert state.config["save_thinking"] is True
        _, err = self.run_config(state, "nonsense")
        assert "save_thinking" in err.call_args.args[0]

    def test_help_and_usage_text_survive_rich_markup(self):
        # Square brackets such as "[on|off]" would be eaten as Rich markup, so
        # render the real output instead of mocking the display functions.
        state = make_state(self.tmp, save_thinking=True)
        with ui.console.capture() as cap:
            ui.print_help()
            handle_command("/config", "save_thinking maybe", None, self.conv, state)
        out = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", cap.get()).split())
        assert "/config save_thinking on|off" in out
        assert "Usage: /config save_thinking on|off" in out

    def test_no_args_still_displays_config(self):
        state = make_state(self.tmp)
        with mock.patch.object(chat, "display_config") as show:
            handle_command("/config", "", None, self.conv, state)
        show.assert_called_once()

    def test_save_and_autosave_honor_the_setting(self, subtests):
        for setting, expected in [(True, THINK), (False, "The answer.")]:
            with subtests.test(save_thinking=setting):
                state = make_state(self.tmp, save_thinking=setting)
                with mock.patch.object(chat, "display_info"):
                    handle_command("/save", f"manual_{setting}", None, self.conv, state)
                _auto_save(self.conv, state)
                assert saved_messages(self.tmp, f"manual_{setting}")[1]["content"] == expected
                assert saved_messages(self.tmp, "auto_test")[1]["content"] == expected
        assert self.conv.messages[1]["content"] == THINK  # memory untouched
