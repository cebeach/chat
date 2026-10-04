import json
import re
from pathlib import Path
from unittest import mock

import pytest

import chat
import ui
from chat import State, _auto_save, handle_command
from config import DEFAULTS
from conversation import Conversation, split_think

THINKING = "reasoning here"
ANSWER = "The answer."
TAGS = ("<think>", "</think>", "detected")


def make_state(tmp, **config):
    return State(
        model="m",
        config={"conversations_dir": str(tmp), "auto_save": True, **config},
        context_length=None,
        auto_save_name="auto_test",
        think_tags=TAGS,
    )


def saved_messages(tmp, name):
    return json.loads((Path(tmp) / f"{name}.json").read_text())["messages"]


class TestSplitThink:
    def test_a_balanced_block_is_split_off_without_its_tags(self):
        assert split_think("<think>\nreasoning here\n</think>\nThe answer.", TAGS) == (THINKING, ANSWER)

    def test_tags_may_be_a_pair_or_the_three_item_tuple(self):
        assert split_think("<t>a</t>b", ("<t>", "</t>")) == ("a", "b")
        assert split_think("<t>a</t>b", ("<t>", "</t>", "config")) == ("a", "b")

    def test_several_blocks_are_joined_with_a_blank_line(self):
        assert split_think("<think>a</think>X<think>b</think>Y", TAGS) == ("a\n\nb", "XY")

    def test_an_unclosed_block_takes_the_rest_and_leaves_no_content(self):
        assert split_think("<think>\nreasoning cut off", TAGS) == ("reasoning cut off", "")
        assert split_think("answer first<think>then cut off", TAGS) == ("then cut off", "answer first")

    def test_the_stream_prefixed_opener_is_just_the_start_of_the_text(self):
        # A forced-open template (Qwen) never generates the opener; the stream re-emits it.
        assert split_think("<think>\nwhy\n</think>\n\nanswer", TAGS) == ("why", "answer")

    def test_no_match_returns_the_text_untouched(self, subtests):
        for tags, text in [
            (None, "<think>a</think>b"),
            (TAGS, "no tags at all"),
            (TAGS, "reasoning\n</think>\nanswer"),  # a lone closer is not a block
            (("", "</think>"), "<think>a</think>b"),
            (("<think>", ""), "<think>a</think>b"),
            (("[THINK]", "[/THINK]"), "<think>a</think>b"),  # another model's tags
        ]:
            with subtests.test(tags=tags, text=text):
                assert split_think(text, tags) == ("", text)

    def test_other_models_tags_in_the_text_are_left_alone(self):
        assert split_think("<think>a</think>b [THINK]c[/THINK]", TAGS) == ("a", "b [THINK]c[/THINK]")


class TestAddAssistant:
    def test_thinking_is_stored_before_content_when_given(self):
        conv = Conversation()
        conv.add_assistant(ANSWER, model="m", thinking=THINKING)
        assert list(conv.messages[0]) == ["role", "timestamp", "model", "thinking", "content"]

    def test_empty_or_missing_thinking_adds_no_key(self, subtests):
        for thinking in (None, ""):
            with subtests.test(thinking=thinking):
                conv = Conversation()
                conv.add_assistant(ANSWER, thinking=thinking)
                assert "thinking" not in conv.messages[0]


class TestGetMessages:
    def test_thinking_is_never_sent_to_the_model(self):
        conv = Conversation(system_prompt="sys")
        conv.add_user("q")
        conv.add_assistant(ANSWER, thinking=THINKING)
        assert conv.get_messages() == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": ANSWER},
        ]

    def test_an_empty_reply_is_left_out_with_the_user_message_before_it(self):
        conv = Conversation()
        conv.add_user("q1")
        conv.add_assistant("", thinking="cut off")
        conv.add_user("q2")
        conv.add_assistant("a2")
        conv.add_user("q3")
        assert [m["content"] for m in conv.get_messages()] == ["q2", "a2", "q3"]
        assert len(conv.messages) == 5  # memory untouched

    def test_only_the_assistant_message_goes_when_no_user_message_precedes_it(self):
        conv = Conversation()
        conv.add_assistant("a0")
        conv.add_assistant("", thinking="cut off")
        conv.add_user("q")
        assert [m["content"] for m in conv.get_messages()] == ["a0", "q"]

    def test_no_new_consecutive_user_pair_appears(self):
        conv = Conversation()
        conv.add_user("q1")
        conv.add_assistant("a1")
        conv.add_user("q2")
        conv.add_assistant("", thinking="cut off")
        conv.add_user("q3")
        roles = [m["role"] for m in conv.get_messages()]
        assert all(a != b for a, b in zip(roles, roles[1:]))

    def test_a_pending_user_message_at_the_end_is_never_dropped(self):
        conv = Conversation()
        conv.add_user("q")
        assert conv.get_messages() == [{"role": "user", "content": "q"}]


class TestSave:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
        self.conv = Conversation()
        self.conv.add_user("keep this <think>x</think>")
        self.conv.add_assistant(ANSWER, thinking=THINKING)
        self.conv.add_user("again")
        self.conv.add_assistant("plain reply")

    def raw(self, name):
        return json.loads((Path(self.tmp) / f"{name}.json").read_text())

    def test_the_file_has_no_think_pairs_and_messages_stay_last(self):
        self.conv.save(self.tmp, name="plain")
        data = self.raw("plain")
        assert "think_pairs" not in data
        assert list(data)[-1] == "messages"

    def test_thinking_is_written_by_default(self):
        self.conv.save(self.tmp, name="plain")
        saved = saved_messages(self.tmp, "plain")
        assert saved[1]["thinking"] == THINKING
        assert saved[1]["content"] == ANSWER
        assert "thinking" not in saved[3]

    def test_omit_thinking_drops_the_field_and_keeps_memory(self):
        before = json.dumps(self.conv.messages)
        self.conv.save(self.tmp, name="omit", omit_thinking=True)
        saved = saved_messages(self.tmp, "omit")
        assert all("thinking" not in m for m in saved)
        assert [m["content"] for m in saved] == [m["content"] for m in self.conv.messages]
        assert saved[1]["role"] == "assistant"
        assert "timestamp" in saved[1]
        assert json.dumps(self.conv.messages) == before  # memory not mutated

    def test_a_thinking_only_reply_is_saved_with_empty_content(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("", thinking="only thoughts")
        conv.save(self.tmp, name="empty")
        saved = saved_messages(self.tmp, "empty")[1]
        assert (saved["thinking"], saved["content"]) == ("only thoughts", "")

    def test_load_round_trip(self):
        self.conv.save(self.tmp, name="rt")
        loaded, _ = Conversation.load(self.tmp, "rt")
        assert loaded.messages == self.conv.messages


class TestConfigCommand:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
        self.conv = Conversation()
        self.conv.add_user("q")
        self.conv.add_assistant(ANSWER, thinking=THINKING)

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
        for setting, expected in [(True, THINKING), (False, None)]:
            with subtests.test(save_thinking=setting):
                state = make_state(self.tmp, save_thinking=setting)
                with mock.patch.object(chat, "display_info"):
                    handle_command("/save", f"manual_{setting}", None, self.conv, state)
                _auto_save(self.conv, state)
                assert saved_messages(self.tmp, f"manual_{setting}")[1].get("thinking") == expected
                assert saved_messages(self.tmp, "auto_test")[1].get("thinking") == expected
        assert self.conv.messages[1]["thinking"] == THINKING  # memory untouched
