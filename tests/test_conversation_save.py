import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chat
import ui
from chat import State, _auto_save, handle_command
from config import DEFAULTS
from conversation import Conversation, strip_think

THINK = "<think>\nreasoning here\n</think>\nThe answer."


def make_state(tmp, **config):
    return State(
        model="m",
        config={"conversations_dir": str(tmp), "auto_save": True, **config},
        context_length=None,
        auto_save_name="auto_test",
    )


def saved_messages(tmp, name):
    return json.loads((Path(tmp) / f"{name}.json").read_text())["messages"]


class StripThinkTests(unittest.TestCase):
    def test_balanced_blocks_removed(self):
        self.assertEqual(strip_think(THINK), "The answer.")
        self.assertEqual(strip_think("<think>a</think>X<think>b</think>Y"), "XY")

    def test_unbalanced_left_byte_identical(self):
        for text in [
            "reasoning\n</think>\nanswer",  # lone closing tag
            "<think>\nreasoning cut off [interrupted]",  # lone opening tag
            "</think>a<think>",  # reversed
            "no tags at all",
        ]:
            with self.subTest(text=text):
                self.assertEqual(strip_think(text), text)

    def test_pair_plus_stray_tag_only_removes_pair(self):
        self.assertEqual(strip_think("<think>a</think> x </think>"), "x </think>")

    def test_nested_only_the_innermost_balanced_pair_is_removed(self):
        # The inner pair is balanced; the outer tags are strays and stay as evidence.
        self.assertEqual(strip_think("<think><think>a</think>b"), "<think>b")
        self.assertEqual(strip_think("<think>x<think>a</think>y</think>"), "<think>xy</think>")


class SaveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.conv = Conversation()
        self.conv.add_user(f"keep this {THINK}")
        self.conv.add_assistant(THINK)
        self.conv.add_user("again")
        self.conv.add_assistant("lone\n</think>\nstray")

    def test_default_save_is_unchanged(self):
        self.conv.save(self.tmp, name="plain")
        self.assertEqual(
            [m["content"] for m in saved_messages(self.tmp, "plain")],
            [m["content"] for m in self.conv.messages],
        )

    def test_omit_think_strips_assistant_only_and_keeps_memory(self):
        before = json.dumps(self.conv.messages)
        self.conv.save(self.tmp, name="omit", omit_think=True)
        saved = saved_messages(self.tmp, "omit")
        self.assertEqual(saved[0]["content"], f"keep this {THINK}")  # user message untouched
        self.assertEqual(saved[1]["content"], "The answer.")
        self.assertEqual(saved[3]["content"], "lone\n</think>\nstray")  # unbalanced untouched
        self.assertEqual(saved[1]["role"], "assistant")
        self.assertIn("timestamp", saved[1])
        self.assertEqual(json.dumps(self.conv.messages), before)  # memory not mutated

    def test_thinking_only_reply_is_kept_as_empty(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("<think>only thoughts</think>")
        conv.save(self.tmp, name="empty", omit_think=True)
        self.assertEqual(saved_messages(self.tmp, "empty")[1]["content"], "")

    def test_load_round_trip(self):
        self.conv.save(self.tmp, name="rt", omit_think=True)
        loaded, _ = Conversation.load(self.tmp, "rt")
        self.assertEqual(loaded.messages[1]["content"], "The answer.")


class ConfigCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.conv = Conversation()
        self.conv.add_user("q")
        self.conv.add_assistant(THINK)

    def run_config(self, state, args):
        with mock.patch.object(chat, "display_error") as err, mock.patch.object(chat, "display_info") as info:
            handle_command("/config", args, None, self.conv, state)
        return info, err

    def test_default_is_on(self):
        self.assertIs(DEFAULTS["save_thinking"], True)

    def test_toggle_and_explicit_values(self):
        state = make_state(self.tmp, save_thinking=True)
        info, _ = self.run_config(state, "save_thinking")
        self.assertIs(state.config["save_thinking"], False)
        info.assert_called_once_with("save_thinking: off")
        self.run_config(state, "save_thinking")
        self.assertIs(state.config["save_thinking"], True)
        self.run_config(state, "save_thinking off")
        self.assertIs(state.config["save_thinking"], False)
        self.run_config(state, "save_thinking on")
        self.assertIs(state.config["save_thinking"], True)

    def test_missing_key_counts_as_on(self):
        state = make_state(self.tmp)
        self.run_config(state, "save_thinking")
        self.assertIs(state.config["save_thinking"], False)

    def test_bad_value_and_unknown_key(self):
        state = make_state(self.tmp, save_thinking=True)
        _, err = self.run_config(state, "save_thinking maybe")
        err.assert_called_once()
        self.assertIs(state.config["save_thinking"], True)
        _, err = self.run_config(state, "nonsense")
        self.assertIn("save_thinking", err.call_args.args[0])

    def test_help_and_usage_text_survive_rich_markup(self):
        # Square brackets such as "[on|off]" would be eaten as Rich markup, so
        # render the real output instead of mocking the display functions.
        state = make_state(self.tmp, save_thinking=True)
        with ui.console.capture() as cap:
            ui.print_help()
            handle_command("/config", "save_thinking maybe", None, self.conv, state)
        out = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", cap.get()).split())
        self.assertIn("/config save_thinking on|off", out)
        self.assertIn("Usage: /config save_thinking on|off", out)

    def test_no_args_still_displays_config(self):
        state = make_state(self.tmp)
        with mock.patch.object(chat, "display_config") as show:
            handle_command("/config", "", None, self.conv, state)
        show.assert_called_once()

    def test_save_and_autosave_honor_the_setting(self):
        for setting, expected in [(True, THINK), (False, "The answer.")]:
            with self.subTest(save_thinking=setting):
                state = make_state(self.tmp, save_thinking=setting)
                with mock.patch.object(chat, "display_info"):
                    handle_command("/save", f"manual_{setting}", None, self.conv, state)
                _auto_save(self.conv, state)
                self.assertEqual(saved_messages(self.tmp, f"manual_{setting}")[1]["content"], expected)
                self.assertEqual(saved_messages(self.tmp, "auto_test")[1]["content"], expected)
        self.assertEqual(self.conv.messages[1]["content"], THINK)  # memory untouched


if __name__ == "__main__":
    unittest.main()
