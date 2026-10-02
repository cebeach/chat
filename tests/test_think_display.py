"""A newline is drawn after the tag that closes a thinking block (display only).

Models differ: Qwen and DeepSeek already write "\\n\\n" after </think>, while Gemma
and gpt-oss run straight from the closing tag into the answer. The stored reply
must stay exactly what the model produced; only what is drawn changes.
"""

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

import ui
from chat import State, _think_end, handle_command
from config import DEFAULTS
from conversation import Conversation
from ui import ThinkSeparator, display_assistant_stream, separate_thinking

GEMMA_END = "<channel|>"
GPTOSS_END = "<|end|><|start|>assistant<|channel|>final<|message|>"


def feed_all(end, pieces):
    sep = ThinkSeparator(end)
    return "".join(sep.feed(p) for p in pieces)


def chunks(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


class ThinkSeparatorTests(unittest.TestCase):
    def test_tag_as_its_own_piece_followed_by_the_answer(self):
        # What Gemma's stream really looks like: the tag is one piece.
        self.assertEqual(
            feed_all(GEMMA_END, ["The answer is 144.", "<channel|>", "1", "4", "4"]),
            "The answer is 144.<channel|>\n144",
        )

    def test_answer_in_the_same_piece_as_the_tag(self):
        self.assertEqual(feed_all(GEMMA_END, ["why<channel|>144"]), "why<channel|>\n144")

    def test_tag_split_across_pieces(self):
        self.assertEqual(feed_all("</think>", ["why", "</", "think", ">", "answer"]), "why</think>\nanswer")
        self.assertEqual(feed_all("</think>", ["why</th", "ink>ans", "wer"]), "why</think>\nanswer")

    def test_no_newline_added_when_one_already_follows(self):
        for follow in ("\n\nanswer", "\nanswer", "\r\nanswer"):
            with self.subTest(follow=follow):
                self.assertEqual(feed_all("</think>", ["why", "</think>", follow]), "why</think>" + follow)
                self.assertEqual(feed_all("</think>", ["why</think>" + follow]), "why</think>" + follow)

    def test_nothing_is_added_when_the_reply_ends_at_the_tag(self):
        self.assertEqual(feed_all(GEMMA_END, ["only thinking", "<channel|>"]), "only thinking<channel|>")
        self.assertEqual(feed_all(GEMMA_END, ["only thinking<channel|>"]), "only thinking<channel|>")

    def test_a_multi_marker_end_split_at_every_position(self):
        text = f"reason{GPTOSS_END}Hi"
        expected = f"reason{GPTOSS_END}\nHi"
        for cut in range(1, len(text)):
            with self.subTest(cut=cut):
                self.assertEqual(feed_all(GPTOSS_END, [text[:cut], text[cut:]]), expected)
        for size in range(1, 12):
            with self.subTest(size=size):
                self.assertEqual(feed_all(GPTOSS_END, chunks(text, size)), expected)

    def test_every_chunk_size_gives_the_same_result(self):
        text = "a</think>b</think>\nc</think>d</think>"
        expected = "a</think>\nb</think>\nc</think>\nd</think>"
        for size in range(1, len(text) + 1):
            with self.subTest(size=size):
                self.assertEqual(feed_all("</think>", chunks(text, size)), expected)

    def test_each_thinking_block_is_handled(self):
        self.assertEqual(
            feed_all(GEMMA_END, ["t1<channel|>", "a1", "t2<channel|>", "a2"]),
            "t1<channel|>\na1t2<channel|>\na2",
        )

    def test_text_that_only_looks_like_the_tag_is_left_alone(self):
        for text in ("x<chan", "x<channel|", "<channel>", "channel|>", "plain text"):
            with self.subTest(text=text):
                self.assertEqual(feed_all(GEMMA_END, [text, " more"]), text + " more")

    def test_an_empty_end_does_nothing(self):
        self.assertEqual(feed_all("", ["a", "b", "\n"]), "ab\n")

    def test_a_single_character_end(self):
        self.assertEqual(feed_all("|", ["a|", "b|\n", "c"]), "a|\nb|\nc")


def draw(tokens, think_end=None):
    """Run display_assistant_stream on a token list; return (what was drawn, returned text)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        text = display_assistant_stream(iter(tokens), think_end=think_end)
    return out.getvalue(), text


class DisplayAssistantStreamTests(unittest.TestCase):
    TOKENS = ["<|channel>thought\n", "The answer is 144.", "<channel|>", "1", "4", "4"]

    def test_the_screen_gets_the_newline_but_the_stored_reply_does_not(self):
        drawn, text = draw(self.TOKENS, think_end=GEMMA_END)
        self.assertIn("The answer is 144.<channel|>\n144", drawn)
        self.assertEqual(text, "".join(self.TOKENS))  # byte for byte what the model produced
        self.assertNotIn("<channel|>\n", text)

    def test_without_a_known_tag_nothing_changes(self):
        drawn, text = draw(self.TOKENS, think_end=None)
        self.assertIn("The answer is 144.<channel|>144", drawn)
        self.assertEqual(text, "".join(self.TOKENS))

    def test_a_model_that_already_writes_a_newline_looks_as_before(self):
        tokens = ["<think>\n", "why", "</think>", "\n\n", "answer"]
        with_end, text = draw(tokens, think_end="</think>")
        without, _ = draw(tokens, think_end=None)
        self.assertEqual(with_end.split("\n", 1)[1], without.split("\n", 1)[1])  # after the header line
        self.assertEqual(text, "".join(tokens))

    def test_the_tag_arriving_in_pieces_is_still_separated(self):
        drawn, text = draw(["why", "</", "think", ">", "answer"], think_end="</think>")
        self.assertIn("why</think>\nanswer", drawn)
        self.assertEqual(text, "why</think>answer")

    def test_an_interrupted_stream_keeps_the_stored_text_unchanged(self):
        # A tag that ended exactly at the interruption leaves a pending newline; it
        # is never drawn and never leaks into the stored reply. (As before this
        # feature, the last partial word of an interrupted reply is not drawn.)
        def gen():
            yield "thinking<channel|>"
            raise KeyboardInterrupt

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            text = display_assistant_stream(gen(), think_end=GEMMA_END)
        self.assertEqual(text, "thinking<channel|> [interrupted]")
        self.assertNotIn("<channel|>\n", text)
        self.assertTrue(out.getvalue().endswith("\n"))

    def test_a_long_reply_still_wraps(self):
        words = " ".join(["word"] * 60)
        drawn, text = draw([words, "<channel|>", "x"], think_end=GEMMA_END)
        self.assertEqual(text, words + "<channel|>x")
        self.assertIn("<channel|>\nx", drawn)


class SeparateThinkingTests(unittest.TestCase):
    def test_adds_a_newline_after_the_tag_unless_one_follows_or_the_text_ends(self):
        self.assertEqual(separate_thinking("why<channel|>144", [GEMMA_END]), "why<channel|>\n144")
        self.assertEqual(separate_thinking("why</think>\n\nanswer", ["</think>"]), "why</think>\n\nanswer")
        self.assertEqual(separate_thinking("why</think>\r\nanswer", ["</think>"]), "why</think>\r\nanswer")
        self.assertEqual(separate_thinking("why<channel|>", [GEMMA_END]), "why<channel|>")

    def test_every_occurrence_and_every_known_end(self):
        text = "a<channel|>b</think>c<channel|>d"
        self.assertEqual(
            separate_thinking(text, [GEMMA_END, "</think>"]), "a<channel|>\nb</think>\nc<channel|>\nd"
        )

    def test_empty_ends_and_no_ends_change_nothing(self):
        self.assertEqual(separate_thinking("a<channel|>b", []), "a<channel|>b")
        self.assertEqual(separate_thinking("a<channel|>b", [""]), "a<channel|>b")

    def test_tags_with_regex_characters_are_matched_literally(self):
        self.assertEqual(separate_thinking("a[/THINK]b", ["[/THINK]"]), "a[/THINK]\nb")
        self.assertEqual(separate_thinking("a<|end|>b", ["<|end|>"]), "a<|end|>\nb")

    def test_streaming_and_static_agree_for_every_chunking(self):
        samples = [
            ("why<channel|>144", GEMMA_END),
            ("a</think>b</think>\nc</think>d</think>", "</think>"),
            (f"reason{GPTOSS_END}Hi", GPTOSS_END),
            ("[THINK]t[/THINK]answer", "[/THINK]"),
            ("no tag here at all", GEMMA_END),
            ("ends at the tag<channel|>", GEMMA_END),
        ]
        for text, end in samples:
            expected = separate_thinking(text, [end])
            for size in range(1, len(text) + 1):
                with self.subTest(text=text, size=size):
                    self.assertEqual(feed_all(end, chunks(text, size)), expected)


def saved_conversation(tmp, think_pairs, with_pairs=True):
    conv = Conversation()
    conv.add_user("question mentioning <channel|> literally")
    conv.add_assistant("<|channel>thought\nwhy<channel|>The answer.", model="/m/gemma.gguf")
    conv.save(tmp, name="saved", model="/m/gemma.gguf", think_pairs=think_pairs if with_pairs else ())


class CatSeparationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.state = State(model="m", config={**DEFAULTS, "conversations_dir": self.tmp}, context_length=None)

    def cat(self):
        with ui.console.capture() as cap:
            handle_command("/cat", "saved", None, Conversation(), self.state)
        return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())

    def test_assistant_replies_get_the_newline_and_user_messages_do_not(self):
        saved_conversation(self.tmp, [("<|channel>thought", "<channel|>")])
        out = self.cat()
        self.assertIn("why<channel|>\nThe answer.", out)
        self.assertIn("question mentioning <channel|> literally", out)  # user text untouched

    def test_a_file_without_recorded_pairs_prints_as_before(self):
        saved_conversation(self.tmp, [], with_pairs=False)
        self.assertIn("why<channel|>The answer.", self.cat())

    def test_invalid_recorded_pairs_are_ignored(self):
        saved_conversation(self.tmp, [])
        path = Path(self.tmp) / "saved.json"
        data = json.loads(path.read_text())
        data["think_pairs"] = [["", ""], ["the", "The"], ["<a>", "so the answer is"], "junk", ["<x>"]]
        path.write_text(json.dumps(data))
        self.assertIn("why<channel|>The answer.", self.cat())

    def test_saved_text_that_looks_like_markup_prints_literally_instead_of_raising(self):
        conv = Conversation()
        conv.add_user("see [/path] please")
        conv.add_assistant("[THINK]hmm[/THINK]answer [/etc/hosts]", model="/m/x.gguf")
        conv.save(self.tmp, name="saved", model="/m/x.gguf", think_pairs=[("[THINK]", "[/THINK]")])
        out = self.cat()
        self.assertIn("see [/path] please", out)
        self.assertIn("[THINK]hmm[/THINK]\nanswer [/etc/hosts]", out)  # bracket-style pair separated too

    def test_the_stored_file_never_contains_the_added_newline(self):
        saved_conversation(self.tmp, [("<|channel>thought", "<channel|>")])
        self.cat()
        stored = json.loads((Path(self.tmp) / "saved.json").read_text())["messages"][1]["content"]
        self.assertEqual(stored, "<|channel>thought\nwhy<channel|>The answer.")


class ThinkEndTests(unittest.TestCase):
    def state(self, tags):
        return State(model="m", config={**DEFAULTS}, context_length=None, think_tags=tags)

    def test_returns_the_active_end_tag(self):
        self.assertEqual(
            _think_end(self.state(("<|channel>thought", "<channel|>", "detected"))), "<channel|>"
        )

    def test_none_when_no_tags_are_known(self):
        self.assertIsNone(_think_end(self.state(None)))


if __name__ == "__main__":
    unittest.main()
