"""A newline is drawn after the tag that closes a thinking block (display only).

Models differ: Qwen and DeepSeek already write "\\n\\n" after </think>, while Gemma
and gpt-oss run straight from the closing tag into the answer. The stored reply
must stay exactly what the model produced; only what is drawn changes.
"""

import contextlib
import io
import unittest

from chat import State, _think_end
from config import DEFAULTS
from ui import ThinkSeparator, display_assistant_stream

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
