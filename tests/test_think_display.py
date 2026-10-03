"""A newline is drawn after the tag that closes a thinking block (display only).

Models differ: Qwen and DeepSeek already write "\\n\\n" after </think>, while Gemma
and gpt-oss run straight from the closing tag into the answer. The stored reply
must stay exactly what the model produced; only what is drawn changes.
"""

import contextlib
import io
import json
import re
from pathlib import Path

import pytest

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


class TestThinkSeparator:
    def test_tag_as_its_own_piece_followed_by_the_answer(self):
        # What Gemma's stream really looks like: the tag is one piece.
        assert (
            feed_all(GEMMA_END, ["The answer is 144.", "<channel|>", "1", "4", "4"])
            == "The answer is 144.<channel|>\n144"
        )

    def test_answer_in_the_same_piece_as_the_tag(self):
        assert feed_all(GEMMA_END, ["why<channel|>144"]) == "why<channel|>\n144"

    def test_tag_split_across_pieces(self):
        assert feed_all("</think>", ["why", "</", "think", ">", "answer"]) == "why</think>\nanswer"
        assert feed_all("</think>", ["why</th", "ink>ans", "wer"]) == "why</think>\nanswer"

    def test_no_newline_added_when_one_already_follows(self, subtests):
        for follow in ("\n\nanswer", "\nanswer", "\r\nanswer"):
            with subtests.test(follow=follow):
                assert feed_all("</think>", ["why", "</think>", follow]) == "why</think>" + follow
                assert feed_all("</think>", ["why</think>" + follow]) == "why</think>" + follow

    def test_nothing_is_added_when_the_reply_ends_at_the_tag(self):
        assert feed_all(GEMMA_END, ["only thinking", "<channel|>"]) == "only thinking<channel|>"
        assert feed_all(GEMMA_END, ["only thinking<channel|>"]) == "only thinking<channel|>"

    def test_a_multi_marker_end_split_at_every_position(self, subtests):
        text = f"reason{GPTOSS_END}Hi"
        expected = f"reason{GPTOSS_END}\nHi"
        for cut in range(1, len(text)):
            with subtests.test(cut=cut):
                assert feed_all(GPTOSS_END, [text[:cut], text[cut:]]) == expected
        for size in range(1, 12):
            with subtests.test(size=size):
                assert feed_all(GPTOSS_END, chunks(text, size)) == expected

    def test_every_chunk_size_gives_the_same_result(self, subtests):
        text = "a</think>b</think>\nc</think>d</think>"
        expected = "a</think>\nb</think>\nc</think>\nd</think>"
        for size in range(1, len(text) + 1):
            with subtests.test(size=size):
                assert feed_all("</think>", chunks(text, size)) == expected

    def test_each_thinking_block_is_handled(self):
        assert (
            feed_all(GEMMA_END, ["t1<channel|>", "a1", "t2<channel|>", "a2"])
            == "t1<channel|>\na1t2<channel|>\na2"
        )

    def test_text_that_only_looks_like_the_tag_is_left_alone(self, subtests):
        for text in ("x<chan", "x<channel|", "<channel>", "channel|>", "plain text"):
            with subtests.test(text=text):
                assert feed_all(GEMMA_END, [text, " more"]) == text + " more"

    def test_an_empty_end_does_nothing(self):
        assert feed_all("", ["a", "b", "\n"]) == "ab\n"

    def test_a_single_character_end(self):
        assert feed_all("|", ["a|", "b|\n", "c"]) == "a|\nb|\nc"


def draw(tokens, think_end=None):
    """Run display_assistant_stream on a token list; return (what was drawn, returned text)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        text = display_assistant_stream(iter(tokens), think_end=think_end)
    return out.getvalue(), text


class TestDisplayAssistantStream:
    TOKENS = ["<|channel>thought\n", "The answer is 144.", "<channel|>", "1", "4", "4"]

    def test_the_screen_gets_the_newline_but_the_stored_reply_does_not(self):
        drawn, text = draw(self.TOKENS, think_end=GEMMA_END)
        assert "The answer is 144.<channel|>\n144" in drawn
        assert text == "".join(self.TOKENS)  # byte for byte what the model produced
        assert "<channel|>\n" not in text

    def test_without_a_known_tag_nothing_changes(self):
        drawn, text = draw(self.TOKENS, think_end=None)
        assert "The answer is 144.<channel|>144" in drawn
        assert text == "".join(self.TOKENS)

    def test_a_model_that_already_writes_a_newline_looks_as_before(self):
        tokens = ["<think>\n", "why", "</think>", "\n\n", "answer"]
        with_end, text = draw(tokens, think_end="</think>")
        without, _ = draw(tokens, think_end=None)
        assert with_end.split("\n", 1)[1] == without.split("\n", 1)[1]  # after the header line
        assert text == "".join(tokens)

    def test_the_tag_arriving_in_pieces_is_still_separated(self):
        drawn, text = draw(["why", "</", "think", ">", "answer"], think_end="</think>")
        assert "why</think>\nanswer" in drawn
        assert text == "why</think>answer"

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
        assert text == "thinking<channel|> [interrupted]"
        assert "<channel|>\n" not in text
        assert out.getvalue().endswith("\n")

    def test_a_long_reply_still_wraps(self):
        words = " ".join(["word"] * 60)
        drawn, text = draw([words, "<channel|>", "x"], think_end=GEMMA_END)
        assert text == words + "<channel|>x"
        assert "<channel|>\nx" in drawn


class TestSeparateThinking:
    def test_adds_a_newline_after_the_tag_unless_one_follows_or_the_text_ends(self):
        assert separate_thinking("why<channel|>144", [GEMMA_END]) == "why<channel|>\n144"
        assert separate_thinking("why</think>\n\nanswer", ["</think>"]) == "why</think>\n\nanswer"
        assert separate_thinking("why</think>\r\nanswer", ["</think>"]) == "why</think>\r\nanswer"
        assert separate_thinking("why<channel|>", [GEMMA_END]) == "why<channel|>"

    def test_every_occurrence_and_every_known_end(self):
        text = "a<channel|>b</think>c<channel|>d"
        assert separate_thinking(text, [GEMMA_END, "</think>"]) == "a<channel|>\nb</think>\nc<channel|>\nd"

    def test_empty_ends_and_no_ends_change_nothing(self):
        assert separate_thinking("a<channel|>b", []) == "a<channel|>b"
        assert separate_thinking("a<channel|>b", [""]) == "a<channel|>b"

    def test_tags_with_regex_characters_are_matched_literally(self):
        assert separate_thinking("a[/THINK]b", ["[/THINK]"]) == "a[/THINK]\nb"
        assert separate_thinking("a<|end|>b", ["<|end|>"]) == "a<|end|>\nb"

    def test_streaming_and_static_agree_for_every_chunking(self, subtests):
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
                with subtests.test(text=text, size=size):
                    assert feed_all(end, chunks(text, size)) == expected


def saved_conversation(tmp, think_pairs, with_pairs=True):
    conv = Conversation()
    conv.add_user("question mentioning <channel|> literally")
    conv.add_assistant("<|channel>thought\nwhy<channel|>The answer.", model="/m/gemma.gguf")
    conv.save(tmp, name="saved", model="/m/gemma.gguf", think_pairs=think_pairs if with_pairs else ())


class TestCatSeparation:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
        self.state = State(model="m", config={**DEFAULTS, "conversations_dir": self.tmp}, context_length=None)

    def cat(self):
        with ui.console.capture() as cap:
            handle_command("/cat", "saved", None, Conversation(), self.state)
        return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())

    def test_assistant_replies_get_the_newline_and_user_messages_do_not(self):
        saved_conversation(self.tmp, [("<|channel>thought", "<channel|>")])
        out = self.cat()
        assert "why<channel|>\nThe answer." in out
        assert "question mentioning <channel|> literally" in out  # user text untouched

    def test_a_file_without_recorded_pairs_prints_as_before(self):
        saved_conversation(self.tmp, [], with_pairs=False)
        assert "why<channel|>The answer." in self.cat()

    def test_invalid_recorded_pairs_are_ignored(self):
        saved_conversation(self.tmp, [])
        path = Path(self.tmp) / "saved.json"
        data = json.loads(path.read_text())
        data["think_pairs"] = [["", ""], ["the", "The"], ["<a>", "so the answer is"], "junk", ["<x>"]]
        path.write_text(json.dumps(data))
        assert "why<channel|>The answer." in self.cat()

    def test_saved_text_that_looks_like_markup_prints_literally_instead_of_raising(self):
        conv = Conversation()
        conv.add_user("see [/path] please")
        conv.add_assistant("[THINK]hmm[/THINK]answer [/etc/hosts]", model="/m/x.gguf")
        conv.save(self.tmp, name="saved", model="/m/x.gguf", think_pairs=[("[THINK]", "[/THINK]")])
        out = self.cat()
        assert "see [/path] please" in out
        assert "[THINK]hmm[/THINK]\nanswer [/etc/hosts]" in out  # bracket-style pair separated too

    def test_the_stored_file_never_contains_the_added_newline(self):
        saved_conversation(self.tmp, [("<|channel>thought", "<channel|>")])
        self.cat()
        stored = json.loads((Path(self.tmp) / "saved.json").read_text())["messages"][1]["content"]
        assert stored == "<|channel>thought\nwhy<channel|>The answer."


class TestThinkEnd:
    def state(self, tags):
        return State(model="m", config={**DEFAULTS}, context_length=None, think_tags=tags)

    def test_returns_the_active_end_tag(self):
        assert _think_end(self.state(("<|channel>thought", "<channel|>", "detected"))) == "<channel|>"

    def test_none_when_no_tags_are_known(self):
        assert _think_end(self.state(None)) is None
