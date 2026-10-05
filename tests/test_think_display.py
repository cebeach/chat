"""An "*** END OF THINKING ***" line is drawn after the tag that closes a thinking block (display only).

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
from ui import ThinkSeparator, display_assistant_stream

D = ui.THINK_DELIMITER  # "\n\n*** END OF THINKING ***\n\n\n"
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
            == f"The answer is 144.<channel|>{D}144"
        )

    def test_answer_in_the_same_piece_as_the_tag(self):
        assert feed_all(GEMMA_END, ["why<channel|>144"]) == f"why<channel|>{D}144"

    def test_tag_split_across_pieces(self):
        assert feed_all("</think>", ["why", "</", "think", ">", "answer"]) == f"why</think>{D}answer"
        assert feed_all("</think>", ["why</th", "ink>ans", "wer"]) == f"why</think>{D}answer"

    def test_every_newline_after_the_tag_is_absorbed_into_the_delimiter(self, subtests):
        cases = {
            "\nanswer": f"{D}answer",
            "\n\nanswer": f"{D}answer",
            "\n\n\nanswer": f"{D}answer",
            "\r\nanswer": f"{D}\r\nanswer",  # a "\r" is ordinary text
        }
        for follow, drawn in cases.items():
            with subtests.test(follow=follow):
                assert feed_all("</think>", ["why", "</think>", follow]) == "why</think>" + drawn
                assert feed_all("</think>", ["why</think>" + follow]) == "why</think>" + drawn

    def test_the_newline_may_arrive_as_its_own_piece_or_after_a_split_tag(self):
        want = f"why</think>{D}answer"
        assert feed_all("</think>", ["why</think>", "\n", "answer"]) == want
        assert feed_all("</think>", ["why</think>", "\nanswer"]) == want
        assert feed_all("</think>", ["why</th", "ink>\n", "answer"]) == want
        assert feed_all("</think>", ["why</think>", "", "\nanswer"]) == want
        assert feed_all("</think>", ["why</think>", "\n", "\n", "answer"]) == want
        assert feed_all("</think>", ["why</think>\n", "\n", "answer"]) == want

    def test_two_tags_in_a_row_each_get_one_delimiter(self):
        assert feed_all("</think>", ["a</think>", "\n", "\nb"]) == f"a</think>{D}b"
        assert feed_all("</think>", ["a</think></think>b"]) == f"a</think>{D}</think>{D}b"

    def test_the_delimiter_is_drawn_when_the_reply_ends_at_the_tag(self):
        assert feed_all(GEMMA_END, ["only thinking", "<channel|>"]) == f"only thinking<channel|>{D}"
        assert feed_all(GEMMA_END, ["only thinking<channel|>"]) == f"only thinking<channel|>{D}"

    def test_a_multi_marker_end_split_at_every_position(self, subtests):
        text = f"reason{GPTOSS_END}Hi"
        expected = f"reason{GPTOSS_END}{D}Hi"
        for cut in range(1, len(text)):
            with subtests.test(cut=cut):
                assert feed_all(GPTOSS_END, [text[:cut], text[cut:]]) == expected
        for size in range(1, 12):
            with subtests.test(size=size):
                assert feed_all(GPTOSS_END, chunks(text, size)) == expected

    def test_every_chunk_size_gives_the_same_result(self, subtests):
        text = "a</think>b</think>\nc</think>d</think>"
        expected = f"a</think>{D}b</think>{D}c</think>{D}d</think>{D}"
        for size in range(1, len(text) + 1):
            with subtests.test(size=size):
                assert feed_all("</think>", chunks(text, size)) == expected

    def test_each_thinking_block_is_handled(self):
        assert (
            feed_all(GEMMA_END, ["t1<channel|>", "a1", "t2<channel|>", "a2"])
            == f"t1<channel|>{D}a1t2<channel|>{D}a2"
        )

    def test_text_that_only_looks_like_the_tag_is_left_alone(self, subtests):
        for text in ("x<chan", "x<channel|", "<channel>", "channel|>", "plain text"):
            with subtests.test(text=text):
                assert feed_all(GEMMA_END, [text, " more"]) == text + " more"

    def test_an_empty_end_does_nothing(self):
        assert feed_all("", ["a", "b", "\n"]) == "ab\n"

    def test_a_single_character_end(self):
        assert feed_all("|", ["a|", "b|\n", "c"]) == f"a|{D}b|{D}c"


def draw(tokens, think_end=None):
    """Run display_assistant_stream on a token list; return (what was drawn, returned text)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        text = display_assistant_stream(iter(tokens), think_end=think_end)
    return out.getvalue(), text


class TestDisplayAssistantStream:
    TOKENS = ["<|channel>thought\n", "The answer is 144.", "<channel|>", "1", "4", "4"]

    def test_the_screen_gets_the_delimiter_but_the_stored_reply_does_not(self):
        drawn, text = draw(self.TOKENS, think_end=GEMMA_END)
        assert f"The answer is 144.<channel|>{D}144" in drawn
        assert text == "".join(self.TOKENS)  # byte for byte what the model produced
        assert "***" not in text

    def test_without_a_known_tag_nothing_changes(self):
        drawn, text = draw(self.TOKENS, think_end=None)
        assert "The answer is 144.<channel|>144" in drawn
        assert text == "".join(self.TOKENS)

    def test_a_model_that_already_writes_blank_lines_gets_one_on_each_side(self):
        tokens = ["<think>\n", "why", "</think>", "\n\n", "answer"]
        drawn, text = draw(tokens, think_end="</think>")
        assert f"why</think>{D}answer" in drawn  # the model's own newlines are absorbed
        assert text == "".join(tokens)

    def test_the_tag_arriving_in_pieces_is_still_separated(self):
        drawn, text = draw(["why", "</", "think", ">", "answer"], think_end="</think>")
        assert f"why</think>{D}answer" in drawn
        assert text == "why</think>answer"

    def test_the_delimiter_is_drawn_before_the_next_piece_arrives(self):
        out = io.StringIO()
        seen = []

        def gen():
            yield "why</think>"
            seen.append(out.getvalue())
            yield "answer"

        with contextlib.redirect_stdout(out):
            display_assistant_stream(gen(), think_end="</think>")
        assert seen[0].endswith(f"why</think>{D}")

    def test_an_interrupt_right_after_the_tag_keeps_the_delimiter_and_the_stored_text(self):
        # The delimiter ends in a newline, so the display loop draws it at once and
        # it survives the interrupt (which drops the last partial word). It is never
        # added to the stored reply, which only gains the " [interrupted]" marker.
        def gen():
            yield "thinking<channel|>"
            raise KeyboardInterrupt

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            text = display_assistant_stream(gen(), think_end=GEMMA_END)
        assert text == "thinking<channel|> [interrupted]"
        assert f"thinking<channel|>{D}" in out.getvalue()
        assert out.getvalue().endswith("\n")

    def test_a_long_reply_still_wraps(self):
        words = " ".join(["word"] * 60)
        drawn, text = draw([words, "<channel|>", "x"], think_end=GEMMA_END)
        assert text == words + "<channel|>x"
        assert f"<channel|>{D}x" in drawn


def separate_thinking(text, ends):
    """The reference rule ThinkSeparator streams: THINK_DELIMITER after each closing
    tag in `ends`, absorbing every "\\n" that directly follows the tag."""
    for end in ends:
        if end:
            text = re.sub(re.escape(end) + r"\n*", lambda m: end + D, text)
    return text


class TestSeparateThinking:
    def test_adds_the_delimiter_after_the_tag_and_absorbs_following_newlines(self):
        assert separate_thinking("why<channel|>144", [GEMMA_END]) == f"why<channel|>{D}144"
        assert separate_thinking("why</think>\nanswer", ["</think>"]) == f"why</think>{D}answer"
        assert separate_thinking("why</think>\n\nanswer", ["</think>"]) == f"why</think>{D}answer"
        assert separate_thinking("why</think>\r\nanswer", ["</think>"]) == f"why</think>{D}\r\nanswer"
        assert separate_thinking("why<channel|>", [GEMMA_END]) == f"why<channel|>{D}"

    def test_every_occurrence_and_every_known_end(self):
        text = "a<channel|>b</think>c<channel|>d"
        assert (
            separate_thinking(text, [GEMMA_END, "</think>"]) == f"a<channel|>{D}b</think>{D}c<channel|>{D}d"
        )

    def test_empty_ends_and_no_ends_change_nothing(self):
        assert separate_thinking("a<channel|>b", []) == "a<channel|>b"
        assert separate_thinking("a<channel|>b", [""]) == "a<channel|>b"

    def test_tags_with_regex_characters_are_matched_literally(self):
        assert separate_thinking("a[/THINK]b", ["[/THINK]"]) == f"a[/THINK]{D}b"
        assert separate_thinking("a<|end|>b", ["<|end|>"]) == f"a<|end|>{D}b"

    def test_streaming_and_static_agree_for_every_chunking(self, subtests):
        samples = [
            ("why<channel|>144", GEMMA_END),
            ("a</think>b</think>\nc</think>d</think>", "</think>"),
            (f"reason{GPTOSS_END}Hi", GPTOSS_END),
            ("[THINK]t[/THINK]answer", "[/THINK]"),
            ("no tag here at all", GEMMA_END),
            ("ends at the tag<channel|>", GEMMA_END),
            ("a</think>\n\nb</think></think>c", "</think>"),
            ("a|\nb|", "|"),
        ]
        for text, end in samples:
            expected = separate_thinking(text, [end])
            for size in range(1, len(text) + 1):
                with subtests.test(text=text, size=size):
                    assert feed_all(end, chunks(text, size)) == expected


def saved_conversation(tmp, thinking="why", content="The answer."):
    conv = Conversation()
    conv.add_user("question mentioning <channel|> literally")
    conv.add_assistant(content, model="/m/gemma.gguf", thinking=thinking)
    conv.save(tmp, name="saved", model="/m/gemma.gguf")


class TestCatSeparation:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
        self.state = State(model="m", config={**DEFAULTS, "conversations_dir": self.tmp}, context_length=None)

    def cat(self):
        with ui.console.capture() as cap:
            handle_command("/cat", "saved", None, Conversation(), self.state)
        return re.sub(r"\x1b\[[0-9;]*m", "", cap.get())

    def test_thinking_is_left_out_and_the_content_is_shown(self):
        saved_conversation(self.tmp, thinking="PRIVATE-REASONING")
        out = self.cat()
        assert "The answer." in out
        assert "PRIVATE-REASONING" not in out
        assert "***" not in out
        assert "question mentioning <channel|> literally" in out  # user text untouched

    def test_a_reply_that_was_only_thinking_is_skipped(self):
        saved_conversation(self.tmp, thinking="PRIVATE-REASONING", content="")
        out = self.cat()
        assert "PRIVATE-REASONING" not in out
        assert "Assistant:" not in out
        assert "You:" in out

    def test_a_reply_without_thinking_prints_as_its_content(self):
        saved_conversation(self.tmp, thinking=None, content="why<channel|>The answer.")
        out = self.cat()
        assert "why<channel|>The answer." in out
        assert "***" not in out

    def test_a_legacy_think_pairs_key_changes_nothing(self):
        saved_conversation(self.tmp, thinking=None, content="why<channel|>The answer.")
        path = Path(self.tmp) / "saved.json"
        data = json.loads(path.read_text())
        data["think_pairs"] = [["<|channel>thought", "<channel|>"]]
        path.write_text(json.dumps(data))
        out = self.cat()
        assert "why<channel|>The answer." in out
        assert "***" not in out

    def test_saved_text_that_looks_like_markup_prints_literally_instead_of_raising(self):
        conv = Conversation()
        conv.add_user("see [/path] please")
        conv.add_assistant("answer [/etc/hosts]", model="/m/x.gguf", thinking="[THINK]hmm[/THINK]")
        conv.save(self.tmp, name="saved", model="/m/x.gguf")
        out = self.cat()
        assert "see [/path] please" in out
        assert "answer [/etc/hosts]" in out
        assert "hmm" not in out

    def test_the_stored_file_never_contains_the_delimiter(self):
        saved_conversation(self.tmp)
        self.cat()
        stored = json.loads((Path(self.tmp) / "saved.json").read_text())["messages"][1]
        assert (stored["thinking"], stored["content"]) == ("why", "The answer.")


class TestThinkEnd:
    def state(self, tags):
        return State(model="m", config={**DEFAULTS}, context_length=None, think_tags=tags)

    def test_returns_the_active_end_tag(self):
        assert _think_end(self.state(("<|channel>thought", "<channel|>", "detected"))) == "<channel|>"

    def test_none_when_no_tags_are_known(self):
        assert _think_end(self.state(None)) is None


def test_the_delimiter_is_the_documented_literal():
    assert ui.THINK_DELIMITER == "\n\n*** END OF THINKING ***\n\n\n"
