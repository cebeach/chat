"""@@<path> includes: files spliced into a prompt as flat text, with the file list kept as metadata."""

import json
import re

import pytest

import chat
import conv2txt
import ui
from chat import State, _prepare_user_message, build_message, expand_includes, handle_command
from config import DEFAULTS
from conversation import Conversation
from tests.helpers import render

FILES = {
    "a.txt": "ALPHA\n",
    "b.txt": "BETA",
    "my notes.txt": "NOTES",
    "empty.txt": "",
    "blank.txt": "  \n\n",
    "padded.txt": "\n\nL1\n\nL2\n\n",
    "nested.txt": "uses @@<a.txt> inside",
}


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    for name, content in FILES.items():
        (tmp_path / name).write_text(content)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def expand(text):
    return expand_includes(text)


def make_state(tmp_path, **config):
    return State(
        model="m",
        config={**DEFAULTS, "conversations_dir": str(tmp_path), **config},
        context_length=8192,
    )


class TestSplice:
    CASES = {
        "mid-text": ("Setting. @@<a.txt> Then act.", "Setting.\n\nALPHA\n\nThen act."),
        "at the start": ("@@<a.txt> then", "ALPHA\n\nthen"),
        "at the end": ("end @@<a.txt>", "end\n\nALPHA"),
        "whole message": ("@@<a.txt>", "ALPHA"),
        "two files": ("@@<a.txt> @@<b.txt>", "ALPHA\n\nBETA"),
        "punctuation after the closer stays after the separator": (
            "Read @@<a.txt>, then act",
            "Read\n\nALPHA\n\n, then act",
        ),
        "a path with spaces": ("@@<my notes.txt>", "NOTES"),
        "spaces inside the brackets are trimmed": ("@@< a.txt >", "ALPHA"),
        "a home-relative path": ("@@<~/a.txt>", "ALPHA"),
        "typed blank lines are kept, none added": ("Intro\n\n@@<a.txt>\n\nOutro", "Intro\n\nALPHA\n\nOutro"),
        "single newlines are raised to a blank line": ("Intro\n@@<a.txt>\nOutro", "Intro\n\nALPHA\n\nOutro"),
        "typed extra newlines are never removed": ("Intro\n\n\n@@<a.txt>", "Intro\n\n\nALPHA"),
        "the file's own edge newlines go, inner ones stay": ("@@<padded.txt>", "L1\n\nL2"),
        "an empty file in the middle still separates": ("X @@<empty.txt> Y", "X\n\nY"),
        "an empty file at the start adds nothing": ("@@<empty.txt> Y", "Y"),
        "an empty file at the end adds nothing": ("X @@<empty.txt>", "X"),
        "a whitespace-only file counts as empty": ("X @@<blank.txt> Y", "X\n\nY"),
        "@@ inside an included file is ordinary text": ("@@<nested.txt>", "uses @@<a.txt> inside"),
    }

    def test_flat_text(self, cwd, subtests):
        for name, (text, expected) in self.CASES.items():
            with subtests.test(name):
                flat, _, errors = expand(text)
                assert errors == []
                assert flat == expected

    def test_text_without_a_token_is_untouched(self, cwd, subtests):
        for text in [
            "@@count += 1",
            "@property\ndef f(): ...",
            'say @@"x"',
            "me@@example.com",
            "a @@ b",
            "x @< y",
        ]:
            with subtests.test(text):
                assert expand(text) == (text, [], [])

    def test_escape_writes_a_literal_token(self, cwd):
        flat, includes, errors = expand("write \\@@<a.txt> literally")
        assert (flat, includes, errors) == ("write @@<a.txt> literally", [], [])

    def test_escape_is_not_reported_as_unterminated(self, cwd):
        assert expand("\\@@<x") == ("@@<x", [], [])

    def test_escape_and_a_real_token_together(self, cwd):
        flat, includes, _ = expand("\\@@<a.txt> and @@<b.txt>")
        assert flat == "@@<a.txt> and\n\nBETA"
        assert [i["typed"] for i in includes] == ["b.txt"]

    def test_includes_describe_each_file(self, cwd):
        _, includes, _ = expand("@@<a.txt> @@<empty.txt> @@<my notes.txt>")
        assert includes == [
            {"typed": "a.txt", "path": str(cwd / "a.txt"), "bytes": 6},
            {"typed": "empty.txt", "path": str(cwd / "empty.txt"), "bytes": 0, "empty": True},
            {"typed": "my notes.txt", "path": str(cwd / "my notes.txt"), "bytes": 5},
        ]


class TestAborts:
    def test_a_missing_file_aborts_and_is_named(self, cwd):
        flat, includes, errors = expand("hello @@<nope.txt>")
        assert (flat, includes) == ("", [])
        assert errors == ["@@<nope.txt>: File not found."]

    def test_every_missing_file_is_reported(self, cwd):
        _, _, errors = expand("@@<one.txt> and @@<a.txt> and @@<two.txt>")
        assert [e.split(":")[0] for e in errors] == ["@@<one.txt>", "@@<two.txt>"]

    def test_a_file_over_the_read_guard_aborts(self, cwd, monkeypatch):
        """The fixed memory guard, not a context limit: that is the token check's job."""
        monkeypatch.setattr(chat, "_READ_FILE_MAX_BYTES", 1024)
        (cwd / "big.txt").write_text("x" * 2048)
        flat, _, errors = expand("@@<big.txt>")
        assert flat == ""
        assert errors == ["@@<big.txt>: File too large to read."]

    def test_a_file_far_over_the_old_32_kb_limit_is_read(self, cwd):
        (cwd / "long.txt").write_text("x" * 200_000)
        flat, _, errors = expand("@@<long.txt>")
        assert errors == [] and len(flat) == 200_000

    def test_an_unterminated_token_aborts(self, cwd):
        flat, _, errors = expand("see @@<a.txt and more")
        assert flat == ""
        assert len(errors) == 1 and errors[0].startswith("Unterminated include: @@<a.txt and more")

    def test_a_token_does_not_span_lines(self, cwd):
        _, _, errors = expand("@@<a.txt\nmore>")
        assert len(errors) == 1 and errors[0].startswith("Unterminated include")

    def test_an_empty_path_aborts(self, cwd):
        _, _, errors = expand("x @@<> y")
        assert errors == ["@@<>: no file name between the brackets."]

    def test_unterminated_and_missing_are_both_reported(self, cwd):
        _, _, errors = expand("@@<nope.txt> then @@<oops")
        assert len(errors) == 2

    def test_a_token_ends_at_the_first_closer(self, cwd):
        _, _, errors = expand("x @@<a and 3 > 2")
        assert errors == ["@@<a and 3>: File not found."]

    def test_a_message_with_nothing_left_aborts(self, cwd):
        flat, includes, errors = expand("@@<empty.txt>")
        assert (flat, includes) == ("", [])
        assert errors == ["Nothing to send: @@<empty.txt> is empty."]

    def test_an_empty_file_beside_text_is_allowed(self, cwd):
        assert expand("hi @@<empty.txt>")[2] == []


class TestPrepareUserMessage:
    def test_success_returns_flat_text_and_prints_one_line_per_file(self, cwd):
        state = make_state(cwd)
        out = render(lambda: self._run("A @@<a.txt> B @@<empty.txt>", state))
        assert self.result == ("A\n\nALPHA\n\nB", self.result[1])
        assert [i["typed"] for i in self.result[1]] == ["a.txt", "empty.txt"]
        assert "Included a.txt (6 B)" in out and "Included empty.txt (empty)" in out

    def test_failure_returns_none_and_prints_the_error(self, cwd):
        state = make_state(cwd)
        out = render(lambda: self._run("A @@<nope.txt>", state))
        assert self.result is None
        assert "@@<nope.txt>: File not found." in out

    def test_a_plain_message_prints_nothing(self, cwd):
        state = make_state(cwd)
        out = render(lambda: self._run("just text", state))
        assert self.result == ("just text", [])
        assert out == ""

    def _run(self, text, state):
        self.result = _prepare_user_message(text)


class TestReadCommand:
    def run(self, cwd, args):
        state = make_state(cwd)
        render(lambda: handle_command("/read", args, None, Conversation(), state))
        return state

    def test_joins_files_like_adjacent_tokens(self, cwd):
        state = self.run(cwd, "a.txt b.txt")
        assert state.retry_text == expand("@@<a.txt> @@<b.txt>")[0] == "ALPHA\n\nBETA"
        assert [i["typed"] for i in state.pending_includes] == ["a.txt", "b.txt"]

    def test_a_quoted_path_with_spaces(self, cwd):
        assert self.run(cwd, '"my notes.txt"').retry_text == "NOTES"

    def test_a_name_with_a_closing_bracket_is_reachable(self, cwd):
        (cwd / "odd>name.txt").write_text("ODD")
        assert self.run(cwd, "'odd>name.txt'").retry_text == "ODD"

    def test_one_missing_file_aborts_the_whole_command(self, cwd):
        state = self.run(cwd, "a.txt nope.txt")
        assert state.retry_text is None
        assert state.pending_includes == []

    def test_a_missing_file_is_named(self, cwd):
        state = make_state(cwd)
        out = render(lambda: handle_command("/read", "nope.txt", None, Conversation(), state))
        assert "nope.txt: File not found." in out

    def test_only_an_empty_file_aborts(self, cwd):
        assert self.run(cwd, "empty.txt").retry_text is None


class TestRetryCommand:
    INCLUDES = [{"typed": "a.txt", "path": "/x/a.txt", "bytes": 6}]

    def retry(self, cwd, conv):
        state = make_state(cwd)
        render(lambda: handle_command("/retry", "", None, conv, state))
        return state

    def test_carries_the_includes_and_the_flat_text(self, cwd):
        conv = Conversation()
        conv.add_user("Setting.\n\nALPHA", includes=self.INCLUDES)
        conv.add_assistant("reply")
        state = self.retry(cwd, conv)
        assert state.retry_text == "Setting.\n\nALPHA"
        assert state.pending_includes == self.INCLUDES
        assert conv.messages == []

    def test_does_not_expand_the_text_again(self, cwd):
        conv = Conversation()
        conv.add_user("kept literally: @@<a.txt>")
        conv.add_assistant("reply")
        state = self.retry(cwd, conv)
        assert state.retry_text == "kept literally: @@<a.txt>"
        assert state.pending_includes == []


class TestStorage:
    INCLUDES = [
        {"typed": "a.txt", "path": "/x/a.txt", "bytes": 2048},
        {"typed": "e.txt", "path": "/x/e.txt", "bytes": 0},
    ]

    def test_the_model_gets_role_and_content_only(self):
        conv = Conversation()
        conv.add_user("flat text", includes=self.INCLUDES)
        assert conv.get_messages() == [{"role": "user", "content": "flat text"}]

    def test_a_message_without_includes_has_no_key(self):
        conv = Conversation()
        conv.add_user("plain")
        assert "includes" not in conv.messages[0]

    def test_save_and_load_round_trip(self, tmp_path):
        conv = Conversation()
        conv.add_user("flat", includes=self.INCLUDES)
        conv.add_assistant("ok")
        conv.save(tmp_path, name="c")
        loaded, _ = Conversation.load(tmp_path, "c")
        assert loaded.messages[0]["includes"] == self.INCLUDES
        assert loaded.get_messages()[0] == {"role": "user", "content": "flat"}

    def test_an_old_file_without_includes_loads(self, tmp_path):
        data = {
            "model": "m",
            "system_prompt": "",
            "messages": [{"role": "user", "content": "old", "source_file": "/p"}],
        }
        (tmp_path / "old.json").write_text(json.dumps(data))
        loaded, _ = Conversation.load(tmp_path, "old")
        assert loaded.summary()["included_files"] == 0
        assert loaded.messages[0]["source_file"] == "/p"

    def test_summary_counts_included_files(self):
        conv = Conversation()
        conv.add_user("x", includes=self.INCLUDES)
        conv.add_user("y", includes=self.INCLUDES[:1])
        summary = conv.summary()
        assert (summary["included_files"], summary["included_bytes"]) == (3, 4096)


def table_text(callable_):
    """render() with the table borders removed, as in test_conversation_info.py."""
    return " ".join(re.sub(r"[│┃┏┓┡┩└┘━─┳╇┴]", " ", render(callable_)).split())


class TestDisplay:
    INCLUDES = [
        {"typed": "a.txt", "path": "/x/a.txt", "bytes": 2150},
        {"typed": "e.txt", "path": "/x/e.txt", "bytes": 0, "empty": True},
    ]

    def test_info_lists_the_included_files(self):
        conv = Conversation()
        conv.add_user("x", includes=self.INCLUDES)
        out = table_text(lambda: ui.display_conversation_info(conv.summary()))
        assert "Included files 2 (2.1 KB)" in out

    def test_info_without_includes_has_no_row(self):
        conv = Conversation()
        conv.add_user("x")
        assert "Included" not in table_text(lambda: ui.display_conversation_info(conv.summary()))

    def test_cat_shows_the_includes_line_under_the_message(self):
        conv = Conversation()
        conv.add_user("flat text", includes=self.INCLUDES)
        out = render(lambda: ui.display_cat_conversation("n", conv, "m"))
        assert "flat text [included: a.txt, 2.1 KB; e.txt, empty]" in out

    def test_conv2txt_shows_includes_and_still_shows_an_old_source_file(self):
        data = {
            "messages": [
                {"role": "user", "content": "new", "includes": self.INCLUDES},
                {"role": "user", "content": "old", "source_file": "/p/old.txt"},
            ]
        }
        text = conv2txt.convert(data)
        assert "[included: a.txt, 2.1 KB; e.txt, empty]" in text
        assert "[from: /p/old.txt]" in text


def test_build_message_keeps_the_order_of_files_only_segments(cwd):
    segments = [("file", "b.txt", "b.txt"), ("file", "a.txt", "a.txt")]
    flat, includes, errors = build_message(segments)
    assert (flat, errors) == ("BETA\n\nALPHA", [])
    assert [i["typed"] for i in includes] == ["b.txt", "a.txt"]
