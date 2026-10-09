"""projects.py (names, settings, notes files, diff, reply cleaning) and the notes in a Conversation."""

import os
from unittest import mock

import pytest

import projects
from config import DEFAULTS, apply_project
from conversation import Conversation
from llama_client import LlamaChatStream, LlamaClient
import tokens
from tests.helpers import FakeResponse, qwen_server, words_template


class TestNames:
    @pytest.mark.parametrize("name", ["novel", "my-novel_2", "A1"])
    def test_valid_names_are_kept(self, tmp_path, name):
        assert projects.project_dir(tmp_path, name) == tmp_path / name

    @pytest.mark.parametrize("name", ["", "my novel", "../x", "a/b", "café", ".hidden", None])
    def test_invalid_names_are_rejected_not_sanitized(self, tmp_path, name):
        with pytest.raises(ValueError, match="Invalid project name"):
            projects.project_dir(tmp_path, name)


class TestCreateAndList:
    def test_create_makes_the_layout(self, tmp_path):
        path = projects.create_project(tmp_path, "novel")
        assert (path / "project.md").read_text() == "" and (path / "system.md").read_text() == ""
        assert (path / "conversations").is_dir()

    def test_create_refuses_an_existing_project(self, tmp_path):
        projects.create_project(tmp_path, "novel")
        with pytest.raises(FileExistsError):
            projects.create_project(tmp_path, "novel")

    def test_list_is_newest_first_and_skips_invalid_names_and_files(self, tmp_path):
        old = projects.create_project(tmp_path, "old")
        projects.create_project(tmp_path, "new")
        (tmp_path / "bad name").mkdir()
        (tmp_path / "file.txt").write_text("x")
        os.utime(old, (1, 1))
        assert projects.list_projects(tmp_path) == ["new", "old"]

    def test_list_of_a_missing_directory_is_empty(self, tmp_path):
        assert projects.list_projects(tmp_path / "nope") == []


class TestSettings:
    def test_allowed_keys_are_kept_and_unknown_ignored(self, tmp_path):
        (tmp_path / "project.toml").write_text('temperature = 0.2\nseed = 3\nembedding_url = "x"\n')
        assert projects.load_settings(tmp_path) == ({"temperature": 0.2, "seed": 3}, [])

    def test_a_missing_file_is_empty(self, tmp_path):
        assert projects.load_settings(tmp_path) == ({}, [])

    def test_system_prompt_is_not_a_setting_and_is_reported(self, tmp_path):
        (tmp_path / "project.toml").write_text('system_prompt = "hi"\ntop_p = 0.9\n')
        assert projects.load_settings(tmp_path) == ({"top_p": 0.9}, ["system_prompt"])

    def test_a_malformed_file_names_itself(self, tmp_path):
        (tmp_path / "project.toml").write_text("temperature = = 1")
        with pytest.raises(ValueError, match="project.toml"):
            projects.load_settings(tmp_path)

    @pytest.mark.parametrize(
        "line", ['context_check = "yes"', "reserve_output_tokens = -1", "reserve_output_tokens = true"]
    )
    def test_a_wrong_type_is_an_error(self, tmp_path, line):
        (tmp_path / "project.toml").write_text(line + "\n")
        with pytest.raises(ValueError, match="project.toml"):
            projects.load_settings(tmp_path)

    def test_apply_project_lays_settings_over_a_copy(self):
        base = {**DEFAULTS, "temperature": 1.0}
        merged = apply_project(base, {"temperature": 0.1})
        assert merged["temperature"] == 0.1 and base["temperature"] == 1.0


class TestFiles:
    def test_missing_files_read_as_empty(self, tmp_path):
        assert projects.read_notes(tmp_path) == "" and projects.read_system_prompt(tmp_path) == ""

    def test_a_blank_system_md_is_no_prompt(self, tmp_path):
        (tmp_path / "system.md").write_text("  \n")
        assert projects.read_system_prompt(tmp_path) == ""

    def test_system_md_is_read_as_written(self, tmp_path):
        (tmp_path / "system.md").write_text("Be brief.\n")
        assert projects.read_system_prompt(tmp_path) == "Be brief.\n"

    def test_write_notes_replaces_the_file_and_leaves_no_temp(self, tmp_path):
        (tmp_path / "project.md").write_text("old")
        projects.write_notes(tmp_path, "new")
        assert (tmp_path / "project.md").read_text() == "new"
        assert [p.name for p in tmp_path.iterdir()] == ["project.md"]

    def test_a_failed_replace_leaves_the_old_file_and_no_temp(self, tmp_path):
        (tmp_path / "project.md").write_text("old")
        with mock.patch("projects.os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                projects.write_notes(tmp_path, "new")
        assert (tmp_path / "project.md").read_text() == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["project.md"]

    def test_digest_tracks_content(self):
        assert projects.notes_digest("a") == projects.notes_digest("a") != projects.notes_digest("b")


class TestDiff:
    def test_equal_texts_have_no_diff(self):
        assert projects.diff_text("a\n", "a\n") == ""

    def test_a_change_is_a_unified_diff(self):
        diff = projects.diff_text("a\nb\n", "a\nc\n")
        assert "--- project.md" in diff and "-b\n" in diff and "+c\n" in diff

    def test_a_last_line_without_newline_still_ends_in_one(self):
        assert projects.diff_text("a", "b").endswith("\n")


class TestCleanReply:
    def test_a_plain_file_gets_one_trailing_newline(self):
        assert projects.clean_reply("# Notes\n- a\n\n", None) == "# Notes\n- a\n"

    def test_a_surrounding_code_fence_is_removed(self):
        assert projects.clean_reply("```markdown\n# N\n- a\n```", None) == "# N\n- a\n"

    def test_thinking_is_removed(self):
        reply = "<think>hmm</think>\n# N\n"
        assert projects.clean_reply(reply, ("<think>", "</think>", "detected")) == "# N\n"

    def test_an_empty_reply_stays_empty(self):
        assert projects.clean_reply("  \n", None) == ""
        assert projects.clean_reply("<think>x</think>", ("<think>", "</think>", "d")) == ""

    def test_the_shrink_guard(self):
        old = "x" * 100
        assert projects.shrank_too_much(old, "x" * 49)
        assert not projects.shrank_too_much(old, "x" * 50)
        assert not projects.shrank_too_much("", "")


class TestRememberMessages:
    def test_the_prompt_carries_the_file_and_the_note(self):
        system, user = projects.remember_messages("# N\n- a\n", "b is true")
        assert system["role"] == "system" and user["role"] == "user"
        assert "# N\n- a\n" in user["content"] and "b is true" in user["content"]


class TestNotesInTheConversation:
    def test_notes_follow_the_system_prompt_in_one_system_message(self):
        conv = Conversation(system_prompt="Be brief.")
        conv.project_notes = "- a\n"
        conv.add_user("hi")
        msgs = conv.get_messages()
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == "Be brief.\n\n# Project notes\n- a\n"

    def test_notes_without_a_prompt_have_no_leading_blank_line(self):
        conv = Conversation()
        conv.project_notes = "- a"
        assert conv.get_messages() == [{"role": "system", "content": "# Project notes\n- a"}]

    def test_no_notes_is_unchanged(self):
        assert Conversation(system_prompt="p").get_messages() == [{"role": "system", "content": "p"}]
        assert Conversation().get_messages() == []

    def test_notes_are_not_saved_and_survive_clear(self, tmp_path):
        import json

        conv = Conversation(system_prompt="p")
        conv.project_notes = "secret notes"
        conv.add_user("hi")
        path = conv.save(tmp_path, name="c")
        assert "secret notes" not in open(path).read()
        assert json.load(open(path))["system_prompt"] == "p"
        conv.clear()
        assert conv.project_notes == "secret notes"


class TestTokens:
    @pytest.fixture
    def client(self):
        server = qwen_server()
        server.render = words_template
        with server.patched():
            yield LlamaClient("http://llama.test")

    def test_notes_are_counted_apart_from_the_system_prompt(self, client):
        conv = Conversation(system_prompt="be brief")
        conv.project_notes = "one two three four"
        conv.add_user("hello there")
        counts = tokens.breakdown(client, conv)
        assert counts["system"] == 2 and counts["notes"] == 4 and counts["user"] == 2
        assert counts["overhead"] == max(0, counts["total"] - 2 - 4 - 2 - counts["assistant"])

    def test_notes_alone_give_a_breakdown_for_an_empty_conversation(self, client):
        conv = Conversation()
        conv.project_notes = "one two"
        assert tokens.breakdown(client, conv)["notes"] == 2

    def test_notes_heavy_boundary(self):
        assert not tokens.notes_heavy(1024, 4096)
        assert tokens.notes_heavy(1025, 4096)
        assert not tokens.notes_heavy(10**6, None)


def sse(*events):
    import json

    return [f"data: {json.dumps(e)}".encode() for e in events]


class TestComplete:
    def make(self, lines):
        client = LlamaClient("http://x")
        stream = LlamaChatStream(FakeResponse(lines=lines))
        stream.think_tags = ("<t>", "</t>", "detected")
        client.chat = mock.Mock(return_value=stream)
        return client

    def test_a_finished_reply(self):
        client = self.make(sse({"content": "ab"}, {"content": "c", "stop": True, "tokens_predicted": 3}))
        assert client.complete("m", [], {"temperature": 0}) == ("abc", False, ("<t>", "</t>", "detected"))
        client.chat.assert_called_once_with("m", [], options={"temperature": 0}, refreshed=False)

    @pytest.mark.parametrize("final", [{"truncated": True}, {"stop_type": "limit"}])
    def test_a_cut_off_reply_is_flagged(self, final):
        client = self.make(sse({"content": "ab"}, {"content": "", "stop": True, **final}))
        assert client.complete("m", [])[1] is True

    def test_a_stream_that_ends_without_a_final_chunk_is_cut_off(self):
        client = self.make(sse({"content": "ab"}))
        assert client.complete("m", [])[1] is True
