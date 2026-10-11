"""Slash commands driven through handle_command: the branches no other file pins."""

from unittest import mock

import pytest

import chat
from chat import State, handle_command
from config import DEFAULTS
from conversation import Conversation
from tests.helpers import render


@pytest.fixture
def state(tmp_path):
    return State(
        model="/m/qwen.gguf",
        config={**DEFAULTS, "conversations_dir": str(tmp_path / "convs")},
        context_length=4096,
    )


def run(cmd, args, conv, state, client=None):
    """(output, handle_command's return value)."""
    result = []
    out = render(lambda: result.append(handle_command(cmd, args, client, conv, state)))
    return out, result[0]


def exchange(conv, n=2):
    for i in range(1, n + 1):
        conv.add_user(f"question {i}")
        conv.add_assistant(f"answer {i}")


class TestSimpleCommands:
    def test_help_lists_the_commands_and_continues(self, state):
        out, keep_going = run("/?", "", Conversation(), state)
        assert "Commands" in out and "/exit" in out and keep_going is True

    def test_exit_says_goodbye_and_stops_the_repl(self, state):
        out, keep_going = run("/exit", "", Conversation(), state)
        assert "Goodbye!" in out and keep_going is False

    def test_exit_auto_saves_a_conversation_with_messages(self, state, tmp_path):
        conv = Conversation()
        exchange(conv, 1)
        state.auto_save_name = "auto"
        run("/exit", "", conv, state)
        assert [n for n, _ in Conversation.list_saved(tmp_path / "convs")] == ["auto"]

    def test_exit_does_not_save_when_auto_save_is_off(self, state, tmp_path):
        state.config["auto_save"] = False
        conv = Conversation()
        exchange(conv, 1)
        run("/exit", "", conv, state)
        assert Conversation.list_saved(tmp_path / "convs") == []

    def test_clear_empties_the_conversation(self, state):
        conv = Conversation()
        exchange(conv)
        out, keep_going = run("/clear", "", conv, state)
        assert conv.messages == [] and "Conversation cleared." in out and keep_going is True

    def test_stats_toggles_the_stats_line(self, state):
        assert state.show_stats is True
        out, _ = run("/stats", "", Conversation(), state)
        assert state.show_stats is False and "Stats display: off" in out
        out, _ = run("/stats", "", Conversation(), state)
        assert state.show_stats is True and "Stats display: on" in out

    def test_conversations_with_none_saved(self, state):
        out, keep_going = run("/conversations", "", Conversation(), state)
        assert "No saved conversations." in out and keep_going is True

    def test_conversations_lists_the_saved_names(self, state, tmp_path):
        conv = Conversation()
        exchange(conv, 1)
        conv.save(str(tmp_path / "convs"), name="alpha")
        out, _ = run("/conversations", "", Conversation(), state)
        assert "alpha" in out

    def test_an_unknown_command_is_reported_and_the_repl_continues(self, state):
        out, keep_going = run("/nope", "", Conversation(), state)
        assert "Unknown command: /nope" in out and keep_going is True


class TestSave:
    def test_save_writes_the_named_file(self, state, tmp_path):
        conv = Conversation()
        exchange(conv, 1)
        out, _ = run("/save", "mine", conv, state)
        assert "Conversation saved" in out
        assert [n for n, _ in Conversation.list_saved(tmp_path / "convs")] == ["mine"]

    def test_a_failed_write_is_reported(self, state):
        conv = Conversation()
        exchange(conv, 1)
        with mock.patch.object(Conversation, "save", side_effect=OSError("disk full")):
            out, keep_going = run("/save", "mine", conv, state)
        assert "Failed to save: disk full" in out and keep_going is True


class TestLoadAndCat:
    @pytest.fixture
    def saved(self, state, tmp_path):
        conv = Conversation(system_prompt="saved prompt")
        exchange(conv, 1)
        conv.save(str(tmp_path / "convs"), name="old")

    @pytest.mark.parametrize("cmd", ["/load", "/cat"])
    def test_no_name_is_a_usage_error(self, cmd, state):
        out, keep_going = run(cmd, "  ", Conversation(), state)
        assert f"Usage: {cmd} <name>" in out and keep_going is True

    @pytest.mark.parametrize("cmd", ["/load", "/cat"])
    def test_a_missing_conversation_is_reported(self, cmd, state):
        out, _ = run(cmd, "ghost", Conversation(), state)
        assert "No saved conversation named 'ghost'." in out

    def test_load_reports_any_other_failure(self, state, saved):
        with mock.patch.object(Conversation, "load", side_effect=ValueError("bad json")):
            out, _ = run("/load", "old", Conversation(), state)
        assert "Failed to load: bad json" in out

    def test_cat_reports_any_other_failure(self, state, saved):
        with mock.patch.object(Conversation, "load", side_effect=ValueError("bad json")):
            out, _ = run("/cat", "old", Conversation(), state)
        assert "Failed to read conversation: bad json" in out

    def test_cat_prints_the_messages_and_leaves_the_conversation_alone(self, state, saved):
        conv = Conversation(system_prompt="current")
        out, _ = run("/cat", "old", conv, state)
        assert "question 1" in out and "answer 1" in out
        assert conv.messages == [] and conv.system_prompt == "current"

    def test_load_uses_the_projects_conversation_directory(self, state, tmp_path):
        project_dir = tmp_path / "proj"
        (project_dir / "conversations").mkdir(parents=True)
        saved = Conversation()
        exchange(saved, 1)
        saved.save(str(project_dir / "conversations"), name="inproj")
        state.project_dir = project_dir
        conv = Conversation()
        out, _ = run("/load", "inproj", conv, state)
        assert "Loaded conversation: inproj (2 messages" in out and len(conv.messages) == 2


class TestRecall:
    def test_no_argument_is_a_usage_error(self, state):
        out, keep_going = run("/recall", "", Conversation(), state)
        assert "Usage: /recall <pair_number>" in out and keep_going is True

    def test_a_non_integer_is_refused(self, state):
        conv = Conversation()
        exchange(conv)
        out, keep_going = run("/recall", "one", conv, state)
        assert "Pair number must be an integer." in out and keep_going is True
        assert len(conv.messages) == 4

    def test_an_out_of_range_pair_is_reported(self, state):
        conv = Conversation()
        exchange(conv)
        out, _ = run("/recall", "9", conv, state)
        assert "Pair 9 out of range (1-2)" in out and len(conv.messages) == 4

    def test_a_pair_is_reinjected_at_the_end(self, state):
        conv = Conversation()
        exchange(conv)
        out, _ = run("/recall", "1", conv, state)
        assert "Recalled pair 1 into context." in out
        assert [m["content"] for m in conv.messages[-2:]] == ["question 1", "answer 1"]
        assert len(conv.messages) == 7  # four, the context note and the pair


class TestRetry:
    def test_nothing_to_retry(self, state):
        conv = Conversation()
        conv.add_user("lonely")
        out, keep_going = run("/retry", "", conv, state)
        assert "Nothing to retry" in out and keep_going is True
        assert state.retry_text is None and len(conv.messages) == 1

    def test_the_last_message_must_be_the_assistants(self, state):
        conv = Conversation()
        conv.add_user("one")
        conv.add_assistant("two")
        conv.add_user("three")
        out, _ = run("/retry", "", conv, state)
        assert "Last message is not an assistant response." in out
        assert state.retry_text is None and len(conv.messages) == 3

    def test_the_last_exchange_is_removed_and_queued_for_resend(self, state):
        conv = Conversation()
        exchange(conv)
        out, keep_going = run("/retry", "", conv, state)
        assert keep_going is True
        assert state.retry_text == "question 2" and state.pending_includes == []
        assert [m["content"] for m in conv.messages] == ["question 1", "answer 1"]


class TestRead:
    def test_no_path_is_a_usage_error(self, state):
        out, keep_going = run("/read", "", Conversation(), state)
        assert "Usage: /read <path> [<path> ...]" in out and keep_going is True
        assert state.retry_text is None

    def test_an_unbalanced_quote_is_reported(self, state):
        out, keep_going = run("/read", '"unclosed', Conversation(), state)
        assert "Could not parse paths:" in out and keep_going is True
        assert state.retry_text is None

    def test_a_missing_file_aborts_the_whole_command(self, state, tmp_path):
        good = tmp_path / "good.txt"
        good.write_text("fine")
        out, _ = run("/read", f"{good} {tmp_path / 'gone.txt'}", Conversation(), state)
        assert "gone.txt" in out
        assert state.retry_text is None and state.pending_includes == []

    def test_several_files_are_spliced_and_queued(self, state, tmp_path):
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text("alpha text")
        b.write_text("beta text")
        out, _ = run("/read", f"{a} {b}", Conversation(), state)
        assert "User:" in out and "alpha text" in out and "beta text" in out
        assert "alpha text" in state.retry_text and "beta text" in state.retry_text
        assert len(state.pending_includes) == 2


class TestSystem:
    """The prompt-fits check has its own tests (test_tokens.py); here it always passes."""

    @pytest.fixture(autouse=True)
    def _fits(self):
        with mock.patch.object(chat, "_system_prompt_fits", return_value=True):
            yield

    def test_no_argument_shows_the_current_prompt(self, state):
        out, keep_going = run("/system", "", Conversation(system_prompt="be brief"), state)
        assert "Current system prompt: be brief" in out and keep_going is True

    def test_no_argument_with_no_prompt(self, state):
        out, _ = run("/system", "", Conversation(), state)
        assert "Current system prompt: (none)" in out

    def test_triple_quotes_read_a_multiline_prompt(self, state):
        conv = Conversation()
        with mock.patch.object(chat, "get_multiline_input", return_value="line one\nline two"):
            out, _ = run("/system", '"""', conv, state)
        assert conv.system_prompt == "line one\nline two" and "System prompt set." in out

    def test_a_cancelled_multiline_prompt_changes_nothing(self, state):
        conv = Conversation(system_prompt="old")
        with mock.patch.object(chat, "get_multiline_input", return_value=None):
            run("/system", '"""', conv, state)
        assert conv.system_prompt == "old"

    def test_a_file_inside_the_cwd_is_read_and_remembered(self, state, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "sys.txt").write_text("from a file")
        conv = Conversation()
        out, _ = run("/system", "sys.txt", conv, state)
        assert conv.system_prompt == "from a file" and "System prompt set from file." in out
        assert conv.source_file == str((tmp_path / "sys.txt").resolve())

    def test_a_file_in_a_subdirectory_of_the_cwd_is_read(self, state, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "sys.txt").write_text("nested")
        conv = Conversation()
        run("/system", "sub/sys.txt", conv, state)
        assert conv.system_prompt == "nested"

    def test_a_file_outside_the_cwd_is_taken_as_plain_text(self, state, tmp_path, monkeypatch):
        cwd = tmp_path / "work"
        cwd.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret contents")
        monkeypatch.chdir(cwd)
        conv = Conversation()
        conv.source_file = "stale"
        run("/system", str(outside), conv, state)
        assert conv.system_prompt == str(outside) and conv.source_file is None

    def test_a_directory_is_taken_as_plain_text(self, state, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        conv = Conversation()
        run("/system", ".", conv, state)
        assert conv.system_prompt == "."

    def test_a_file_that_cannot_be_read_is_reported(self, state, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "sys.txt").write_text("x")
        conv = Conversation(system_prompt="old")
        with mock.patch.object(chat, "_read_file", return_value=(False, "Cannot read sys.txt")):
            out, _ = run("/system", "sys.txt", conv, state)
        assert "Cannot read sys.txt" in out and conv.system_prompt == "old"

    def test_plain_text_clears_a_stale_source_file(self, state):
        conv = Conversation()
        conv.source_file = "stale"
        out, _ = run("/system", "be brief", conv, state)
        assert conv.system_prompt == "be brief" and conv.source_file is None
        assert "System prompt set." in out
