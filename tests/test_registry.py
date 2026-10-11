"""The command registry (commands.py): dispatch, help, and completion derive from one table."""

import readline
from unittest import mock

import pytest

import chat
import commands
import ui
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


def run(cmd, state):
    result = []
    out = render(lambda: result.append(handle_command(cmd, "", None, Conversation(), state)))
    return out, result[0]


class TestTable:
    def test_every_name_and_alias_is_offered(self):
        for command in commands.COMMANDS:
            for name in (command.name, *command.aliases):
                assert name in ui.COMMANDS

    def test_every_command_is_in_the_help_table(self):
        out = render(ui.print_help)
        for command in commands.COMMANDS:
            for usage, _ in command.rows:
                assert usage in out

    def test_help_is_sorted_and_keeps_the_multiline_row(self):
        names = [c.name for c in commands.COMMANDS]
        assert names == sorted(names)
        assert '"""' in render(ui.print_help)

    def test_every_handler_exists(self):
        commands.check_handlers(vars(chat))

    def test_a_missing_handler_is_reported(self):
        with pytest.raises(RuntimeError, match="_handle_cat"):
            commands.check_handlers({})


class TestDispatch:
    @pytest.mark.parametrize("cmd", ["/help", "/?"])
    def test_help_and_its_alias_print_the_table(self, cmd, state):
        out, keep_going = run(cmd, state)
        assert "Commands" in out and keep_going is True

    def test_every_command_reaches_its_handler(self, state):
        for command in commands.COMMANDS:
            for name in (command.name, *command.aliases):
                with mock.patch.object(chat, command.handler_name, return_value=None) as handler:
                    assert handle_command(name, "x", None, Conversation(), state) is True
                handler.assert_called_once()

    def test_only_false_stops_the_repl(self, state):
        with mock.patch.object(chat, "_handle_clear", return_value=False):
            assert handle_command("/clear", "", None, Conversation(), state) is False

    def test_an_unknown_command_names_help(self, state):
        out, keep_going = run("/nope", state)
        assert "Unknown command: /nope" in out and "/help" in out and keep_going is True


class TestCompletion:
    @pytest.fixture
    def complete(self, tmp_path):
        convs = tmp_path / "convs"
        convs.mkdir()
        (convs / "alpha.json").write_text("{}")
        projects = tmp_path / "projects"
        for name in ("novel", "notes"):
            (projects / name).mkdir(parents=True)
        with (
            mock.patch.object(ui, "HISTORY_FILE", tmp_path / "history"),
            mock.patch("ui.Conversation.list_saved", return_value=[("alpha", None), ("beta", None)]),
            mock.patch("ui.list_projects", return_value=["novel", "notes"]),
        ):
            ui.init_readline(lambda: convs, lambda: projects)
            completer = readline.get_completer()

            def candidates(line, text):
                found = []
                with mock.patch.object(readline, "get_line_buffer", return_value=line):
                    while (c := completer(text, len(found))) is not None:
                        found.append(c)
                return found

            yield candidates

    def test_command_names(self, complete):
        assert complete("/lo", "/lo") == ["/load"]
        assert "/help" in complete("/", "/") and "/?" in complete("/", "/")

    def test_conversation_names(self, complete):
        assert complete("/load ", "") == ["alpha", "beta"]
        assert complete("/cat b", "b") == ["beta"]

    def test_project_subcommands(self, complete):
        assert complete("/project ", "") == ["info", "list", "new", "use", "reload", "leave"]
        assert complete("/project u", "u") == ["use"]

    def test_project_names_after_use_and_new(self, complete):
        assert complete("/project use ", "") == ["novel", "notes"]
        assert complete("/project new no", "no") == ["novel", "notes"]

    def test_no_subcommands_once_a_second_word_started(self, complete):
        assert complete("/project list ", "") == []
