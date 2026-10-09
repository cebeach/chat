"""Projects in the real chat.main(): switching, the notes in the prompt, /remember.

Uses run_session from test_turn_echo.py (a mock LlamaClient and scripted input), so the
REPL, the commands and the files on disk are the real ones.
"""

import json
from unittest import mock

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

import projects
from conversation import Conversation
from tests.test_turn_echo import FakeStream, run_session


def make_project(tmp_path, name="novel", notes="", prompt="", toml=""):
    path = projects.create_project(tmp_path / "projects", name)
    (path / "project.md").write_text(notes)
    (path / "system.md").write_text(prompt)
    if toml:
        (path / "project.toml").write_text(toml)
    return path


def reply():
    return FakeStream(["r"])


def session_in(tmp_path, inputs, **kwargs):
    return run_session(tmp_path, reply, inputs=inputs, **kwargs)


def sent(session, n=0):
    return session.chat_calls[n]["messages"]


def saved(directory):
    return sorted(p.name for p in directory.glob("*.json"))


class TestNotesInThePrompt:
    def test_notes_are_priced_and_sent(self, tmp_path):
        make_project(tmp_path, notes="- Anna is left-handed\n", prompt="Be brief.")
        session = session_in(tmp_path, ("hello", None), argv=("--project", "novel"))
        expected = "Be brief.\n\n# Project notes\n- Anna is left-handed\n"
        assert session.check_messages[0] == {"role": "system", "content": expected}
        assert sent(session)[0] == {"role": "system", "content": expected}

    def test_no_project_sends_no_notes(self, tmp_path):
        session = session_in(tmp_path, ("hello", None))
        assert [m["role"] for m in sent(session)] == ["user"]

    def test_an_edit_of_project_md_is_picked_up_on_the_next_send(self, tmp_path):
        path = make_project(tmp_path, notes="- old\n")

        def edit_during_the_first_reply():
            (path / "project.md").write_text("- new\n")
            return reply()

        session = run_session(
            tmp_path,
            edit_during_the_first_reply,
            inputs=("one", "two", None),
            argv=("--project", "novel"),
        )
        assert "- old" in sent(session, 0)[0]["content"]
        assert "- new" in sent(session, 1)[0]["content"]
        assert sum(1 for i in session.infos if i.startswith("project.md changed")) == 1

    def test_the_system_command_prices_the_notes_too(self, tmp_path):
        make_project(tmp_path, notes="- a fact\n")
        session = session_in(tmp_path, ("/system be brief", None), argv=("--project", "novel"))
        assert session.check_messages == [
            {"role": "system", "content": "be brief\n\n# Project notes\n- a fact\n"}
        ]

    def test_a_refused_prompt_with_notes_stores_nothing(self, tmp_path):
        make_project(tmp_path, notes="- a fact\n")
        session = session_in(tmp_path, ("hello", None), argv=("--project", "novel"), check_fit=(5000, 4096))
        assert session.conversation.messages == [] and "5,000 tokens" in session.errors[0]


class TestSystemPrompt:
    def test_system_md_is_the_projects_prompt(self, tmp_path):
        make_project(tmp_path, prompt="Be brief.")
        session = session_in(
            tmp_path, ("hello", None), argv=("--project", "novel"), config={"system_prompt": "G"}
        )
        assert sent(session)[0]["content"] == "Be brief."

    def test_an_empty_system_md_falls_back_to_the_global_prompt(self, tmp_path):
        make_project(tmp_path)
        session = session_in(
            tmp_path, ("hello", None), argv=("--project", "novel"), config={"system_prompt": "G"}
        )
        assert sent(session)[0]["content"] == "G"

    def test_load_inside_a_project_keeps_the_projects_prompt(self, tmp_path):
        path = make_project(tmp_path, prompt="Be brief.")
        old = Conversation(system_prompt="OLD PROMPT")
        old.add_user("earlier")
        old.save(path / "conversations", name="c")
        session = session_in(tmp_path, ("/load c", "hello", None), argv=("--project", "novel"))
        assert sent(session)[0]["content"] == "Be brief."
        assert [m["content"] for m in sent(session)[1:]] == ["earlier", "hello"]

    def test_load_without_a_project_still_takes_the_saved_prompt(self, tmp_path):
        old = Conversation(system_prompt="OLD PROMPT")
        old.add_user("earlier")
        old.save(tmp_path, name="c")
        session = session_in(tmp_path, ("/load c", "hello", None))
        assert sent(session)[0]["content"] == "OLD PROMPT"

    def test_a_stray_system_prompt_key_is_reported(self, tmp_path):
        make_project(tmp_path, toml='system_prompt = "x"\n')
        session = session_in(tmp_path, (None,), argv=("--project", "novel"))
        assert any("system_prompt is ignored" in i for i in session.infos)


class TestSwitching:
    def test_use_saves_the_old_conversation_then_the_new_one_in_the_project(self, tmp_path):
        path = make_project(tmp_path)
        session = session_in(tmp_path, ("one", "/project use novel", "two", None))
        assert len(saved(tmp_path)) == 1  # the conversation from before the switch
        assert len(saved(path / "conversations")) == 1
        first = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert [m["content"] for m in first["messages"]] == ["one", "r"]
        assert [m["content"] for m in session.conversation.messages] == ["two", "r"]

    def test_a_switch_saves_even_with_auto_save_off(self, tmp_path):
        make_project(tmp_path)
        session_in(tmp_path, ("one", "/project use novel", None), config={"auto_save": False})
        assert len(saved(tmp_path)) == 1

    def test_a_failed_save_aborts_the_switch(self, tmp_path):
        make_project(tmp_path, prompt="Be brief.")
        with mock.patch.object(Conversation, "save", side_effect=OSError("disk full")):
            session = session_in(tmp_path, ("one", "/project use novel", "two", None))
        assert any("Not switched" in e for e in session.errors)
        assert [m["content"] for m in session.conversation.messages] == ["one", "r", "two", "r"]
        assert sent(session, 1)[0]["content"] == "one"  # still the old, project-less conversation

    def test_a_bad_project_toml_changes_and_saves_nothing(self, tmp_path):
        make_project(tmp_path, toml="temperature = = 1")
        session = session_in(tmp_path, ("one", "/project use novel", None), config={"auto_save": False})
        assert any("project.toml" in e for e in session.errors)
        assert saved(tmp_path) == []  # a forced save would have written the file
        assert [m["content"] for m in session.conversation.messages] == ["one", "r"]

    def test_a_missing_project_is_an_error(self, tmp_path):
        session = session_in(tmp_path, ("/project use nope", None))
        assert any("No project named 'nope'" in e for e in session.errors)

    def test_settings_are_reset_on_a_switch_and_project_toml_applies(self, tmp_path):
        make_project(tmp_path, toml="temperature = 0.2\n")
        session = session_in(tmp_path, ("/set seed 5", "/project use novel", "hello", None))
        assert session.chat_calls[0]["options"] == {"temperature": 0.2}
        assert any("Session settings were reset" in i for i in session.infos)

    def test_leave_restores_the_global_settings_and_directory(self, tmp_path):
        path = make_project(tmp_path, prompt="Be brief.", toml="temperature = 0.2\n")
        session = session_in(
            tmp_path,
            ("/project use novel", "in project", "/project leave", "outside", None),
            config={"system_prompt": "G"},
        )
        assert sent(session, 0)[0]["content"] == "Be brief."
        assert sent(session, 1)[0]["content"] == "G" and session.chat_calls[1]["options"] == {}
        assert len(saved(path / "conversations")) == 1 and len(saved(tmp_path)) == 1

    def test_save_load_and_conversations_use_the_project_directory(self, tmp_path):
        path = make_project(tmp_path)
        session_in(tmp_path, ("/project use novel", "hello", "/save mine", None))
        assert "mine.json" in saved(path / "conversations")
        assert "mine.json" not in saved(tmp_path)

    def test_startup_with_a_missing_project_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exit_:
            session_in(tmp_path, (None,), argv=("--project", "nope"))
        assert exit_.value.code == 1

    def test_startup_with_a_bad_project_toml_exits(self, tmp_path):
        make_project(tmp_path, toml="temperature = = 1")
        with pytest.raises(SystemExit):
            session_in(tmp_path, (None,), argv=("--project", "novel"))

    def test_an_invalid_name_is_rejected_by_new_use_and_the_flag(self, tmp_path):
        session = session_in(tmp_path, ("/project new ../x", "/project use ../x", None))
        assert sum("Invalid project name" in e for e in session.errors) == 2
        assert not (tmp_path / "x").exists() and not (tmp_path / "projects" / ".." / "x").exists()
        with pytest.raises(SystemExit):
            session_in(tmp_path, (None,), argv=("--project", "../x"))

    def test_new_creates_the_project_and_refuses_a_duplicate(self, tmp_path):
        session = session_in(tmp_path, ("/project new novel", "/project new novel", None))
        assert (tmp_path / "projects" / "novel" / "system.md").is_file()
        assert any("already exists" in e for e in session.errors)

    def test_reload_and_leave_need_a_project(self, tmp_path):
        session = session_in(tmp_path, ("/project reload", "/project leave", None))
        assert sum("No project selected" in e for e in session.errors) == 2

    def test_reload_keeps_the_conversation_and_applies_the_new_files(self, tmp_path):
        path = make_project(tmp_path, notes="- old\n", prompt="Old prompt.")

        def edit_files():
            (path / "system.md").write_text("New prompt.")
            (path / "project.md").write_text("- new\n")
            return reply()

        session = run_session(
            tmp_path,
            edit_files,
            inputs=("one", "/project reload", "two", None),
            argv=("--project", "novel"),
        )
        second = sent(session, 1)
        assert second[0]["content"] == "New prompt.\n\n# Project notes\n- new\n"
        assert [m["content"] for m in second[1:]] == ["one", "r", "two"]

    def test_system_md_is_not_reread_without_a_reload(self, tmp_path):
        path = make_project(tmp_path, prompt="Old prompt.")

        def edit_prompt():
            (path / "system.md").write_text("New prompt.")
            return reply()

        session = run_session(tmp_path, edit_prompt, inputs=("one", "two", None), argv=("--project", "novel"))
        assert sent(session, 1)[0]["content"] == "Old prompt."

    def test_list_and_info_run(self, tmp_path, capsys):
        make_project(tmp_path, notes="- a\n", toml="temperature = 0.2\n")
        session = session_in(
            tmp_path,
            ("/project list", "/project", None),
            argv=("--project", "novel"),
            setup=lambda client: setattr(client.count_tokens, "return_value", 7),
        )
        out = capsys.readouterr().out
        assert "Projects" in out and "novel" in out and "yes" in out
        assert "project.md: 7 tokens" in session.infos
        assert "project.toml: temperature = 0.2" in session.infos


def remember_setup(new_text, cut_off=False, tokens_each=10):
    def setup(client):
        client.count_tokens.return_value = tokens_each
        client.complete.return_value = (new_text, cut_off, None)

    return setup


NOTES = "# Facts\n- Anna is left-handed\n- Ben is her brother\n"
MERGED = NOTES + "- Ben is a baker\n"


def remember(tmp_path, answers=("y",), new_text=MERGED, cut_off=False, **kwargs):
    path = make_project(tmp_path, notes=NOTES)
    session = run_session(
        tmp_path,
        reply,
        inputs=kwargs.pop("inputs", ("/remember Ben is a baker", None)),
        argv=("--project", "novel"),
        setup=kwargs.pop("setup", remember_setup(new_text, cut_off)),
        answers=answers,
        **kwargs,
    )
    return path, session


class TestRemember:
    def test_yes_writes_the_file_and_the_next_send_carries_it(self, tmp_path):
        path, session = remember(tmp_path, inputs=("/remember Ben is a baker", "hello", None))
        assert (path / "project.md").read_text() == MERGED
        assert "- Ben is a baker" in sent(session)[0]["content"]
        assert "project.md updated." in session.infos
        assert not any(i.startswith("project.md changed") for i in session.infos)

    def test_the_conversation_is_not_touched(self, tmp_path):
        _, session = remember(tmp_path)
        assert session.conversation.messages == [] and session.chat_calls == []

    def test_the_model_call_uses_only_temperature_zero(self, tmp_path):
        _, session = remember(tmp_path, inputs=("/set temperature 0.9", "/set seed 4", "/remember x", None))
        call = session.client.complete.call_args
        assert call.kwargs["options"] == {"temperature": 0} and call.kwargs["refreshed"] is True

    def test_the_diff_is_shown(self, tmp_path, capsys):
        remember(tmp_path)
        out = capsys.readouterr().out
        assert "+- Ben is a baker" in out and "project.md" in out

    @pytest.mark.parametrize("answer", ["n", "", "yes please", EOFError(), KeyboardInterrupt()])
    def test_anything_but_y_leaves_the_file(self, tmp_path, answer):
        path, _ = remember(tmp_path, answers=(answer,))
        assert (path / "project.md").read_text() == NOTES

    @pytest.mark.parametrize(
        "new_text, cut_off, message",
        [
            ("", False, "empty"),
            ("# Facts\n", False, "under half"),
            (MERGED[:-10], True, "cut off"),
        ],
    )
    def test_a_bad_reply_changes_nothing(self, tmp_path, new_text, cut_off, message):
        path, session = remember(tmp_path, new_text=new_text, cut_off=cut_off)
        assert (path / "project.md").read_text() == NOTES
        assert any(message in e for e in session.errors)

    def test_an_edit_during_the_decision_aborts_the_write(self, tmp_path):
        path = make_project(tmp_path, notes=NOTES)

        def edit_then_answer(*args, **kwargs):
            (path / "project.md").write_text("# Facts\n- edited elsewhere\n")
            return MERGED, False, None

        def setup(client):
            client.count_tokens.return_value = 10
            client.complete.side_effect = edit_then_answer

        session = run_session(
            tmp_path,
            reply,
            inputs=("/remember x", None),
            argv=("--project", "novel"),
            setup=setup,
            answers=("y",),
        )
        assert (path / "project.md").read_text() == "# Facts\n- edited elsewhere\n"
        assert any("changed while you were deciding" in e for e in session.errors)

    def test_an_unchanged_result_asks_nothing(self, tmp_path):
        path, session = remember(tmp_path, answers=(), new_text=NOTES)
        assert (path / "project.md").read_text() == NOTES
        assert any("No change" in i for i in session.infos)

    def test_a_code_fenced_reply_is_unwrapped(self, tmp_path):
        path, _ = remember(tmp_path, new_text=f"```markdown\n{MERGED}```")
        assert (path / "project.md").read_text() == MERGED

    def test_no_project_is_refused(self, tmp_path):
        session = run_session(tmp_path, reply, inputs=("/remember x", None))
        assert any("No project selected" in e for e in session.errors)
        session.client.complete.assert_not_called()

    def test_connection_loss_changes_nothing(self, tmp_path):
        def setup(client):
            client.count_tokens.return_value = 10
            client.complete.side_effect = RequestsConnectionError()

        path, session = remember(tmp_path, answers=(), setup=setup)
        assert (path / "project.md").read_text() == NOTES
        assert any("Lost connection" in e for e in session.errors)


class TestRememberRoom:
    @pytest.mark.parametrize(
        "check_fit, message",
        [
            ((100, None), "unknown"),
            ((4090, 4096), "Not remembered"),
            (RequestsConnectionError("down"), "could not check"),
        ],
    )
    def test_when_there_is_no_room_it_refuses_with_the_reason(self, tmp_path, check_fit, message):
        path, session = remember(tmp_path, answers=(), check_fit=check_fit)
        assert (path / "project.md").read_text() == NOTES
        assert any(message in e for e in session.errors)
        session.client.complete.assert_not_called()

    def test_without_room_for_the_thinking_allowance_it_still_goes_ahead(self, tmp_path):
        path, _ = remember(tmp_path, check_fit=(3000, 4096))  # reply 10 fits, +2000 does not
        assert (path / "project.md").read_text() == MERGED

    def test_it_checks_even_with_context_check_off(self, tmp_path):
        path, session = remember(
            tmp_path, answers=(), check_fit=(5000, 4096), config={"context_check": False}
        )
        assert any("Not remembered" in e for e in session.errors)
        session.client.complete.assert_not_called()

    def test_the_configured_reserve_is_not_added(self, tmp_path):
        path, _ = remember(tmp_path, check_fit=(3000, 4096), config={"reserve_output_tokens": 3000})
        assert (path / "project.md").read_text() == MERGED
