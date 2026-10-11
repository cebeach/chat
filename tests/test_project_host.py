"""chat._project_host(): every member reaches the current chat.<name> when it is used."""

from dataclasses import fields
from unittest import mock

import pytest

import chat

MEMBERS = {
    "auto_save": "_auto_save",
    "initial_duel": "_initial_duel",
    "initial_options": "_initial_options",
    "projects_dir": "_projects_dir",
    "display_error": "display_error",
    "display_info": "display_info",
    "display_notes_heavy": "display_notes_heavy",
    "display_projects": "display_projects",
    "display_warning": "display_warning",
    "echo_suppressed": "echo_suppressed",
}


def test_the_host_has_exactly_the_expected_members():
    assert {f.name for f in fields(chat.ProjectHost)} == set(MEMBERS) | {"console"}


@pytest.mark.parametrize("member", sorted(MEMBERS))
def test_a_member_uses_the_chat_name_patched_after_the_host_was_built(member):
    host = chat._project_host()
    with mock.patch.object(chat, MEMBERS[member], return_value="patched") as patched:
        assert getattr(host, member)(1, key=2) == "patched"
    patched.assert_called_once_with(1, key=2)


def test_the_console_is_the_current_chat_console():
    host = chat._project_host()
    replacement = mock.Mock()
    with mock.patch.object(chat, "console", replacement):
        host.console.print("y")
    replacement.print.assert_called_once_with("y")


@pytest.mark.parametrize("name", ["_reset_session", "_switch_project", "_sync_notes"])
def test_the_moved_entry_points_stay_importable_from_chat(name):
    assert callable(getattr(chat, name))
