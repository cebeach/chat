"""chat._duel_host(): every member reaches the current chat.<name> when it is used."""

from dataclasses import fields
from unittest import mock

import pytest

import chat

MEMBERS = {
    "auto_save": "_auto_save",
    "conversations_dir": "_conversations_dir",
    "fits_window": "_fits_window",
    "initial_duel": "_initial_duel",
    "prepare_user_message": "_prepare_user_message",
    "reply_model": "_reply_model",
    "sync_notes": "_sync_notes",
    "sync_server_info": "_sync_server_info",
    "update_think_tags": "_update_think_tags",
    "validated": "_validated",
    "display_error": "display_error",
    "display_info": "display_info",
    "display_ledger": "display_ledger",
    "display_stats": "display_stats",
    "display_assistant_stream": "display_assistant_stream",
    "echo_suppressed": "echo_suppressed",
    "get_multiline_input": "get_multiline_input",
}


def test_the_host_has_exactly_the_expected_members():
    assert {f.name for f in fields(chat.DuelHost)} == set(MEMBERS) | {"console"}


@pytest.mark.parametrize("member", sorted(MEMBERS))
def test_a_member_uses_the_chat_name_patched_after_the_host_was_built(member):
    host = chat._duel_host()
    with mock.patch.object(chat, MEMBERS[member], return_value="patched") as patched:
        assert getattr(host, member)(1, key=2) == "patched"
    patched.assert_called_once_with(1, key=2)


def test_the_console_is_the_current_chat_console():
    host = chat._duel_host()
    with mock.patch.object(chat.console, "print") as printed:
        host.console.print("x", markup=False)
    printed.assert_called_once_with("x", markup=False)
    replacement = mock.Mock()
    with mock.patch.object(chat, "console", replacement):
        host.console.print("y")
    replacement.print.assert_called_once_with("y")
