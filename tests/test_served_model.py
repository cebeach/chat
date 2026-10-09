"""The model the REPL reports follows the server (see chat._sync_server_info).

The server decides which model runs: `state.model` and `state.context_length` are
re-read from /props, never chosen by the user. The old /model, /models, --model,
default_model and /load-overwrite paths are gone.
"""

import sys
from unittest import mock

import pytest

import chat
import ui
from chat import State, _reply_model, _sync_server_info, handle_command, parse_args
from config import DEFAULTS
from conversation import Conversation
from llama_client import LlamaChatStream, LlamaClient
from tests.helpers import GEMMA_PAIR, QWEN_PAIR, devstral_server, gemma_server, qwen_server, render


def make_state(model="/m/gemma.gguf", n_ctx=262144, tags=(*GEMMA_PAIR, "detected"), **kw):
    config = {**DEFAULTS, "conversations_dir": "/nonexistent"}
    return State(model=model, config=config, context_length=n_ctx, think_tags=tags, **kw)


class TestSyncServerInfo:
    def test_follows_the_server_and_announces_a_model_change_once(self):
        state = make_state()
        out = render(lambda: _sync_server_info(state, "/m/qwen.gguf", 131072))
        assert (state.model, state.context_length) == ("/m/qwen.gguf", 131072)
        assert "model: /m/gemma.gguf → /m/qwen.gguf (context 131,072 tokens)" in out
        assert render(lambda: _sync_server_info(state, "/m/qwen.gguf", 131072)) == ""

    def test_no_announcement_at_startup_or_without_a_model_change(self):
        state = make_state()
        assert render(lambda: _sync_server_info(state, "/m/other.gguf", 4096, announce=False)) == ""
        assert state.model == "/m/other.gguf"
        # same model, new context length (server restarted with another --ctx-size): updated silently
        assert render(lambda: _sync_server_info(state, "/m/other.gguf", 8192)) == ""
        assert state.context_length == 8192

    def test_unknown_values_never_overwrite_what_is_known(self):
        state = make_state()
        assert render(lambda: _sync_server_info(state, None, None)) == ""
        assert render(lambda: _sync_server_info(state, "", None)) == ""
        assert (state.model, state.context_length) == ("/m/gemma.gguf", 262144)

    def test_names_with_square_brackets_are_shown_intact_and_do_not_raise(self):
        state = make_state(model="[a] old")
        out = render(lambda: _sync_server_info(state, "[/THINK] new", 10))
        assert "model: [a] old → [/THINK] new (context 10 tokens)" in out


class TestConfigRefresh:
    def test_config_refreshes_the_model_and_tags_and_announces_a_swap_once(self):
        server = qwen_server()
        server.n_ctx = 131072
        client = LlamaClient("http://llama.test")
        state = make_state()  # still describes Gemma
        with server.patched():
            out = render(lambda: handle_command("/config", "", client, Conversation(), state))
            again = render(lambda: handle_command("/config", "", client, Conversation(), state))
        assert (state.model, state.context_length) == ("/m/qwen.gguf", 131072)
        assert state.think_tags == (*QWEN_PAIR, "detected")
        assert "model: /m/gemma.gguf → /m/qwen.gguf (context 131,072 tokens)" in out
        assert "thinking tags: <think> … </think> (detected)" in out
        assert "/m/qwen.gguf" in out
        assert "→" not in again  # the swap was already announced
        assert "thinking tags:" not in again

    def test_the_toggle_warning_uses_the_refreshed_tags(self):
        # The state still holds a Gemma pair, but the server now serves a model
        # without thinking tags, so turning save_thinking off must warn.
        client = LlamaClient("http://llama.test")
        state = make_state()
        with devstral_server().patched(), mock.patch.object(chat, "display_info") as info:
            handle_command("/config", "save_thinking off", client, Conversation(), state)
        warning = info.call_args_list[-1].args[0]
        assert "saved whole" in warning

    def test_a_server_that_cannot_be_read_leaves_the_state_alone(self):
        server = gemma_server()
        client = LlamaClient("http://llama.test")
        state = make_state()
        server.fail = True
        with server.patched():
            out = render(lambda: handle_command("/config", "", client, Conversation(), state))
        assert (state.model, state.context_length) == ("/m/gemma.gguf", 262144)
        assert "→" not in out


class TestRemovedLegacy:
    def run_cmd(self, cmd, args=""):
        with mock.patch.object(chat, "display_error") as err:
            handle_command(cmd, args, None, Conversation(), make_state())
        return err

    def test_model_and_models_are_unknown_commands(self, subtests):
        for cmd in ("/model", "/models"):
            with subtests.test(cmd=cmd):
                err = self.run_cmd(cmd, "x")
                assert "Unknown command" in err.call_args.args[0]

    def test_they_are_gone_from_completion_and_help(self):
        assert "/model" not in ui.COMMANDS
        assert "/models" not in ui.COMMANDS
        out = render(ui.print_help)
        assert "/model" not in out
        assert "List available models" not in out

    def test_the_model_flag_is_gone(self, subtests):
        for argv in (["chat.py", "--model", "x"], ["chat.py", "-m", "x"]):
            with subtests.test(argv=argv), mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"):
                with pytest.raises(SystemExit):
                    parse_args()
        with mock.patch.object(sys, "argv", ["chat.py", "--url", "http://h:1"]):
            assert parse_args().url == "http://h:1"

    def test_default_model_is_no_longer_a_setting(self):
        assert "default_model" not in DEFAULTS

    def test_main_exits_when_the_server_properties_cannot_be_read(self):
        fake = mock.Mock(server_model=None, server_n_ctx=None, think_tags=None)
        fake.is_available.return_value = True
        with (
            mock.patch.object(chat, "load_config", return_value=dict(DEFAULTS)),
            mock.patch.object(chat, "LlamaClient", return_value=fake),
            mock.patch.object(sys, "argv", ["chat.py"]),
            mock.patch.object(chat, "display_error") as err,
        ):
            with pytest.raises(SystemExit) as cm:
                chat.main()
        assert cm.value.code == 1
        assert "server's properties" in err.call_args.args[0]


class TestReplyModel:
    def test_the_servers_own_statement_wins(self):
        state = make_state(model="/m/refreshed.gguf")
        assert _reply_model(mock.Mock(model="/m/real.gguf"), state) == "/m/real.gguf"

    def test_falls_back_to_the_refreshed_model_for_an_interrupted_stream(self):
        state = make_state(model="/m/refreshed.gguf")
        interrupted = LlamaChatStream(_FakeStreamResponse([b'data: {"content": "x", "stop": false}']))
        list(interrupted)
        assert interrupted.model is None
        assert _reply_model(interrupted, state) == "/m/refreshed.gguf"


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)


class TestAttributionAcrossASwap:
    def test_each_reply_records_the_model_that_wrote_it(self, tmp_path):
        server = gemma_server()
        server.n_ctx = 262144
        client = LlamaClient("http://llama.test")
        conv = Conversation()
        with server.patched():
            client.refresh()
            state = make_state(model=client.server_model, n_ctx=client.server_n_ctx)

            def turn(text):
                conv.add_user(text)
                stream = client.chat(state.model, conv.get_messages(), {})
                _sync_server_info(state, stream.server_model, stream.server_n_ctx, announce=False)
                reply = "".join(stream)
                conv.add_assistant(reply, model=_reply_model(stream, state))

            turn("one")
            server.become(
                "/m/qwen.gguf",
                qwen_server().source,
                qwen_server().cont,
                qwen_server().gen,
                "<real>",
                n_ctx=131072,
            )
            turn("two")
        assistants = [m for m in conv.messages if m["role"] == "assistant"]
        assert [m["model"] for m in assistants] == ["/m/gemma.gguf", "/m/qwen.gguf"]
        assert all("model" not in m for m in conv.messages if m["role"] == "user")
        assert state.model == "/m/qwen.gguf"
        assert state.context_length == 131072
        # What is sent to the server never carries the extra key.
        assert all(set(m) == {"role", "content"} for m in conv.get_messages())
        # The file-level model is now the model served at save time.
        tmp = str(tmp_path)
        conv.save(tmp, name="swap", model=state.model)
        loaded, file_model = Conversation.load(tmp, "swap")
        assert file_model == "/m/qwen.gguf"
        assert [m.get("model") for m in loaded.messages] == [None, "/m/gemma.gguf", None, "/m/qwen.gguf"]


class TestPerMessageModel:
    def test_the_model_key_sits_before_content_and_only_on_replies_that_have_one(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("a", model="m1")
        conv.add_assistant("old style")
        assert list(conv.messages[1]) == ["role", "timestamp", "model", "content"]
        assert "model" not in conv.messages[0]
        assert "model" not in conv.messages[2]

    def test_save_load_round_trip_and_omit_thinking_preserve_the_model(self, tmp_path):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("answer", model="m1", thinking="t")
        tmp = str(tmp_path)
        conv.save(tmp, name="x", model="last", omit_thinking=True)
        loaded, _ = Conversation.load(tmp, "x")
        assert loaded.messages[1]["model"] == "m1"
        assert loaded.messages[1]["content"] == "answer"
        assert "thinking" not in loaded.messages[1]

    def test_recalled_copies_carry_no_model(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("a", model="m1", thinking="t")
        conv.recall(1)
        assert set(conv.messages[-1]) == {"role", "content"}  # no model, no thinking


class TestCatShowsTheModel:
    def test_assistant_messages_show_their_model_and_old_ones_show_nothing(self):
        conv = Conversation()
        conv.add_user("q1")
        conv.add_assistant("a1", model="[/THINK] odd-name")  # brackets must not break Rich
        conv.add_user("q2")
        conv.add_assistant("a2")
        out = render(lambda: ui.display_cat_conversation("n", conv, "last"))
        assert "[/THINK] odd-name" in out
        assert out.count("odd-name") == 1


class TestLoadDoesNotChooseTheModel:
    def test_load_leaves_state_model_alone_and_reports_the_files_model(self, tmp_path):
        tmp = str(tmp_path)
        conv = Conversation()
        conv.add_user("hi")
        conv.save(tmp, name="old", model="some-other-model.gguf")
        state = make_state()
        state.config["conversations_dir"] = tmp
        target = Conversation()
        out = render(lambda: handle_command("/load", "old", None, target, state))
        assert state.model == "/m/gemma.gguf"
        assert "saved with model: some-other-model.gguf" in out
        assert len(target.messages) == 1

    def test_a_file_without_a_model_has_no_saved_with_note(self, tmp_path):
        tmp = str(tmp_path)
        conv = Conversation()
        conv.add_user("hi")
        conv.save(tmp, name="nomodel", model="")
        state = make_state()
        state.config["conversations_dir"] = tmp
        out = render(lambda: handle_command("/load", "nomodel", None, Conversation(), state))
        assert "saved with model" not in out
        assert state.model == "/m/gemma.gguf"
