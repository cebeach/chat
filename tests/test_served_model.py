"""The model the REPL reports follows the server (see chat._sync_server_info).

The server decides which model runs: `state.model` and `state.context_length` are
re-read from /props, never chosen by the user. The old /model, /models, --model,
default_model and /load-overwrite paths are gone.
"""

import sys
import tempfile
import unittest
from unittest import mock

import chat
import ui
from chat import State, _near_context_limit, _reply_model, _sync_server_info, handle_command, parse_args
from config import DEFAULTS
from conversation import Conversation
from llama_client import LlamaChatStream, LlamaClient
from tests.test_think_tags import GEMMA_PAIR, QWEN_PAIR, devstral_server, gemma_server, qwen_server, render


def make_state(model="/m/gemma.gguf", n_ctx=262144, tags=(*GEMMA_PAIR, "detected"), **kw):
    config = {**DEFAULTS, "conversations_dir": "/nonexistent"}
    pairs = [tags[:2]] if tags else []
    return State(
        model=model, config=config, context_length=n_ctx, think_tags=tags, think_pairs_seen=pairs, **kw
    )


class SyncServerInfoTests(unittest.TestCase):
    def test_follows_the_server_and_announces_a_model_change_once(self):
        state = make_state()
        out = render(lambda: _sync_server_info(state, "/m/qwen.gguf", 131072))
        self.assertEqual((state.model, state.context_length), ("/m/qwen.gguf", 131072))
        self.assertIn("model: /m/gemma.gguf → /m/qwen.gguf (context 131,072 tokens)", out)
        self.assertEqual(render(lambda: _sync_server_info(state, "/m/qwen.gguf", 131072)), "")

    def test_no_announcement_at_startup_or_without_a_model_change(self):
        state = make_state()
        self.assertEqual(render(lambda: _sync_server_info(state, "/m/other.gguf", 4096, announce=False)), "")
        self.assertEqual(state.model, "/m/other.gguf")
        # same model, new context length (server restarted with another --ctx-size): updated silently
        self.assertEqual(render(lambda: _sync_server_info(state, "/m/other.gguf", 8192)), "")
        self.assertEqual(state.context_length, 8192)

    def test_unknown_values_never_overwrite_what_is_known(self):
        state = make_state()
        self.assertEqual(render(lambda: _sync_server_info(state, None, None)), "")
        self.assertEqual(render(lambda: _sync_server_info(state, "", None)), "")
        self.assertEqual((state.model, state.context_length), ("/m/gemma.gguf", 262144))

    def test_names_with_square_brackets_are_shown_intact_and_do_not_raise(self):
        state = make_state(model="[a] old")
        out = render(lambda: _sync_server_info(state, "[/THINK] new", 10))
        self.assertIn("model: [a] old → [/THINK] new (context 10 tokens)", out)


class ContextLimitTests(unittest.TestCase):
    def test_threshold_is_80_percent_of_the_current_context(self):
        self.assertFalse(_near_context_limit(209715, 262144))
        self.assertTrue(_near_context_limit(209716, 262144))
        self.assertFalse(_near_context_limit(0, 262144))

    def test_after_a_swap_to_a_smaller_context_the_same_prompt_now_warns(self):
        # The case the stale value got wrong: 150000 tokens is fine for 262144
        # but over the limit for 131072.
        self.assertFalse(_near_context_limit(150000, 262144))
        self.assertTrue(_near_context_limit(150000, 131072))

    def test_unknown_context_never_warns(self):
        self.assertFalse(_near_context_limit(10**9, None))
        self.assertFalse(_near_context_limit(10**9, 0))


class ConfigRefreshTests(unittest.TestCase):
    def test_config_refreshes_the_model_and_tags_and_announces_a_swap_once(self):
        server = qwen_server()
        server.n_ctx = 131072
        client = LlamaClient("http://llama.test")
        state = make_state()  # still describes Gemma
        with server.patched():
            out = render(lambda: handle_command("/config", "", client, Conversation(), state))
            again = render(lambda: handle_command("/config", "", client, Conversation(), state))
        self.assertEqual((state.model, state.context_length), ("/m/qwen.gguf", 131072))
        self.assertEqual(state.think_tags, (*QWEN_PAIR, "detected"))
        self.assertIn("model: /m/gemma.gguf → /m/qwen.gguf (context 131,072 tokens)", out)
        self.assertIn("thinking tags: <think> … </think> (detected)", out)
        self.assertIn("/m/qwen.gguf", out)
        self.assertNotIn("→", again)  # the swap was already announced
        self.assertNotIn("thinking tags:", again)
        self.assertEqual(state.think_pairs_seen, [GEMMA_PAIR, QWEN_PAIR])

    def test_the_toggle_warning_uses_the_refreshed_tags(self):
        # The state still holds a Gemma pair, but the server now serves a model
        # without thinking tags, so turning save_thinking off must warn.
        client = LlamaClient("http://llama.test")
        state = make_state()
        with devstral_server().patched(), mock.patch.object(chat, "display_info") as info:
            handle_command("/config", "save_thinking off", client, Conversation(), state)
        warning = info.call_args_list[-1].args[0]
        self.assertIn("will not be stripped", warning)

    def test_a_server_that_cannot_be_read_leaves_the_state_alone(self):
        server = gemma_server()
        client = LlamaClient("http://llama.test")
        state = make_state()
        server.fail = True
        with server.patched():
            out = render(lambda: handle_command("/config", "", client, Conversation(), state))
        self.assertEqual((state.model, state.context_length), ("/m/gemma.gguf", 262144))
        self.assertNotIn("→", out)


class RemovedLegacyTests(unittest.TestCase):
    def run_cmd(self, cmd, args=""):
        with mock.patch.object(chat, "display_error") as err:
            handle_command(cmd, args, None, Conversation(), make_state())
        return err

    def test_model_and_models_are_unknown_commands(self):
        for cmd in ("/model", "/models"):
            with self.subTest(cmd=cmd):
                err = self.run_cmd(cmd, "x")
                self.assertIn("Unknown command", err.call_args.args[0])

    def test_they_are_gone_from_completion_and_help(self):
        self.assertNotIn("/model", ui.COMMANDS)
        self.assertNotIn("/models", ui.COMMANDS)
        out = render(ui.print_help)
        self.assertNotIn("/model", out)
        self.assertNotIn("List available models", out)

    def test_the_model_flag_is_gone(self):
        for argv in (["chat.py", "--model", "x"], ["chat.py", "-m", "x"]):
            with self.subTest(argv=argv), mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()
        with mock.patch.object(sys, "argv", ["chat.py", "--url", "http://h:1"]):
            self.assertEqual(parse_args().url, "http://h:1")

    def test_default_model_is_no_longer_a_setting(self):
        self.assertNotIn("default_model", DEFAULTS)

    def test_main_exits_when_the_server_properties_cannot_be_read(self):
        fake = mock.Mock(server_model=None, server_n_ctx=None, think_tags=None)
        fake.is_available.return_value = True
        with (
            mock.patch.object(chat, "load_config", return_value=dict(DEFAULTS)),
            mock.patch.object(chat, "LlamaClient", return_value=fake),
            mock.patch.object(sys, "argv", ["chat.py"]),
            mock.patch.object(chat, "display_error") as err,
        ):
            with self.assertRaises(SystemExit) as cm:
                chat.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("server's properties", err.call_args.args[0])


class ReplyModelTests(unittest.TestCase):
    def test_the_servers_own_statement_wins(self):
        state = make_state(model="/m/refreshed.gguf")
        self.assertEqual(_reply_model(mock.Mock(model="/m/real.gguf"), state), "/m/real.gguf")

    def test_falls_back_to_the_refreshed_model_for_an_interrupted_stream(self):
        state = make_state(model="/m/refreshed.gguf")
        interrupted = LlamaChatStream(_FakeStreamResponse([b'data: {"content": "x", "stop": false}']))
        list(interrupted)
        self.assertIsNone(interrupted.model)
        self.assertEqual(_reply_model(interrupted, state), "/m/refreshed.gguf")


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)


class AttributionAcrossASwapTests(unittest.TestCase):
    def test_each_reply_records_the_model_that_wrote_it(self):
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
        self.assertEqual([m["model"] for m in assistants], ["/m/gemma.gguf", "/m/qwen.gguf"])
        self.assertTrue(all("model" not in m for m in conv.messages if m["role"] == "user"))
        self.assertEqual(state.model, "/m/qwen.gguf")
        self.assertEqual(state.context_length, 131072)
        # What is sent to the server never carries the extra key.
        self.assertTrue(all(set(m) == {"role", "content"} for m in conv.get_messages()))
        # The file-level model is now the model served at save time.
        with tempfile.TemporaryDirectory() as tmp:
            conv.save(tmp, name="swap", model=state.model)
            loaded, file_model = Conversation.load(tmp, "swap")
        self.assertEqual(file_model, "/m/qwen.gguf")
        self.assertEqual(
            [m.get("model") for m in loaded.messages], [None, "/m/gemma.gguf", None, "/m/qwen.gguf"]
        )


class PerMessageModelTests(unittest.TestCase):
    def test_the_model_key_sits_before_content_and_only_on_replies_that_have_one(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("a", model="m1")
        conv.add_assistant("old style")
        self.assertEqual(list(conv.messages[1]), ["role", "timestamp", "model", "content"])
        self.assertNotIn("model", conv.messages[0])
        self.assertNotIn("model", conv.messages[2])

    def test_save_load_round_trip_and_strip_think_preserve_the_model(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("<think>t</think>answer", model="m1")
        with tempfile.TemporaryDirectory() as tmp:
            conv.save(tmp, name="x", model="last", omit_think=True, think_pairs=[("<think>", "</think>")])
            loaded, _ = Conversation.load(tmp, "x")
        self.assertEqual(loaded.messages[1]["model"], "m1")
        self.assertEqual(loaded.messages[1]["content"], "answer")

    def test_recalled_copies_carry_no_model(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("a", model="m1")
        conv.recall(1)
        self.assertEqual(set(conv.messages[-1]), {"role", "content"})


class CatShowsTheModelTests(unittest.TestCase):
    def test_assistant_messages_show_their_model_and_old_ones_show_nothing(self):
        conv = Conversation()
        conv.add_user("q1")
        conv.add_assistant("a1", model="[/THINK] odd-name")  # brackets must not break Rich
        conv.add_user("q2")
        conv.add_assistant("a2")
        out = render(lambda: ui.display_cat_conversation("n", conv, "last"))
        self.assertIn("[/THINK] odd-name", out)
        self.assertEqual(out.count("odd-name"), 1)


class LoadDoesNotChooseTheModelTests(unittest.TestCase):
    def test_load_leaves_state_model_alone_and_reports_the_files_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            conv = Conversation()
            conv.add_user("hi")
            conv.save(tmp, name="old", model="some-other-model.gguf")
            state = make_state()
            state.config["conversations_dir"] = tmp
            target = Conversation()
            out = render(lambda: handle_command("/load", "old", None, target, state))
        self.assertEqual(state.model, "/m/gemma.gguf")
        self.assertIn("saved with model: some-other-model.gguf", out)
        self.assertEqual(len(target.messages), 1)

    def test_a_file_without_a_model_has_no_saved_with_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            conv = Conversation()
            conv.add_user("hi")
            conv.save(tmp, name="nomodel", model="")
            state = make_state()
            state.config["conversations_dir"] = tmp
            out = render(lambda: handle_command("/load", "nomodel", None, Conversation(), state))
        self.assertNotIn("saved with model", out)
        self.assertEqual(state.model, "/m/gemma.gguf")


if __name__ == "__main__":
    unittest.main()
