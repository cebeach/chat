"""/load applies the whole saved conversation against the currently served model.

The model names recorded in the file (file-level and per message) are information
only: they never select a model, never reach the server, and the saved system
prompt's source file is applied along with the prompt itself.
"""

import json
import tempfile
import unittest
from pathlib import Path

from chat import State, _reply_model, _sync_server_info, handle_command
from config import DEFAULTS
from conversation import Conversation
from llama_client import LlamaClient
from tests.test_think_tags import qwen_server, render

RECORDED = ["/m/a-model.gguf", "/m/b-model.gguf", "/m/file-level-model.gguf"]


def save_old_conversation(tmp, source_file="/home/someone/saved_prompt.txt"):
    """A conversation saved by a session that ran two other models."""
    saved = Conversation(system_prompt="SAVED SYSTEM PROMPT")
    saved.source_file = source_file
    saved.add_user("question one")
    saved.add_assistant("answer one from a", model=RECORDED[0])
    saved.add_user("question two")
    saved.add_assistant("answer two from b", model=RECORDED[1])
    saved.save(tmp, name="old", model=RECORDED[2])


class LoadAppliesAgainstTheCurrentModelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        save_old_conversation(self.tmp)
        self.server = qwen_server()  # serves /m/qwen.gguf
        self.bodies = []
        orig_post = self.server.post

        def spy_post(url, json=None, **kw):
            self.bodies.append((url.rsplit("/", 1)[-1], json))
            return orig_post(url, json=json, **kw)

        self.server.post = spy_post  # installed before patched() binds it
        self.client = LlamaClient("http://llama.test")
        self.conv = Conversation(system_prompt="CURRENT SYSTEM PROMPT")
        self.conv.source_file = "/home/someone/current_prompt.txt"
        self.conv.add_user("typed before /load")

    def start(self):
        self.client.refresh()
        self.state = State(
            model=self.client.server_model,
            config={**DEFAULTS, "conversations_dir": self.tmp},
            context_length=self.client.server_n_ctx,
            think_tags=self.client.think_tags,
            think_pairs_seen=[("<think>", "</think>")],
        )

    def load(self):
        return render(lambda: handle_command("/load", "old", self.client, self.conv, self.state))

    def continue_chat(self):
        self.bodies.clear()
        self.conv.add_user("question three")
        stream = self.client.chat(self.state.model, self.conv.get_messages(), {})
        _sync_server_info(self.state, stream.server_model, stream.server_n_ctx)
        self.conv.add_assistant("".join(stream), model=_reply_model(stream, self.state))

    def test_recorded_model_names_do_not_touch_the_session(self):
        with self.server.patched():
            self.start()
            before = (self.state.model, self.state.context_length, self.state.think_tags)
            out = self.load()
        self.assertEqual((self.state.model, self.state.context_length, self.state.think_tags), before)
        self.assertEqual(self.state.model, "/m/qwen.gguf")
        self.assertIn("saved with model: /m/file-level-model.gguf", out)  # information only

    def test_no_recorded_model_name_reaches_the_server(self):
        with self.server.patched():
            self.start()
            self.load()
            self.continue_chat()
        blob = json.dumps(self.bodies)
        self.assertFalse(any(name in blob for name in RECORDED), blob)
        self.assertEqual([kind for kind, _ in self.bodies], ["apply-template", "completion"])
        template_body = self.bodies[0][1]
        self.assertEqual({k for m in template_body["messages"] for k in m}, {"role", "content"})
        self.assertEqual(len(template_body["messages"]), 6)  # system + 5 messages
        self.assertEqual(template_body["model"], "/m/qwen.gguf")
        self.assertEqual(self.bodies[1][1]["model"], "/m/qwen.gguf")

    def test_the_new_reply_is_attributed_to_the_served_model_and_loaded_ones_keep_theirs(self):
        with self.server.patched():
            self.start()
            self.load()
            self.continue_chat()
        assistants = [m.get("model") for m in self.conv.messages if m["role"] == "assistant"]
        self.assertEqual(assistants, [RECORDED[0], RECORDED[1], "/m/qwen.gguf"])

    def test_messages_and_system_prompt_are_replaced_by_the_file(self):
        with self.server.patched():
            self.start()
            self.load()
        self.assertEqual(self.conv.system_prompt, "SAVED SYSTEM PROMPT")
        self.assertEqual(
            [m["content"] for m in self.conv.messages],
            ["question one", "answer one from a", "question two", "answer two from b"],
        )


class LoadAppliesTheSystemPromptSourceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.state = State(model="m", config={**DEFAULTS, "conversations_dir": self.tmp}, context_length=None)

    def load_into(self, conv):
        render(lambda: handle_command("/load", "old", None, conv, self.state))

    def test_the_files_source_file_is_applied(self):
        save_old_conversation(self.tmp)
        conv = Conversation(system_prompt="CURRENT")
        conv.source_file = "/home/someone/current_prompt.txt"
        self.load_into(conv)
        self.assertEqual(conv.source_file, "/home/someone/saved_prompt.txt")

    def test_a_file_without_a_source_file_clears_the_current_one(self):
        save_old_conversation(self.tmp, source_file=None)
        conv = Conversation(system_prompt="CURRENT")
        conv.source_file = "/home/someone/current_prompt.txt"
        self.load_into(conv)
        self.assertIsNone(conv.source_file)

    def test_a_resave_records_the_loaded_provenance_not_the_old_one(self):
        save_old_conversation(self.tmp)
        conv = Conversation(system_prompt="CURRENT")
        conv.source_file = "/home/someone/current_prompt.txt"
        self.load_into(conv)
        conv.save(self.tmp, name="again", model="m")
        data = json.loads((Path(self.tmp) / "again.json").read_text())
        self.assertEqual(data["system_prompt"], "SAVED SYSTEM PROMPT")
        self.assertEqual(data["source_file"], "/home/someone/saved_prompt.txt")


if __name__ == "__main__":
    unittest.main()
