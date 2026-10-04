"""/load applies the whole saved conversation against the currently served model.

The model names recorded in the file (file-level and per message) are information
only: they never select a model, never reach the server, and the saved system
prompt's source file is applied along with the prompt itself.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

import chat
from chat import State, _reply_model, _sync_server_info, handle_command
from config import DEFAULTS
from conversation import Conversation
from llama_client import LlamaClient
from tests.helpers import qwen_server, render

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


class TestLoadAppliesAgainstTheCurrentModel:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
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
        assert (self.state.model, self.state.context_length, self.state.think_tags) == before
        assert self.state.model == "/m/qwen.gguf"
        assert "saved with model: /m/file-level-model.gguf" in out  # information only

    def test_no_recorded_model_name_reaches_the_server(self):
        with self.server.patched():
            self.start()
            self.load()
            self.continue_chat()
        blob = json.dumps(self.bodies)
        assert not any(name in blob for name in RECORDED), blob
        assert [kind for kind, _ in self.bodies] == ["apply-template", "completion"]
        template_body = self.bodies[0][1]
        assert {k for m in template_body["messages"] for k in m} == {"role", "content"}
        assert len(template_body["messages"]) == 6  # system + 5 messages
        assert template_body["model"] == "/m/qwen.gguf"
        assert self.bodies[1][1]["model"] == "/m/qwen.gguf"

    def test_the_new_reply_is_attributed_to_the_served_model_and_loaded_ones_keep_theirs(self):
        with self.server.patched():
            self.start()
            self.load()
            self.continue_chat()
        assistants = [m.get("model") for m in self.conv.messages if m["role"] == "assistant"]
        assert assistants == [RECORDED[0], RECORDED[1], "/m/qwen.gguf"]

    def test_messages_and_system_prompt_are_replaced_by_the_file(self):
        with self.server.patched():
            self.start()
            self.load()
        assert self.conv.system_prompt == "SAVED SYSTEM PROMPT"
        assert [m["content"] for m in self.conv.messages] == [
            "question one",
            "answer one from a",
            "question two",
            "answer two from b",
        ]


class TestLoadAppliesTheSystemPromptSource:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)
        self.state = State(model="m", config={**DEFAULTS, "conversations_dir": self.tmp}, context_length=None)

    def load_into(self, conv):
        render(lambda: handle_command("/load", "old", None, conv, self.state))

    def test_the_files_source_file_is_applied(self):
        save_old_conversation(self.tmp)
        conv = Conversation(system_prompt="CURRENT")
        conv.source_file = "/home/someone/current_prompt.txt"
        self.load_into(conv)
        assert conv.source_file == "/home/someone/saved_prompt.txt"

    def test_a_file_without_a_source_file_clears_the_current_one(self):
        save_old_conversation(self.tmp, source_file=None)
        conv = Conversation(system_prompt="CURRENT")
        conv.source_file = "/home/someone/current_prompt.txt"
        self.load_into(conv)
        assert conv.source_file is None

    def test_a_resave_records_the_loaded_provenance_not_the_old_one(self):
        save_old_conversation(self.tmp)
        conv = Conversation(system_prompt="CURRENT")
        conv.source_file = "/home/someone/current_prompt.txt"
        self.load_into(conv)
        conv.save(self.tmp, name="again", model="m")
        data = json.loads((Path(self.tmp) / "again.json").read_text())
        assert data["system_prompt"] == "SAVED SYSTEM PROMPT"
        assert data["source_file"] == "/home/someone/saved_prompt.txt"


QWEN = ("<think>", "</think>")
GEMMA = ("<|channel>thought", "<channel|>")


class TestLoadKeepsThinkingSeparate:
    """A conversation is split when each reply is generated, so loading it needs
    no tags: whatever model the loading session serves."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)

    def session(self, save_thinking=True, tags=None):
        return State(
            model="/m/qwen.gguf",
            config={**DEFAULTS, "conversations_dir": self.tmp, "save_thinking": save_thinking},
            context_length=4096,
            think_tags=tags,
            auto_save_name="auto_t",
        )

    def load(self, state, name="f", conv=None):
        conv = conv or Conversation()
        render(lambda: handle_command("/load", name, None, conv, state))
        return conv

    def save_gemma_conversation(self):
        gemma = Conversation()
        gemma.add_user("q")
        gemma.add_assistant("Gemma answer", model="/m/gemma.gguf", thinking="Gemma reasoning")
        gemma.save(self.tmp, name="f", model="/m/gemma.gguf")

    def test_a_gemma_conversation_loads_whole_in_a_qwen_session(self):
        self.save_gemma_conversation()
        state = self.session(tags=(*QWEN, "detected"))
        conv = self.load(state)
        assert conv.messages[1]["thinking"] == "Gemma reasoning"
        assert conv.messages[1]["content"] == "Gemma answer"
        assert state.think_tags == (*QWEN, "detected")  # /load never touches the session's tags

    def test_the_loaded_reasoning_is_not_sent_to_the_model(self):
        self.save_gemma_conversation()
        conv = self.load(self.session())
        assert conv.get_messages()[-1] == {"role": "assistant", "content": "Gemma answer"}

    def test_a_resave_with_save_thinking_off_drops_every_thinking_field(self):
        self.save_gemma_conversation()
        state = self.session(save_thinking=False, tags=(*QWEN, "detected"))
        conv = self.load(state)
        conv.add_user("follow-up")
        conv.add_assistant("Qwen answer", model="/m/qwen.gguf", thinking="qwen reasoning")
        render(lambda: handle_command("/save", "resaved", None, conv, state))

        data = json.loads((Path(self.tmp) / "resaved.json").read_text())
        assert [m["content"] for m in data["messages"] if m["role"] == "assistant"] == [
            "Gemma answer",
            "Qwen answer",
        ]
        assert all("thinking" not in m for m in data["messages"])
        assert "thinking" in conv.messages[1]  # memory untouched

    def test_a_file_with_a_legacy_think_pairs_key_is_ignored_not_interpreted(self):
        reply = "<|channel>thought\nGemma reasoning\n<channel|>Gemma answer"
        (Path(self.tmp) / "f.json").write_text(
            json.dumps(
                {
                    "model": "/m/gemma.gguf",
                    "system_prompt": "",
                    "think_pairs": [list(GEMMA)],
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": reply},
                    ],
                }
            )
        )
        conv = self.load(self.session())
        assert conv.messages[1]["content"] == reply
        assert "thinking" not in conv.messages[1]
        assert not hasattr(conv, "think_pairs")

    def test_the_toggle_warning_says_replies_cannot_be_separated(self):
        state = self.session(tags=None)
        with mock.patch.object(chat, "display_info") as info:
            handle_command("/config", "save_thinking off", None, Conversation(), state)
        assert "saved whole" in info.call_args.args[0]
