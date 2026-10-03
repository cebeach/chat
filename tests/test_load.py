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
from conversation import MAX_THINK_PAIRS, Conversation
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
GEMMA_REPLY = "<|channel>thought\nGemma reasoning\n<channel|>Gemma answer"


def write_raw(tmp, name, pairs_raw=None, reply=GEMMA_REPLY):
    """A saved file with arbitrary (even hostile) think_pairs content."""
    data = {
        "model": "/m/gemma.gguf",
        "system_prompt": "",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "model": "/m/gemma.gguf", "content": reply},
        ],
    }
    if pairs_raw is not None:
        data["think_pairs"] = pairs_raw
    (Path(tmp) / f"{name}.json").write_text(json.dumps(data))


class TestLoadMergesRecordedThinkPairs:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = str(tmp_path)

    def session(self, seen, save_thinking=True):
        return State(
            model="/m/qwen.gguf",
            config={**DEFAULTS, "conversations_dir": self.tmp, "save_thinking": save_thinking},
            context_length=4096,
            think_pairs_seen=list(seen),
            auto_save_name="auto_t",
        )

    def load(self, state, name="f", conv=None):
        conv = conv or Conversation()
        render(lambda: handle_command("/load", name, None, conv, state))
        return conv

    def test_a_gemma_conversation_is_stripped_in_a_qwen_only_session(self):
        # The scenario that used to leave the Gemma reasoning in the re-saved file.
        gemma_session = Conversation()
        gemma_session.add_user("q")
        gemma_session.add_assistant(GEMMA_REPLY, model="/m/gemma.gguf")
        gemma_session.save(self.tmp, name="f", model="/m/gemma.gguf", think_pairs=[GEMMA])

        state = self.session([QWEN], save_thinking=False)
        conv = self.load(state)
        conv.add_user("follow-up")
        conv.add_assistant("<think>qwen reasoning</think>Qwen answer", model="/m/qwen.gguf")
        render(lambda: handle_command("/save", "resaved", None, conv, state))

        data = json.loads((Path(self.tmp) / "resaved.json").read_text())
        replies = [m["content"] for m in data["messages"] if m["role"] == "assistant"]
        assert replies == ["Gemma answer", "Qwen answer"]
        assert data["think_pairs"] == [list(QWEN), list(GEMMA)]  # the file records both now
        assert conv.messages[1]["content"] == GEMMA_REPLY  # memory untouched

    def test_pairs_are_added_without_duplicates_even_though_json_gives_lists(self):
        write_raw(self.tmp, "f", [list(QWEN), list(GEMMA), list(GEMMA)])
        state = self.session([QWEN])
        self.load(state)
        assert state.think_pairs_seen == [QWEN, GEMMA]
        self.load(state)  # loading again changes nothing
        assert state.think_pairs_seen == [QWEN, GEMMA]

    def test_invalid_recorded_pairs_are_ignored(self):
        hostile = [
            ["", ""],  # would build a degenerate pattern
            ["a b", "c"],  # whitespace inside
            ["<x>", "so the answer is"],  # prose
            ["<" + "x" * 80 + ">", "</x>"],  # too long
            "not a pair",
            ["<only-one>"],
            [1, 2],
            ["<ok>", "</ok>"],  # the one valid entry
        ]
        write_raw(self.tmp, "f", hostile)
        state = self.session([QWEN])
        self.load(state)
        assert state.think_pairs_seen == [QWEN, ("<ok>", "</ok>")]

    def test_the_session_never_keeps_more_than_the_cap(self):
        write_raw(self.tmp, "f", [[f"<t{i}>", f"</t{i}>"] for i in range(30)])
        empty = self.session([])
        self.load(empty)
        assert len(empty.think_pairs_seen) == MAX_THINK_PAIRS
        assert empty.think_pairs_seen[0] == ("<t0>", "</t0>")
        nearly_full = self.session([(f"<s{i}>", f"</s{i}>") for i in range(MAX_THINK_PAIRS - 1)])
        self.load(nearly_full)
        assert len(nearly_full.think_pairs_seen) == MAX_THINK_PAIRS  # only one more fit

    def test_an_old_file_without_the_key_changes_nothing(self):
        write_raw(self.tmp, "f", None)
        state = self.session([QWEN])
        conv = self.load(state)
        assert state.think_pairs_seen == [QWEN]
        assert len(conv.messages) == 2

    def test_the_toggle_warning_reflects_loaded_pairs(self):
        write_raw(self.tmp, "f", [list(GEMMA)])
        state = self.session([])
        self.load(state)
        with mock.patch.object(chat, "display_info") as info:
            handle_command("/config", "save_thinking off", None, Conversation(), state)
        assert "earlier replies with known tags still are" in info.call_args.args[0]
