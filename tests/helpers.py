"""Fixtures and fakes shared by the tests: captured /apply-template renders, a fake llama-server."""

import _thread
import re
import threading
from dataclasses import dataclass
from unittest import mock

import requests

import ui

U, R, A = "QZUSERQZ", "QZREASONQZ", "QZANSWERQZ"

QWEN_CONT = f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>{R}</think>{A}"
QWEN_GEN = f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n"
# Two verbatim excerpts of the Qwen template as served by /props (joined by "..."):
# the history stripping and the forced-open generation prompt. Captured live.
QWEN_SRC = (
    "        {%- if '</think>' in content %}\n"
    "            {%- set content = content.split('</think>')[-1] | trim %}\n"
    "        {%- endif %}\n...\n"
    "    {{- '<|im_start|>assistant\\n<think>\\n' }}"
)

GEMMA_CONT = f"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{U}<turn|>\n<|turn>model\n<|channel>thought\n{R}<channel|>{A}"
GEMMA_GEN = f"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{U}<turn|>\n<|turn>model\n"
# Verbatim fragments of Gemma's real template as served by /props (joined by "..."),
# captured live; they contain the guard words and both tags literally.
GEMMA_SRC = (
    "{%- if '<|channel>' in part -%}\n...\n"
    "{%- for part in text.split('<channel|>') -%}\n...\n"
    "{{- '<|think|>\\n' -}}\n...\n"
    "{{- '<|channel>thought\\n' + thinking_text"
)

DEVSTRAL_CONT = f"[INST]{U}[/INST][THINK]{R}[/THINK]{A}"
DEVSTRAL_GEN = f"[INST]{U}[/INST]"
# Real head of Devstral's template: no thinking-related word anywhere in it.
DEVSTRAL_SRC = (
    "{#- Default system message if no system prompt is passed. #}\n"
    "{%- set default_system_message = '' %}\n\n{#- Begin of sequence token. #}\n{{- bos_token }}\n"
)

QWEN_PAIR = ("<think>", "</think>")
GEMMA_PAIR = ("<|channel>thought", "<channel|>")


class FakeResponse:
    def __init__(self, json_data=None, lines=()):
        self._json, self._lines = json_data, lines

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    def raise_for_status(self):
        pass

    def close(self):
        self.closed = True

    def iter_lines(self):
        return iter(self._lines)


class FakeServer:
    """Stands in for requests.get/post against llama-server's native endpoints."""

    def __init__(self, model_path, source, cont, gen, real_prompt="<real>", n_ctx=4096):
        self.model_path, self.source, self.n_ctx = model_path, source, n_ctx
        self.cont, self.gen, self.real_prompt = cont, gen, real_prompt
        self.fail = False
        self.bad_json = False
        # What /props reports as total_slots; None leaves the field out.
        self.total_slots = None
        # The JSON body of every /completion request.
        self.completion_payloads = []
        # What /props reports as the server's launch sampling defaults. A real server
        # widens float32 values and reports "random" as the unsigned 4294967295.
        self.params = {
            "seed": 4294967295,
            "temperature": 0.800000011920929,
            "top_p": 0.949999988079071,
            "min_p": 0.05000000074505806,
            "repeat_penalty": 1.0,
            "n_predict": -1,
        }
        self.props_calls = 0
        self.template_calls = []
        self.tokenize_calls = []
        # Set to a callable(messages) -> prompt to render real prompts like a template would
        # (see words_template); /tokenize then counts one token per word, plus one for the BOS
        # that add_special asks for. Left None, a real prompt is the fixed real_prompt.
        self.render = None

    def become(self, model_path, source, cont, gen, real_prompt="<real>", n_ctx=4096):
        self.model_path, self.source, self.n_ctx = model_path, source, n_ctx
        self.cont, self.gen, self.real_prompt = cont, gen, real_prompt

    def get(self, url, **kw):
        assert url.endswith("/props"), url
        if self.fail:
            raise requests.ConnectionError("server down")
        self.props_calls += 1
        if self.bad_json:
            return FakeResponse(ValueError("not json"))
        props = {
            "model_path": self.model_path,
            "model_alias": self.model_path,
            "chat_template": self.source,
            "default_generation_settings": {"n_ctx": self.n_ctx, "params": dict(self.params)},
        }
        if self.total_slots is not None:
            props["total_slots"] = self.total_slots
        return FakeResponse(props)

    def post(self, url, json=None, **kw):
        if url.endswith("/completion"):
            self.completion_payloads.append(json)
            # Like llama-server: only the final chunk names the model that produced the reply.
            final = f'data: {{"content": "", "stop": true, "model": "{self.model_path}"}}'
            return FakeResponse(lines=[b'data: {"content": "x", "stop": false}', final.encode()])
        if self.fail:
            raise requests.ConnectionError("server down")
        if url.endswith("/tokenize"):
            self.tokenize_calls.append(json)
            count = len(json["content"].split()) + (1 if json["add_special"] else 0)
            return FakeResponse({"tokens": list(range(count))})
        assert url.endswith("/apply-template"), url
        self.template_calls.append(json)
        if json.get("continue_final_message"):
            return FakeResponse({"prompt": self.cont})
        messages = json["messages"]
        if len(messages) == 1 and messages[0]["content"] == U:
            return FakeResponse({"prompt": self.gen})
        if self.render:
            return FakeResponse({"prompt": self.render(messages)})
        return FakeResponse({"prompt": self.real_prompt})

    def patched(self):
        return mock.patch.multiple("requests", get=self.get, post=self.post)


def words_template(messages):
    """A stand-in chat template: a role marker word before each message, then the reply's marker."""
    return " ".join(f"<{m['role']}> {m['content']}" for m in messages) + " <assistant>"


def qwen_server():
    return FakeServer("/m/qwen.gguf", QWEN_SRC, QWEN_CONT, QWEN_GEN, QWEN_GEN.replace(U, "hello"))


def gemma_server():
    return FakeServer("/m/gemma.gguf", GEMMA_SRC, GEMMA_CONT, GEMMA_GEN, GEMMA_GEN.replace(U, "hello"))


def devstral_server():
    return FakeServer("/m/devstral.gguf", DEVSTRAL_SRC, DEVSTRAL_CONT, DEVSTRAL_GEN, DEVSTRAL_GEN)


def render(callable_):
    with ui.console.capture() as cap:
        callable_()
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", cap.get()).split())


@dataclass
class Reply:
    """One scripted turn of a /duel side or judge, as SlotScript hands it out."""

    text: str = ""
    context_tokens: int = 100  # what the slot holds once the turn is done (prompt + reply)
    truncated: bool = False
    limited: bool = False
    finished: bool = True  # False: the stream ends without the server's final chunk (no stats)
    think_tags: tuple | None = None
    interrupt_at: int | None = None  # Ctrl-C reaches the main thread just before this token
    fail_on_chat: Exception | None = None  # raised by client.chat() itself
    fail_in_stream: Exception | None = None  # raised after the first token


class ScriptedStream:
    """Stands in for LlamaChatStream with the attributes chat._run_phase reads."""

    def __init__(self, reply, barrier=None):
        self.reply, self.barrier = reply, barrier
        self.tokens = re.findall(r"\S+\s*", reply.text)
        self.server_model, self.server_n_ctx, self.model = "/m/x.gguf", 4096, "/m/x.gguf"
        self.think_tags = reply.think_tags
        self.truncated, self.limited = reply.truncated, reply.limited
        self.stats = {}
        self.closed = False

    def close(self):
        self.closed = True

    def __iter__(self):
        if self.barrier is not None:
            self.barrier.wait(timeout=5)  # a sequential implementation times out here
        for i, token in enumerate(self.tokens):
            if self.reply.interrupt_at == i:
                _thread.interrupt_main()
            if self.reply.fail_in_stream and i == 1:
                raise self.reply.fail_in_stream
            yield token
        if self.reply.finished:
            self.stats = {
                "completion_tokens": len(self.tokens),
                "prompt_tokens": max(self.reply.context_tokens - len(self.tokens), 0),
                "context_tokens": self.reply.context_tokens,
            }


class SlotScript:
    """Ordered scripted replies per `id_slot`, handed out one per turn.

    Install with `setup=script.setup` and `chat_behaviour=script.behaviour` in run_session. setup()
    wraps the client's chat side effect instead of replacing it, so the session still logs each call
    and records its kwargs; the wrapper keeps the call's kwargs in a thread-local, because the two
    sides call client.chat from different worker threads and a shared "last call" would race.
    Assertions should use the per-slot sequences (`calls[slot]`), never the interleaving of the sides.

    barrier=N makes each scripted stream block until N streams of the phase have started iterating.
    """

    def __init__(self, replies, barrier=None, slots=2):
        self.replies = {slot: list(items) for slot, items in replies.items()}
        self.slots = slots
        self.barrier = threading.Barrier(barrier) if barrier else None
        self.calls = {}  # slot -> [kwargs of each chat call]
        self.streams = {}  # slot -> [the ScriptedStream of each call]
        self._local = threading.local()
        self._lock = threading.Lock()

    def setup(self, client):
        original = client.chat.side_effect

        def wrapper(**kwargs):
            self._local.kwargs = kwargs
            return original(**kwargs)

        client.chat.side_effect = wrapper
        client.server_slots = self.slots
        client.sampling_defaults.return_value = {"seed": 4294967295}  # the server's "random"

    def behaviour(self):
        kwargs = self._local.kwargs
        slot = kwargs.get("id_slot")
        with self._lock:
            self.calls.setdefault(slot, []).append(kwargs)
            queue_ = self.replies.get(slot)
            if not queue_:
                raise AssertionError(f"no scripted reply left for slot {slot}")
            reply = queue_.pop(0)
        if reply.fail_on_chat:
            raise reply.fail_on_chat
        stream = ScriptedStream(reply, self.barrier)
        with self._lock:
            self.streams.setdefault(slot, []).append(stream)
        return stream

    def exhausted(self):
        return not any(self.replies.values())
