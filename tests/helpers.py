"""Fixtures and fakes shared by the tests: captured /apply-template renders, a fake llama-server."""

import re
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

    def iter_lines(self):
        return iter(self._lines)


class FakeServer:
    """Stands in for requests.get/post against llama-server's native endpoints."""

    def __init__(self, model_path, source, cont, gen, real_prompt="<real>", n_ctx=4096):
        self.model_path, self.source, self.n_ctx = model_path, source, n_ctx
        self.cont, self.gen, self.real_prompt = cont, gen, real_prompt
        self.fail = False
        self.bad_json = False
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
        return FakeResponse(
            {
                "model_path": self.model_path,
                "model_alias": self.model_path,
                "chat_template": self.source,
                "default_generation_settings": {"n_ctx": self.n_ctx, "params": dict(self.params)},
            }
        )

    def post(self, url, json=None, **kw):
        if url.endswith("/completion"):
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
