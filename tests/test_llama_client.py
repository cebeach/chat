import json
from unittest import mock

import pytest
import requests

from llama_client import LlamaChatStream, LlamaClient

URL = "http://llama.test"

PROPS = {
    "model_alias": "served.gguf",
    "model_path": "/m/served.gguf",
    "default_generation_settings": {"n_ctx": 4096},
    "chat_template": "{# mentions think #}",
}


class FakeResponse:
    def __init__(self, json_data=None, lines=(), status=200):
        self._json = json_data
        self._lines = lines
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_lines(self):
        return iter(self._lines)


def sse(**fields):
    return ("data: " + json.dumps(fields)).encode()


def props_patch(props=PROPS):
    return mock.patch("requests.get", return_value=FakeResponse(props))


class TestStream:
    def test_yields_content_and_builds_stats_from_final_chunk(self):
        lines = [
            sse(content="Hel", stop=False),
            b"",
            sse(content="lo", stop=False),
            b"data: not json",
            sse(
                content="",
                stop=True,
                tokens_predicted=2,
                tokens_evaluated=7,
                timings={"predicted_ms": 50.0, "predicted_per_second": 40.0},
            ),
            sse(content="ignored", stop=False),
        ]
        stream = LlamaChatStream(FakeResponse(lines=lines))
        assert "".join(stream) == "Hello"
        assert stream.stats == {
            "completion_tokens": 2,
            "context_tokens": 9,
            "eval_duration_ns": 50_000_000,
            "prompt_tokens": 7,
            "tokens_per_second": 40.0,
        }

    def test_no_final_chunk_leaves_stats_empty(self):
        stream = LlamaChatStream(FakeResponse(lines=[sse(content="x", stop=False)]))
        assert list(stream) == ["x"]
        assert stream.stats == {}

    def test_records_the_model_named_in_the_final_chunk_only(self):
        lines = [
            sse(content="a", stop=False, model="ignored-on-partial-chunks"),
            sse(content="", stop=True, model="served.gguf"),
        ]
        stream = LlamaChatStream(FakeResponse(lines=lines))
        assert stream.model is None  # nothing known before the stream is read
        list(stream)
        assert stream.model == "served.gguf"

    def test_model_stays_none_without_a_final_chunk_or_without_a_model_field(self):
        interrupted = LlamaChatStream(FakeResponse(lines=[sse(content="x", stop=False, model="m")]))
        list(interrupted)
        assert interrupted.model is None
        no_field = LlamaChatStream(FakeResponse(lines=[sse(content="", stop=True)]))
        list(no_field)
        assert no_field.model is None


class TestPrefix:
    def test_prefix_is_yielded_first(self):
        stream = LlamaChatStream(FakeResponse(lines=[sse(content="hi", stop=False)]), prefix="<think>\n")
        assert list(stream) == ["<think>\n", "hi"]


class TestRefresh:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # The override makes refresh() skip thinking-tag detection, so these tests
        # only cover what the server reports about itself.
        self.client = LlamaClient(URL + "/", think_override=("<think>", "</think>"))

    def test_nothing_is_known_before_the_first_refresh(self):
        assert self.client.server_model is None
        assert self.client.server_n_ctx is None

    def test_reads_model_alias_and_nested_n_ctx_from_one_props_get(self):
        with props_patch() as get:
            self.client.refresh()
        get.assert_called_once_with(f"{URL}/props", timeout=10)
        assert self.client.server_model == "served.gguf"
        assert self.client.server_n_ctx == 4096

    def test_falls_back_to_model_path_without_an_alias(self):
        props = {k: v for k, v in PROPS.items() if k != "model_alias"}
        with props_patch(props):
            self.client.refresh()
        assert self.client.server_model == "/m/served.gguf"

    def test_follows_a_restarted_server(self):
        with props_patch():
            self.client.refresh()
        other = {**PROPS, "model_alias": "other.gguf", "default_generation_settings": {"n_ctx": 131072}}
        with props_patch(other):
            self.client.refresh()
        assert (self.client.server_model, self.client.server_n_ctx) == ("other.gguf", 131072)

    def test_an_unreadable_props_keeps_the_previous_values(self):
        with props_patch():
            self.client.refresh()
        with mock.patch("requests.get", side_effect=requests.ConnectionError("down")):
            self.client.refresh()  # must not raise
        with mock.patch("requests.get", return_value=FakeResponse(status=500)):
            self.client.refresh()
        with mock.patch("requests.get", return_value=FakeResponse(["not", "a", "dict"])):
            self.client.refresh()
        assert (self.client.server_model, self.client.server_n_ctx) == ("served.gguf", 4096)

    def test_a_missing_field_keeps_only_that_previous_value(self):
        with props_patch():
            self.client.refresh()
        with props_patch({"model_alias": "other.gguf", "chat_template": "x"}):
            self.client.refresh()
        assert (self.client.server_model, self.client.server_n_ctx) == ("other.gguf", 4096)

    def test_the_override_makes_no_template_calls_but_still_reads_props(self):
        with props_patch() as get, mock.patch("requests.post") as post:
            self.client.refresh()
        assert get.call_count == 1
        post.assert_not_called()
        assert self.client.think_tags == ("<think>", "</think>", "config")


class TestClient:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # The override makes chat() skip detection, so these tests only cover the
        # chat flow. Detection itself is covered in test_think_tags.py.
        self.client = LlamaClient(URL + "/", think_override=("<think>", "</think>"))

    def test_chat_applies_template_then_streams_completion(self):
        template = FakeResponse({"prompt": "<user>hi<assistant>"})
        completion = FakeResponse(lines=[])
        with props_patch(), mock.patch("requests.post", side_effect=[template, completion]) as post:
            stream = self.client.chat(
                "m",
                [{"role": "user", "content": "hi"}],
                {"seed": None, "temperature": 0.2, "top_p": None},
            )
        assert isinstance(stream, LlamaChatStream)
        first, second = post.call_args_list
        assert first.args[0] == f"{URL}/apply-template"
        assert first.kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]
        assert second.args[0] == f"{URL}/completion"
        assert second.kwargs["json"] == {
            "model": "m",
            "prompt": "<user>hi<assistant>",
            "stream": True,
            "temperature": 0.2,
        }
        assert second.kwargs["stream"]
        assert list(stream) == []
        assert stream.think_tags == ("<think>", "</think>", "config")

    def test_chat_attaches_what_the_server_reported_for_this_request(self):
        responses = [FakeResponse({"prompt": "p"}), FakeResponse(lines=[])]
        with props_patch(), mock.patch("requests.post", side_effect=responses):
            stream = self.client.chat("m", [{"role": "user", "content": "hi"}])
        assert (stream.server_model, stream.server_n_ctx) == ("served.gguf", 4096)

    def test_chat_stream_reports_the_model_that_produced_the_reply(self):
        final = sse(content="", stop=True, model="served.gguf")
        responses = [FakeResponse({"prompt": "p"}), FakeResponse(lines=[final])]
        with props_patch(), mock.patch("requests.post", side_effect=responses):
            stream = self.client.chat("some-label", [{"role": "user", "content": "hi"}])
        list(stream)
        assert stream.model == "served.gguf"  # not the label we sent

    def _stream_for_prompt(self, prompt):
        responses = [FakeResponse({"prompt": prompt}), FakeResponse(lines=[sse(content="x", stop=False)])]
        with props_patch(), mock.patch("requests.post", side_effect=responses):
            return self.client.chat("m", [{"role": "user", "content": "hi"}])

    def test_open_think_tag_in_prompt_is_emitted_first(self):
        stream = self._stream_for_prompt("<|im_start|>assistant\n<think>\n")
        assert list(stream) == ["<think>\n", "x"]

    def test_no_open_think_tag_means_no_prefix(self):
        assert list(self._stream_for_prompt("<|im_start|>assistant\n")) == ["x"]
        # A closed block earlier in the prompt (e.g. history) is not an opening tag.
        assert list(self._stream_for_prompt("<think>a</think>\nassistant\n")) == ["x"]
