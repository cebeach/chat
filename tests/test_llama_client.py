import json
import unittest
from unittest import mock

import requests

from llama_client import LlamaChatStream, LlamaClient

URL = "http://llama.test"


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


class StreamTests(unittest.TestCase):
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
        self.assertEqual("".join(stream), "Hello")
        self.assertEqual(
            stream.stats,
            {
                "completion_tokens": 2,
                "eval_duration_ns": 50_000_000,
                "prompt_tokens": 7,
                "tokens_per_second": 40.0,
            },
        )

    def test_no_final_chunk_leaves_stats_empty(self):
        stream = LlamaChatStream(FakeResponse(lines=[sse(content="x", stop=False)]))
        self.assertEqual(list(stream), ["x"])
        self.assertEqual(stream.stats, {})


class PrefixTests(unittest.TestCase):
    def test_prefix_is_yielded_first(self):
        stream = LlamaChatStream(FakeResponse(lines=[sse(content="hi", stop=False)]), prefix="<think>\n")
        self.assertEqual(list(stream), ["<think>\n", "hi"])


class ClientTests(unittest.TestCase):
    def setUp(self):
        # The override makes chat() skip detection, so these tests only cover the
        # chat flow. Detection itself is covered in test_think_tags.py.
        self.client = LlamaClient(URL + "/", think_override=("<think>", "</think>"))

    def test_list_models_uses_native_models_route(self):
        resp = FakeResponse({"models": [{"name": "a"}, {"name": "b"}], "data": []})
        with mock.patch("requests.get", return_value=resp) as get:
            self.assertEqual(self.client.list_models(), ["a", "b"])
        get.assert_called_once_with(f"{URL}/models", timeout=10)

    def test_context_length_reads_nested_n_ctx(self):
        resp = FakeResponse({"default_generation_settings": {"n_ctx": 4096}})
        with mock.patch("requests.get", return_value=resp):
            self.assertEqual(self.client.get_context_length("m"), 4096)

    def test_context_length_missing_returns_none(self):
        with mock.patch("requests.get", return_value=FakeResponse({"n_ctx": 1})):
            self.assertIsNone(self.client.get_context_length("m"))

    def test_chat_applies_template_then_streams_completion(self):
        template = FakeResponse({"prompt": "<user>hi<assistant>"})
        completion = FakeResponse(lines=[])
        with mock.patch("requests.post", side_effect=[template, completion]) as post:
            stream = self.client.chat(
                "m",
                [{"role": "user", "content": "hi"}],
                {"seed": None, "temperature": 0.2, "top_p": None},
            )
        self.assertIsInstance(stream, LlamaChatStream)
        first, second = post.call_args_list
        self.assertEqual(first.args[0], f"{URL}/apply-template")
        self.assertEqual(first.kwargs["json"]["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(second.args[0], f"{URL}/completion")
        self.assertEqual(
            second.kwargs["json"],
            {"model": "m", "prompt": "<user>hi<assistant>", "stream": True, "temperature": 0.2},
        )
        self.assertTrue(second.kwargs["stream"])
        self.assertEqual(list(stream), [])
        self.assertEqual(stream.think_tags, ("<think>", "</think>", "config"))

    def _stream_for_prompt(self, prompt):
        responses = [FakeResponse({"prompt": prompt}), FakeResponse(lines=[sse(content="x", stop=False)])]
        with mock.patch("requests.post", side_effect=responses):
            return self.client.chat("m", [{"role": "user", "content": "hi"}])

    def test_open_think_tag_in_prompt_is_emitted_first(self):
        stream = self._stream_for_prompt("<|im_start|>assistant\n<think>\n")
        self.assertEqual(list(stream), ["<think>\n", "x"])

    def test_no_open_think_tag_means_no_prefix(self):
        self.assertEqual(list(self._stream_for_prompt("<|im_start|>assistant\n")), ["x"])
        # A closed block earlier in the prompt (e.g. history) is not an opening tag.
        self.assertEqual(list(self._stream_for_prompt("<think>a</think>\nassistant\n")), ["x"])


if __name__ == "__main__":
    unittest.main()
