"""Client for the llama.cpp server's native HTTP API.

Only native endpoints are used: /health, /props, /models, /apply-template
and /completion. See tools/server/README.md in the llama.cpp source.
"""

import json
import re

import requests

# Some chat templates end the generation prompt with an opened thinking block,
# so the model only ever writes the closing tag. See LlamaChatStream.
_OPEN_THINK_RE = re.compile(r"<think>\s*$")


class LlamaClient:
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip("/")

    def is_available(self):
        """Check if llama-server is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self):
        """Return a list of available model names."""
        resp = requests.get(f"{self.base_url}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    def get_context_length(self, model):
        """Query the server's context length from /props.

        Returns the context length as an int, or None if unavailable.
        """
        try:
            resp = requests.get(f"{self.base_url}/props", timeout=10)
            resp.raise_for_status()
            return int(resp.json()["default_generation_settings"]["n_ctx"])
        except (requests.ConnectionError, requests.HTTPError, ValueError, KeyError):
            return None

    def chat(self, model, messages, options=None):
        """Send a chat request. Returns a LlamaChatStream that yields tokens.

        The server applies the model's chat template (/apply-template); the
        resulting prompt is then streamed through /completion.

        Args:
            options: Dict of model parameters (seed, temperature, top_p).
                     None values are omitted.
        """
        resp = requests.post(
            f"{self.base_url}/apply-template",
            json={"model": model, "messages": messages},
            timeout=30,
        )
        resp.raise_for_status()
        prompt = resp.json()["prompt"]
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        if options:
            for k, v in options.items():
                if v is not None:
                    payload[k] = v
        resp = requests.post(
            f"{self.base_url}/completion",
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        open_think = _OPEN_THINK_RE.search(prompt)
        return LlamaChatStream(resp, prefix=open_think.group() if open_think else "")


class LlamaChatStream:
    """Iterable wrapper over a streaming /completion response (SSE).

    After iteration, .stats contains completion_tokens, prompt_tokens,
    eval_duration_ns and (when reported) tokens_per_second, taken from the
    server's final chunk. It stays empty if the stream ends without one.

    prefix is yielded first. chat() uses it to emit an opening <think> tag that
    the template put in the prompt, so the reply shows a balanced block.
    """

    def __init__(self, response, prefix=""):
        self._response = response
        self._prefix = prefix
        self.stats = {}

    def __iter__(self):
        if self._prefix:
            yield self._prefix
        for line in self._response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            token = data.get("content", "")
            if token:
                yield token
            if data.get("stop"):
                self._build_stats(data)
                return

    def _build_stats(self, final):
        timings = final.get("timings", {})
        self.stats = {
            "completion_tokens": final.get("tokens_predicted", 0),
            "eval_duration_ns": int(timings.get("predicted_ms", 0) * 1e6),
            "prompt_tokens": final.get("tokens_evaluated", 0),
        }
        if timings.get("predicted_per_second"):
            self.stats["tokens_per_second"] = timings["predicted_per_second"]
