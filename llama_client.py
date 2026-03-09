import json
import time

import requests


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
        resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def get_context_length(self, model):
        """Query the server's context length from /props.

        Returns the context length as an int, or None if unavailable.
        """
        try:
            resp = requests.get(f"{self.base_url}/props", timeout=10)
            resp.raise_for_status()
            return int(resp.json()["n_ctx"])
        except (requests.ConnectionError, requests.HTTPError, ValueError, KeyError):
            return None

    def chat(self, model, messages, options=None):
        """Send a chat request. Returns a LlamaChatStream that yields tokens.

        Args:
            options: Dict of model parameters (seed, temperature, top_p).
                     None values are omitted.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            for k, v in options.items():
                if v is not None:
                    payload[k] = v
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        return LlamaChatStream(resp)


class LlamaChatStream:
    """Iterable wrapper over a streaming llama.cpp chat response (SSE/OpenAI format).

    After iteration, .stats contains token metadata in Ollama-compatible format.
    """

    def __init__(self, response):
        self._response = response
        self.stats = {}

    def __iter__(self):
        start = time.monotonic()
        usage = None
        token_count = 0

        for line in self._response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if token:
                yield token
                token_count += 1
            chunk_usage = data.get("usage")
            if chunk_usage:
                usage = chunk_usage

        elapsed = time.monotonic() - start
        self._build_stats(elapsed, usage, token_count)

    def _build_stats(self, elapsed_seconds, usage, token_count):
        if usage:
            eval_count = usage.get("completion_tokens", token_count)
            prompt_eval_count = usage.get("prompt_tokens", 0)
        else:
            eval_count = token_count
            prompt_eval_count = 0

        eval_duration_ns = int(elapsed_seconds * 1e9)
        self.stats = {
            "eval_count": eval_count,
            "eval_duration": eval_duration_ns,
            "prompt_eval_count": prompt_eval_count,
        }
        if eval_count and elapsed_seconds > 0:
            self.stats["tokens_per_second"] = eval_count / elapsed_seconds
