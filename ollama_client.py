import json

import requests


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def is_available(self):
        """Check if Ollama is running."""
        try:
            resp = requests.get(self.base_url, timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self):
        """Return a list of available model names."""
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    def get_context_length(self, model):
        """Query the model's context length from Ollama.

        Returns the context length as an int, or None if unavailable.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"model": model},
                timeout=10,
            )
            resp.raise_for_status()
            model_info = resp.json().get("model_info", {})
            for key, value in model_info.items():
                if key.endswith(".context_length"):
                    return int(value)
        except (requests.ConnectionError, requests.HTTPError, ValueError, KeyError):
            pass
        return None

    def chat(self, model, messages, stream=True, options=None):
        """Send a chat request. Returns a ChatStream that yields tokens.

        The ChatStream object also exposes a .stats dict after iteration
        completes, containing eval_count, tokens_per_second, etc.

        If stream=False, yields a single string with the full response.

        Args:
            options: Dict of model parameters (seed, temperature, top_p).
                     None values are omitted so Ollama uses its defaults.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            filtered = {k: v for k, v in options.items() if v is not None}
            if filtered:
                payload["options"] = filtered
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=stream,
            timeout=120,
        )
        resp.raise_for_status()

        return ChatStream(resp, stream)


class ChatStream:
    """Iterable wrapper over a streaming Ollama chat response.

    After iteration, .stats contains token metadata from the final chunk.
    """

    def __init__(self, response, stream):
        self._response = response
        self._stream = stream
        self.stats = {}

    def __iter__(self):
        if not self._stream:
            data = self._response.json()
            self._extract_stats(data)
            yield data.get("message", {}).get("content", "")
            return

        for line in self._response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                self._extract_stats(data)
                return

    def _extract_stats(self, data):
        for key in (
            "eval_count",
            "eval_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
        ):
            if key in data:
                self.stats[key] = data[key]
        # Calculate tokens/sec from nanosecond durations
        eval_count = self.stats.get("eval_count", 0)
        eval_duration = self.stats.get("eval_duration", 0)
        if eval_count and eval_duration:
            self.stats["tokens_per_second"] = eval_count / (eval_duration / 1e9)
