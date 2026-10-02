"""Client for the llama.cpp server's native HTTP API.

Only native endpoints are used: /health, /props, /apply-template and
/completion. See tools/server/README.md in the llama.cpp source. The model
being served and its context length are read from /props, never chosen by
the client.

Thinking tags are discovered from the server, not hard-coded; see
find_think_tags() and docs/thinking-tags.md.
"""

import hashlib
import json
import re

import requests

# A marker is one tag-shaped token such as <think>, <|channel|> or [THINK].
_MARK = r"(?:<[^<>\s]+>|\[[^\[\]\s]+\])"
# A tag is a contiguous run of markers and plain words, e.g. "<|channel>thought"
# or "<|end|><|start|>assistant<|channel|>final<|message|>".
_TAG_RE = re.compile(rf"{_MARK}(?:{_MARK}|[A-Za-z0-9_.:-]+)*")
# The template must mention thinking at all, or the server's family-level guess
# (e.g. Mistral's [THINK]) is not evidence that this model uses it.
_THINK_WORDS_RE = re.compile(r"think|reason|thought|analysis|channel", re.I)
_OPENED_MARKER_RE = re.compile(r"(<[^<>\s]+>|\[[^\[\]\s]+\])\s*$")

_START_MAX = 60
_END_MAX = 80

# Sentinels used to probe the chat template through /apply-template.
SENTINEL_USER = "QZUSERQZ"
SENTINEL_REASONING = "QZREASONQZ"
SENTINEL_ANSWER = "QZANSWERQZ"

PAIR = "pair"
NONE = "none"  # conclusive: this model/template has no (acceptable) thinking tags
INCONCLUSIVE = "inconclusive"  # could not tell; callers keep the previous value


def is_valid_think_pair(start, end):
    """True if (start, end) has the shape of a thinking-tag pair.

    Both must be non-empty strings, a contiguous run of markers and plain words
    (no whitespace, starting with a marker), within the length limits. Used for
    detected pairs and for pairs read from a saved file, which is untrusted input:
    an empty or prose-like pair would build a degenerate strip pattern.
    """
    return (
        isinstance(start, str)
        and isinstance(end, str)
        and 0 < len(start) <= _START_MAX
        and 0 < len(end) <= _END_MAX
        and _TAG_RE.fullmatch(start) is not None
        and _TAG_RE.fullmatch(end) is not None
    )


def find_think_tags(continuation, generation, template_source):
    """Infer a model's thinking tags from two /apply-template renders.

    continuation: render of [user, assistant(reasoning, answer)] with
        add_generation_prompt=false and continue_final_message=true; llama.cpp
        writes its own derived tags around the reasoning here.
    generation: render of [user] with add_generation_prompt=true. The user
        message must be the same as in `continuation`.
    template_source: the chat template from GET /props.

    Returns (PAIR, (start, end)), (NONE, None) or (INCONCLUSIVE, None).
    Layer 1 anchors the start tag on the generation prompt, so it is exactly
    what the model must emit to open its reasoning. Layer 2 is a fallback for
    templates whose generation prompt ends in an opened marker. See
    docs/thinking-tags.md for the reasoning behind each rule.
    """
    gen = generation.rstrip()
    anchor_failed = False

    r, a = SENTINEL_REASONING, SENTINEL_ANSWER
    if r in continuation and a in continuation and continuation.index(r) < continuation.index(a):
        before = continuation[: continuation.index(r)]
        between = continuation[continuation.index(r) + len(r) : continuation.index(a)]
        if not before.startswith(gen):
            # Cannot tell (e.g. a timestamp changed between the two renders).
            anchor_failed = True
        else:
            start = before[len(gen) :].strip()
            if not start:
                # Forced-open template: the prompt already ends with the opener.
                m = _OPENED_MARKER_RE.search(gen)
                start = m.group(1) if m else ""
            end = between.strip()
            if is_valid_think_pair(start, end) and _THINK_WORDS_RE.search(template_source):
                return PAIR, (start, end)

    m = _OPENED_MARKER_RE.search(generation)
    if m and m.group(1)[1] != "/":
        opener = m.group(1)
        closer = "</" + opener[1:] if opener[0] == "<" else "[/" + opener[1:]
        if closer in template_source:
            return PAIR, (opener, closer)

    return (INCONCLUSIVE if anchor_failed else NONE), None


class LlamaClient:
    def __init__(self, base_url="http://127.0.0.1:8001", think_override=None):
        """think_override: optional (start, end) from the config file; when set
        it wins over detection and no server call is made for it."""
        self.base_url = base_url.rstrip("/")
        self._think_cache = {}  # server identity -> (start, end); accepted pairs only
        self._think_current = (*think_override, "config") if think_override else None
        self._think_fixed = bool(think_override)
        # What the server is serving, from GET /props (see refresh); None until read.
        self.server_model = None
        self.server_n_ctx = None

    def is_available(self):
        """Check if llama-server is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def _apply_template(self, messages, timeout=30, **body):
        resp = requests.post(
            f"{self.base_url}/apply-template",
            json={"messages": messages, **body},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["prompt"]

    @property
    def think_tags(self):
        """The active thinking tags as of the last refresh: (start, end, source) or None."""
        return self._think_current

    def refresh(self):
        """Re-read what the server is serving with a single GET /props.

        Updates server_model and server_n_ctx, then the thinking tags from the
        same response (see detect_think_tags). The previous values are kept
        when /props cannot be read or lacks a field. Never raises.
        """
        try:
            resp = requests.get(f"{self.base_url}/props", timeout=10)
            resp.raise_for_status()
            props = resp.json()
        except (requests.RequestException, ValueError):
            return
        if not isinstance(props, dict):
            return
        model = props.get("model_alias") or props.get("model_path")
        if model:
            self.server_model = model
        try:
            self.server_n_ctx = int(props["default_generation_settings"]["n_ctx"])
        except (KeyError, TypeError, ValueError):
            pass
        self._detect_think_tags(props)

    def detect_think_tags(self):
        """Return the active thinking tags as (start, end, source), or None.

        source is "config" or "detected". Accepted pairs are memoized by server
        identity (model path + template hash) so a restart with another model
        is noticed; a None result is recomputed on the next call. A conclusive
        "no thinking tags" result clears the value, while an inconclusive one
        (server error, unparseable response, failed anchor) keeps the previous
        value. Never raises.
        """
        self.refresh()
        return self._think_current

    def _detect_think_tags(self, props):
        if self._think_fixed:
            return
        try:
            source = props["chat_template"]
            identity = (props["model_path"], hashlib.sha256(source.encode()).hexdigest())
            if identity in self._think_cache:
                self._think_current = (*self._think_cache[identity], "detected")
                return

            user = {"role": "user", "content": SENTINEL_USER}
            continuation = self._apply_template(
                [
                    user,
                    {
                        "role": "assistant",
                        "reasoning_content": SENTINEL_REASONING,
                        "content": SENTINEL_ANSWER,
                    },
                ],
                add_generation_prompt=False,
                continue_final_message=True,
            )
            generation = self._apply_template([user])
            status, pair = find_think_tags(continuation, generation, source)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return

        if status == PAIR:
            self._think_cache[identity] = pair
            self._think_current = (*pair, "detected")
        elif status == NONE:
            self._think_current = None

    def chat(self, model, messages, options=None):
        """Send a chat request. Returns a LlamaChatStream that yields tokens.

        The server applies the model's chat template (/apply-template); the
        resulting prompt is then streamed through /completion. The server's
        model, context length and thinking tags are re-read first (refresh) and
        attached to the returned stream as .server_model, .server_n_ctx and
        .think_tags.

        Args:
            options: Dict of model parameters (seed, temperature, top_p).
                     None values are omitted.
        """
        self.refresh()
        think = self._think_current
        prompt = self._apply_template(messages, model=model)
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

        # Some templates end the generation prompt with the opened thinking tag,
        # so the model only writes the closing one. Re-emit the opener so the
        # reply shows a balanced block.
        prefix = ""
        stripped_prompt = prompt.rstrip()
        if think and stripped_prompt.endswith(think[0]):
            prefix = prompt[len(stripped_prompt) - len(think[0]) :]
        stream = LlamaChatStream(resp, prefix=prefix)
        stream.think_tags = think
        stream.server_model = self.server_model
        stream.server_n_ctx = self.server_n_ctx
        return stream


class LlamaChatStream:
    """Iterable wrapper over a streaming /completion response (SSE).

    After iteration, .stats contains completion_tokens, prompt_tokens,
    eval_duration_ns and (when reported) tokens_per_second, taken from the
    server's final chunk. It stays empty if the stream ends without one.

    prefix is yielded first (the opening thinking tag the template put in the
    prompt). think_tags is the (start, end, source) used for this request, or
    None; server_model and server_n_ctx are what the server reported for it
    (None if unknown). chat() sets all three.
    """

    def __init__(self, response, prefix=""):
        self._response = response
        self._prefix = prefix
        self.stats = {}
        self.think_tags = None
        self.server_model = None
        self.server_n_ctx = None
        # The model that actually produced the reply, from the final chunk; None
        # if the stream ended (or was interrupted) before that chunk.
        self.model = None

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
                self.model = data.get("model")
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
