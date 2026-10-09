"""Client for the llama.cpp server's native HTTP API.

Only native endpoints are used: /health, /props, /apply-template, /tokenize
and /completion. See docs/llama-server-api.md for what each is used for and
when, and tools/server/README.md in the llama.cpp source for the server's side. The model
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


class ThinkTagsError(RuntimeError):
    """The server is serving a different model than the one the thinking tags
    were last settled for, and its tags could not be determined. Raised by
    chat() instead of splitting the reply with the previous model's tags."""


class LlamaClient:
    def __init__(self, base_url="http://127.0.0.1:8001", think_override=None):
        """think_override: optional (start, end) from the config file; when set
        it wins over detection and no server call is made for it."""
        self.base_url = base_url.rstrip("/")
        self._think_cache = {}  # server identity -> (start, end); accepted pairs only
        self._think_current = (*think_override, "config") if think_override else None
        self._think_fixed = bool(think_override)
        # Identity of the server the current tags were settled for (a pair, or a
        # conclusive "none"); None until one was. See _detect_think_tags.
        self._think_identity = None
        self._think_unresolved = None  # error text while a model swap is unresolved
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

    def count_tokens(self, text, add_special=True):
        """The number of tokens the server makes of `text`, from POST /tokenize.

        /completion tokenizes its prompt with add_special=true and parse_special=true
        (the defaults here); /tokenize itself defaults add_special to false, so it is
        always sent explicitly. Pass add_special=False to count a fragment without the
        BOS a whole prompt carries. The timeout grows with the text.
        """
        resp = requests.post(
            f"{self.base_url}/tokenize",
            json={"content": text, "add_special": add_special, "parse_special": True},
            timeout=30 + len(text) // 100_000,
        )
        resp.raise_for_status()
        return len(resp.json()["tokens"])

    def prompt_tokens(self, messages, model=None):
        """The exact size of the prompt chat() would send for `messages`: the server's
        rendering of the chat template (/apply-template), counted by /tokenize."""
        timeout = 30 + sum(len(m["content"]) for m in messages) // 50_000
        body = {"model": model} if model else {}
        return self.count_tokens(self._apply_template(messages, timeout=timeout, **body))

    def check_fit(self, messages, model=None):
        """Price a prompt against the context window: (needed, n_ctx).

        Re-reads the server first (refresh), so n_ctx is the one in force now; pass
        refreshed=True to the chat() that follows to avoid reading it twice. n_ctx is
        None when the server did not report one. Raises requests exceptions like the
        calls it makes.
        """
        self.refresh()
        return self.prompt_tokens(messages, model=model), self.server_n_ctx

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
        value, except that after a model swap chat() then refuses to run (see
        ThinkTagsError). Never raises.
        """
        self.refresh()
        return self._think_current

    def _detect_think_tags(self, props):
        if self._think_fixed:
            return
        self._think_unresolved = None
        try:
            source = props["chat_template"]
            identity = (props["model_path"], hashlib.sha256(source.encode()).hexdigest())
        except (KeyError, TypeError, AttributeError):
            return  # cannot tell which model this is, so cannot tell a swap either
        if identity in self._think_cache:
            self._settle(identity, self._think_cache[identity])
            return

        try:
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
            status, pair = INCONCLUSIVE, None

        if status == PAIR:
            self._think_cache[identity] = pair
            self._settle(identity, pair)
        elif status == NONE:
            self._settle(identity, None)
        elif self._think_identity is not None and identity != self._think_identity:
            # Inconclusive after a model swap: the kept tags belong to the previous
            # model, so chat() must not use them. Without a swap the previous value
            # stays, so a transient problem cannot make the tags flip.
            self._think_unresolved = (
                f"The server is now serving {props.get('model_alias') or props['model_path']}, "
                "but its thinking tags could not be determined, so replies cannot be "
                "separated from their reasoning. Retry, or set think_start and think_end "
                "in the config file."
            )

    def _settle(self, identity, pair):
        self._think_identity = identity
        self._think_current = (*pair, "detected") if pair else None

    def chat(self, model, messages, options=None, refreshed=False):
        """Send a chat request. Returns a LlamaChatStream that yields tokens.

        The server applies the model's chat template (/apply-template); the
        resulting prompt is then streamed through /completion. The server's
        model, context length and thinking tags are re-read first (refresh) and
        attached to the returned stream as .server_model, .server_n_ctx and
        .think_tags.

        Raises:
            ThinkTagsError: The server swapped models and the new model's thinking
                tags could not be determined (nothing is sent to /completion).

        Args:
            options: Dict of model parameters (seed, temperature, top_p).
                     None values are omitted.
            refreshed: True when check_fit() just re-read the server, so this call
                       does not do it again.
        """
        if not refreshed:
            self.refresh()
        if self._think_unresolved:
            raise ThinkTagsError(self._think_unresolved)
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
    context_tokens (prompt + completion: the context in use once the reply is
    done), eval_duration_ns and (when reported) tokens_per_second, taken from the
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
        completion_tokens = final.get("tokens_predicted", 0)
        prompt_tokens = final.get("tokens_evaluated", 0)
        self.stats = {
            "completion_tokens": completion_tokens,
            "context_tokens": prompt_tokens + completion_tokens,
            "eval_duration_ns": int(timings.get("predicted_ms", 0) * 1e6),
            "prompt_tokens": prompt_tokens,
        }
        if timings.get("predicted_per_second"):
            self.stats["tokens_per_second"] = timings["predicted_per_second"]
