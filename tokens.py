"""Token accounting against the server's context window.

Pure logic on top of LlamaClient's /tokenize calls, so it can be tested with a fake
client. "Prompt tokens" are what the next request carries: the system prompt, the
stored messages and the chat template's own markup. Thinking is stored apart from a
reply's content and never sent back, so it is not part of that figure. "Generated
tokens" (thinking plus answer) are the server's count for a reply and are shown on the
stats line only; they are never added to the prompt figure. See docs/statistics.md.
"""

# A prompt above this share of the window gets a warning before it is sent.
WARN_SHARE = 0.8


def fits(needed, n_ctx, reserve=0):
    """True when a prompt of `needed` tokens, plus `reserve` tokens kept for the reply, fits.

    llama-server rejects a request whose prompt alone is n_ctx tokens or more, so even
    with no reserve the prompt must leave room for one token.
    """
    return needed + reserve + 1 <= n_ctx


def near_limit(needed, n_ctx):
    """True when the prompt takes more than 80% of the window."""
    return bool(n_ctx) and needed > WARN_SHARE * n_ctx


def breakdown(client, conversation, model=None):
    """Where the next prompt's tokens go, or None for an empty conversation.

    Returns a dict: system, user and assistant (each counted on its own, without the
    special tokens a whole prompt starts with), total (the exact prompt, as chat() would
    send it), overhead (what is left: the template's markup, joins and the opening of
    the reply, never below 0) and n_ctx (None if the server did not report one).
    Raises requests exceptions when the server cannot be reached.
    """
    messages = conversation.get_messages()
    if not messages:
        return None
    total, n_ctx = client.check_fit(messages, model=model)

    def count(role):
        text = "\n".join(m["content"] for m in messages if m["role"] == role)
        return client.count_tokens(text, add_special=False) if text else 0

    system, user, assistant = count("system"), count("user"), count("assistant")
    return {
        "system": system,
        "user": user,
        "assistant": assistant,
        "overhead": max(0, total - system - user - assistant),
        "total": total,
        "n_ctx": n_ctx,
    }


def refusal(needed, n_ctx, reserve=0, outcome="Not sent"):
    """The error shown when a prompt cannot be sent; `outcome` says what did not happen."""
    kept = f", {reserve:,} kept free for the reply" if reserve else ""
    return (
        f"{outcome}: the prompt would be {needed:,} tokens but the context window is {n_ctx:,}{kept}. "
        "Shorten the text or file, or free space with /clear."
    )
