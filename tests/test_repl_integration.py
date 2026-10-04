"""The real chat.py, on a pty, against the real llama-server that the integration fixture starts.

A round-trip smoke test: it checks that one prompt -> reply -> next prompt -> Ctrl-D works and that the
conversation is autosaved. It does NOT detect the phantom character, and it asserts nothing about the
tty's ECHO flag: at the second prompt readline has already prepared the terminal (ECHO is off there
whether or not echo_suppressed() restored it), and reply timing is not controllable enough to sample
mid-reply. Those are pinned deterministically in test_prompt_pty.py (held stream) and
test_turn_echo.py (which span chat.py wraps). See docs/testing.md.
"""

import pytest

from tests.pty_helpers import ROOT, PtyChild


@pytest.mark.integration
def test_a_real_turn_round_trips_and_is_autosaved(llama_server, tmp_path):
    child = PtyChild(ROOT / "chat.py", tmp_path, args=["--url", llama_server.url])
    try:
        child.wait_prompts(1, timeout=60)  # health + props + welcome
        child.send("Reply with the single word: ok\r")
        # The reply has finished streaming by the second prompt, and this sequence cannot come from reply text.
        child.wait_prompts(2, timeout=180)
        child.send(b"\x04")  # Ctrl-D
        assert child.exit_code(timeout=30) == 0
    finally:
        child.close()
    saved = list((tmp_path / ".local" / "share" / "chat" / "conversations").glob("auto_*.json"))
    assert len(saved) == 1, saved
