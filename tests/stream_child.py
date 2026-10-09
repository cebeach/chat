"""Child for test_prompt_pty.py: a held fake reply inside ui.echo_suppressed(), then a prompt.

After each token it tells the test it is at hold "token" and blocks until the test releases it, so the
test can read the tty flags and type keys while the reply is "streaming". After the reply and the `with`
have finished it prints DONE, holds again at "done" (the tty is back to its normal mode there), and
only after that release calls ui.get_user_input(). Run on a pty by tests/pty_helpers.py. Not a test.
"""

import os
import tempfile

import ui

W, R = int(os.environ["CONTROL_W"]), int(os.environ["CONTROL_R"])


def hold(name):
    os.write(W, f"{name}\n".encode())
    os.read(R, 1)


def tokens():
    for token in ("Hello ", "world "):
        yield token
        hold("token")


ui.init_readline(lambda d=tempfile.mkdtemp(): d)
with ui.echo_suppressed():
    ui.display_assistant_stream(tokens())
print("DONE", flush=True)
hold("done")
line = ui.get_user_input()
print("GOT", repr(line), flush=True)
