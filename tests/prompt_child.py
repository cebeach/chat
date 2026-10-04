"""Child for test_prompt_pty.py: reads lines with ui.get_user_input() and prints each as `GOT <repr>`.

Run on a pty by tests/pty_helpers.py with cwd and PYTHONPATH set to the repository root. Not a test
(the name does not start with test_), so pytest does not collect it.
"""

import tempfile

import ui

ui.init_readline(tempfile.mkdtemp())
while True:
    line = ui.get_user_input()
    print("GOT", repr(line), flush=True)
    if line is None:
        break
