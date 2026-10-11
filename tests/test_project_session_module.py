"""project_session.py: never imports chat, and every function that takes a host requires it."""

import inspect
import subprocess
import sys
from pathlib import Path

import project_session

ROOT = Path(__file__).resolve().parent.parent


def test_the_module_imports_without_chat():
    code = "import sys; sys.modules['chat'] = None; import project_session"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_host_is_required_wherever_it_is_taken():
    taken = [
        (name, fn)
        for name, fn in inspect.getmembers(project_session, inspect.isfunction)
        if fn.__module__ == "project_session" and "host" in inspect.signature(fn).parameters
    ]
    assert len(taken) == 11
    for name, fn in taken:
        assert inspect.signature(fn).parameters["host"].default is inspect.Parameter.empty, name
