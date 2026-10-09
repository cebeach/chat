# Testing with a real llama-server

This page is for people working on AI Chat itself. Most of the test suite is offline:
it fakes the server. A few integration tests instead start a real `llama-server`, with
arguments read from a config file that is kept in git, and talk to it. This page covers
those tests and the pytest fixtures behind them.

## Quick start

1. Copy the example settings and say where your GGUF files are:
   ```bash
   cp tests/llama-server.local.example.toml tests/llama-server.local.toml
   $EDITOR tests/llama-server.local.toml        # set models_dir
   ```
   Or skip the file and `export LLAMA_TEST_MODEL_DIR=~/gguf` (an absolute path).
2. Stop any llama-server of your own that is using port 8001 (see [Port 8001](#port-8001-one-server-at-a-time)).
3. Run the integration tests:
   ```bash
   venv/bin/python -m pytest -m integration
   ```

To run only the offline tests, with no server and nothing else needed:

```bash
venv/bin/python -m pytest -m "not integration"
```

Plain `venv/bin/python -m pytest` runs both. With no models directory configured the
integration tests are skipped, so a machine that is not set up still gets a green run.
Once a models directory is configured, plain `pytest` starts a server.

## The fixtures

Defined in `tests/conftest.py`.

| Fixture | Scope | What you get |
|---|---|---|
| `llama_server` | session, one per profile | A llama-server for the profile the test is running against. An object with `.url` (`http://127.0.0.1:8001`), `.port`, `.argv` (the full command line), `.profile` (the profile name), `.extra_args` (the flags from a `llama_args` marker, else `()`) and `.log_dir` |
| `llama_client` | function | A new `LlamaClient` pointed at that server, already refreshed from `/props` |

A test marks itself `integration` and asks for the fixture it needs:

```python
import pytest

@pytest.mark.integration
def test_server_reports_a_context_size(llama_client):
    assert llama_client.server_n_ctx > 0
```

By default a test runs once, against the first profile. With `--llama-model` (below) it
runs once per chosen profile: pytest starts one profile's server when the first test that
needs it runs, runs every test that needs it, stops it, then starts the next. Servers never
overlap, so only one model is in memory at a time and port 8001 is reused. Integration
tests talk to the real network, so they must not run inside `FakeServer.patched()` from
`tests/helpers.py`.

## Choosing a model

A plain run exercises **one** profile, the first in `tests/llama-server.toml`, because
loading a model is slow. Asking for more is opt-in. The value is a profile name, a
comma-separated list, or `all`, taken in order of precedence from:

1. `--llama-model VALUE` on the command line,
2. the `LLAMA_TEST_MODEL` environment variable,
3. otherwise the first profile only.

```bash
venv/bin/python -m pytest -m integration                                   # the first profile
venv/bin/python -m pytest -m integration --llama-model gemma-4-E2B-it-Q8_0  # one named profile
venv/bin/python -m pytest -m integration --llama-model "qwen35-2B-Q8_0,gemma-4-E2B-it-Q8_0"
venv/bin/python -m pytest -m integration --llama-model all                  # every profile
```

How a selection runs:

- Each profile is one server start, one after another, so the cost is one model load per
  profile and per run. Every integration test runs once per profile, so adding tests
  multiplies the time.
- The test IDs show the profile, for example `test_server_smoke[gemma-4-E2B-it-Q8_0]`, and
  one run gives one pass or fail across all of them.
- A list runs in the order you give it, `all` in the order of the file, and a repeated
  name runs once.
- A profile that fails to start errors only its own tests. The others still run.
- An unknown name is an error listing the valid profiles. It is reported when the
  integration tests need the server, so `-m "not integration"` is unaffected.
- `all` and names containing a comma or surrounding spaces cannot be profile names.

Run it from the repository root: the option is registered by `tests/conftest.py`, which
pytest loads at startup only for runs that use the configured `testpaths` or a `tests/`
path.

## Profiles: `tests/llama-server.toml`

This file is committed, so the arguments for each model are reviewed and versioned like
code. Each profile names a GGUF file and the llama-server flags it needs.

```toml
[defaults]
args = [                           # one llama-server flag per element, for every profile
    "--ctx-size 4096",
    "--n-gpu-layers 99",
    "--flash-attn on",
    "--no-webui",
    "--log-prefix",
]

[models.my-model]                  # the profile name, as used by --llama-model
args = [                           # flags for this model, merged over the defaults
    "--ctx-size 8192",             # replaces the default above
    "--reasoning-format none",
]
chat_template = "my-model.jinja"   # optional: a file in tests/chat-templates/
file = "My-Model-Q4_K_M.gguf"      # a filename inside models_dir
remove = ["--log-prefix"]          # optional: drop a default (see below)
source_url = "https://example.com/models/My-Model-Q4_K_M.gguf"   # optional: where the file came from
startup_timeout = 180              # optional: seconds to wait for this model to load
think_tags = ["<think>", "</think>"]   # optional: the thinking tags this model uses ([] if it does not think)
vram_mb = 6200                     # optional: measured memory use with these args
```

| Key | Where | Meaning |
|---|---|---|
| `args` | `[defaults]` and each profile | A list of strings, one llama-server flag per element |
| `file` | profile, required | GGUF filename, relative to your models directory |
| `remove` | profile | Defaults to leave out for this profile; see [Overriding a default](#overriding-a-default) |
| `chat_template` | profile | Template file in `tests/chat-templates/`; see [Chat templates](#chat-templates) |
| `vram_mb` | profile | Positive integer. Accepted and checked, not used yet (see [Parallel servers](#parallel-servers-not-supported-yet)) |
| `startup_timeout` | profile | Positive number of seconds. Beats the local file's value |
| `source_url` | profile | Optional. An `http(s)` link to the GGUF file the profile uses: a reference for where it came from, and a download target for a possible future step. **Nothing downloads it today**: the tests use the file already in your models directory, and this is an air-gapped project. Only its shape is checked, and a URL with credentials in it is refused, because the file is committed |
| `think_tags` | profile | Optional. The thinking tags the model is expected to use, as observed on the real model: `[]` for a model that does not think, or `[start, end]`, two non-empty strings. Read by the [thinking-tag tests](#thinking-tag-tests) |

Any other key is an error that names it, so a typo cannot quietly launch the wrong
command. At least one profile is required.

### Writing the flags

Write each element exactly as you would type it to llama-server. The first word is the
flag and the rest is its value. Both dash flavors work, so `"-ngl 99"` and
`"--n-gpu-layers 99"` are both fine, and the text goes to llama-server as written.

| Element | Command line |
|---|---|
| `"--ctx-size 4096"` | `--ctx-size 4096` |
| `"-c 4096"` | `-c 4096` |
| `"--kv-unified"` | `--kv-unified` (a flag with no value) |
| `"--no-log-prefix"` | `--no-log-prefix` |
| `"--lora a.bin"`, `"--lora b.bin"` | `--lora a.bin --lora b.bin` (repeat the flag) |
| `"--lora 'my file.bin'"` | `--lora "my file.bin"` (shell quoting is understood) |

An element that does not start with a dash, such as `"ctx-size 4096"`, is an error
naming it. Nothing else about the flags is checked. **The fixture does not know which
flags llama-server accepts**, and you are expected to know llama-server. If a flag is
wrong, the server fails to start and the test fails with a pointer to its log (see
[When the server does not start](#when-the-server-does-not-start)). `llama-server --help`
lists the flags and their aliases.

The fixture sets `-m` (the model), `--host 127.0.0.1` and `--port 8001` itself, so
`args` may not contain `-m`, `--model`, `--host` or `--port`.

### Overriding a default

A profile's `args` are added after the defaults. If a profile lists a flag that a default
also lists, **spelled the same way**, the default is dropped:

```toml
[defaults]
args = ["--ctx-size 4096", "--log-prefix"]

[models.big]
file = "big.gguf"
args = ["--ctx-size 8192"]    # --ctx-size 4096 is dropped; --log-prefix stays
```

Spelling matters, because the fixture does not know llama-server's aliases:

- `"-c 8192"` does **not** replace a default `"--ctx-size 4096"`. Both reach llama-server.
  Use the same spelling as the default.
- Nor does it infer negatives. llama-server has `--log-prefix` and `--no-log-prefix`,
  but also oddities such as `-no-kvu` and an option like `--no-host` that is not the
  negative of `--host`. A profile that lists `"--no-log-prefix"` leaves a default
  `"--log-prefix"` in place, and both reach llama-server.

To drop a default without replacing it, or to switch one off with its negative, name the
default in `remove`, spelled exactly as the default spells it, with no value:

```toml
[models.quiet]
file = "quiet.gguf"
remove = ["--log-prefix"]                 # leave it out entirely
# or, to switch it off explicitly:
# args = ["--no-log-prefix"]
# remove = ["--log-prefix"]
```

A `remove` entry that matches no default is an error, which catches a typo or an alias
that would otherwise do nothing.

### Changing flags for one test: `llama_args`

A test that needs the server started differently, for example with a small context window,
marks itself instead of adding a profile:

```python
@pytest.mark.integration
@pytest.mark.llama_args("--ctx-size 2048")        # on a test, or on a class for all its tests
def test_prompt_over_the_window_is_refused(llama_client):
    assert llama_client.server_n_ctx == 2048
```

Each string is one flag written as in `args`, and it is merged over the profile's flags by
the rule in [Overriding a default](#overriding-a-default): the same spelling replaces the
profile's flag (`--ctx-size 2048` replaces `--ctx-size 16384`, `-c 2048` would not), a new
flag is added, and `-m`, `--host` and `--port` are refused. The marker cannot `remove` a
flag.

- The test ID shows it, for example `test_x[qwen35-2B-Q8_0+ctx-size-2048]`, and
  `llama_server.extra_args` and `.argv` carry the override.
- Tests with identical flags share one server. Each distinct set of flags is one more
  model load per selected profile, after the unmarked tests' server has been stopped, so
  port 8001 is still used by one server at a time. Use it sparingly, and remember that
  `--llama-model all` repeats it for every profile.
- A bad flag fails when the server is needed, as an error naming the cause, never at collection.

To add a profile, add a `[models.<name>]` table with `file` (and, as a record, `source_url`), add any flags it needs under
`args`, and run it once with `--llama-model <name>` (or add it to a run with `all`).

## Thinking-tag tests

Models mark their reasoning with different delimiters (`<think>` for Qwen, `<|channel>thought`
for Gemma 4, several `<|...|>` markers for gpt-oss, nothing for non-thinking models), and the
app infers them from the server ([thinking tags](thinking-tags.md)). The offline tests check
that inference against captured template excerpts. `tests/test_think_tags_integration.py`
checks it against the real models, once per selected profile, comparing with the profile's
`think_tags`:

| Test | What it checks |
|---|---|
| `test_detected_tags_match_the_model` | After `refresh()`, `LlamaClient.think_tags` equals the recorded pair, with source `detected`. For `[]` it must be `None` |
| `test_a_real_reply_uses_the_tags` | A real chat reply (prompt `What is 17 + 25? Answer with just the number.`, seed 1, temperature 0, at most 1500 tokens) contains the end tag and begins with the start tag, and `split_think()` returns the answer `42` and a non-empty reasoning. For `[]` the reply is exactly `42` and contains none of the tags any profile records |

The second test is what keeps the first from agreeing with itself. A forced-open template
such as Qwen's ends the prompt with `<think>`, so the model never writes it and the app
re-emits it; for Gemma and gpt-oss the model writes the opener itself. Either way the reply
the app shows begins with the start tag, which is what the test asserts.

A profile with no `think_tags` fails both tests with a message saying so, so a new profile
cannot silently skip this coverage. To record the tags for a new model:

1. Start the model and read what the app detects (`/config`, the `think_tags` row), and what
   it actually writes: send a prompt it has to reason about and look at the raw reply.
   [How to check a new model](thinking-tags.md#how-to-check-a-new-model) lists the server calls.
2. Record `[start, end]`, or `[]` if the reply has no reasoning block, in `think_tags` for the
   profile. Take the values from the model's reply, not only from the app's detection, so the
   test has something independent to compare with.
3. Run `pytest -m integration --llama-model <profile>`.

The reply is deterministic for a fixed seed on one machine (two runs gave byte-identical
output here), but the tests assert structure, not exact text, so a different GPU or driver
should not break them. They would break if a model needs far more than 1500 tokens to finish
thinking on this prompt; the failure message says so.

## Chat templates

Some models need a modified chat template to work with llama.cpp. Keep that template in
git too: put it in `tests/chat-templates/` and name it in the profile. No shipped profile
needs one at the moment, so that directory does not exist until one does; create it.

```toml
[models.my-model]
file = "My-Model-Q4_K_M.gguf"
chat_template = "my-model.jinja"
```

The fixture adds `--chat-template-file <full path>` before `-m`. Rules:

- `chat_template` is a plain filename, with no directory part, and the file must exist.
  Both are checked for every profile on every run, not only the one you selected.
- A profile without `chat_template` uses the template inside the GGUF.
- Start each template file with a `{# ... #}` comment saying which model it is for, what
  was changed from the upstream template, and why.
- Every `.jinja` file in the directory must be used by some profile, and the test suite
  checks that.
- The integration test checks that the server reports your file's text as its template.

Template-related flags in `args` (such as `jinja` or `chat-template`) are passed through
as written and are not checked against `chat_template`.

## Machine-specific settings: `tests/llama-server.local.toml`

This file is not committed. It holds what depends on your machine. Every key is optional.
`tests/llama-server.local.example.toml` shows them all.

| Key | Meaning |
|---|---|
| `models_dir` | Directory with the GGUF files. `$LLAMA_TEST_MODEL_DIR` overrides it |
| `binary` | The llama-server to run. Default: the first `llama-server` on `PATH`. A bare name is looked up on `PATH`; anything with a slash must be an executable file |
| `startup_timeout` | Seconds to wait for the model to load. Default 120. A profile's value wins |
| `[budget]` `vram_mb` | Total GPU memory the tests may use. Accepted and checked, not used yet |

Paths may start with `~`, which is expanded. After that they must be absolute: a relative
path would mean something different depending on where you started pytest, so it is an
error.

The profile arguments for the GPU, such as `n-gpu-layers` and `ctx-size`, are committed
with the profile, and there is no local override for them. If your hardware differs,
edit the profile on your own branch.

## Port 8001, one server at a time

The test server always listens on `127.0.0.1:8001`, the same address the chat app uses by
default. That has consequences:

- Before launching, the fixture checks that nothing is using the port. If something is,
  for example your own llama-server, the test run stops with an error saying so. The
  fixture never uses a server it did not start, and never stops one.
- Only one test server can run at a time, so the integration tests cannot run in
  parallel (do not use pytest-xdist), and cannot run while you have a server of your own
  on 8001. Stop yours, or run `-m "not integration"`.
- The environment variables `LLAMA_ARG_*` and `LLAMA_API_KEY` are removed from the
  server's environment. llama-server reads its options from them, and one left in your
  shell would silently change what the committed profile says. An API key would also
  make `/props` refuse the client, which sends none. The names that were removed are
  recorded in `env-removed.txt` (below). Other variables, such as `CUDA_VISIBLE_DEVICES`,
  pass through.

## When the server does not start

If the server exits, or does not answer `/health` within `startup_timeout`, the tests
that needed it fail with an error (never a skip). The server is stopped straight away, so
it does not keep holding the port or GPU memory for the rest of the run.

The error message starts with the path of the server's log. It then gives the exit code
if there is one, the exact command line, and the last 40 lines of the log. The fixture
does not try to explain the failure: read the log.

Every launch gets its own directory, printed at the end of any run that launched a
server (`llama-server logs: ...`):

| File | Contents |
|---|---|
| `llama-server.log` | The server's standard output and standard error together, in the order they were written |
| `argv.txt` | The command line, one argument per line |
| `env-removed.txt` | Names (never values) of the environment variables that were removed |

The directory is under pytest's temporary directory, typically
`/tmp/pytest-of-<user>/pytest-N/llama-server-<profile>0/`. Use `pytest-current` in place
of `pytest-N` for the latest run, for example
`/tmp/pytest-of-<user>/pytest-current/llama-server-<profile>0/llama-server.log`. The
directory is not deleted by the fixture, and pytest keeps the last three runs.

| What you see | Likely cause | Look at |
|---|---|---|
| `... exited before it became ready (exit code 1)` | A bad flag, a bad template, not enough memory, a corrupt model | `llama-server.log` |
| `... did not become ready within N s` | The model is slow to load | Raise `startup_timeout`; see the end of `llama-server.log` |
| `... exited right after /health answered` | Another server was answering on the port | Whatever is using 8001 |
| `port 8001 is already in use` | Another process owns the port | `ss -ltnp \| grep 8001` |

## Skips and errors

There is exactly one skip: **no models directory is configured**, meaning this machine is
not set up for integration tests. Setting a models directory is how you ask for them, so
from then on a problem is an error that names its cause:

| Situation | Result |
|---|---|
| No `models_dir` and no `$LLAMA_TEST_MODEL_DIR` | Skip, with a message saying how to configure it |
| `binary` in the local file is missing or not executable | Error, even when no models directory is set |
| Models directory is not a directory | Error naming the path |
| The profile's GGUF is not in the models directory | Error naming the path |
| No `binary` set and no `llama-server` on `PATH` | Error |
| A relative path where an absolute one is required | Error |
| Unknown profile name, unknown key, bad type, bad `chat_template` | Error naming the key or profile |
| `-m`, `--model`, `--host` or `--port` in `args` | Error |
| Port 8001 in use | Error |
| The server does not start | Error, pointing at the log |

Errors in `tests/llama-server.toml` itself appear on every machine, configured or not,
as soon as an integration test is selected.

## Parallel servers (not supported yet)

Running several test servers at once would need a calibrated memory footprint for each
profile, small GGUF files to test with, and a limit on how many fit in your GPU memory.
None of that is built. The `vram_mb` keys are accepted and checked so that adding it
later does not change the config files, but nothing reads them, and a profile bigger than
the budget is not an error.

## Pseudo-terminal tests (the prompt and the tty)

The prompt (`ui.get_user_input()`) and the echo handling (`ui.echo_suppressed()`) depend on a real tty
and a real readline, so some tests run a small child process on a pseudo-terminal. They are offline: no
model, no network, nothing on port 8001, and they need only the standard library.

| File | What it covers |
|---|---|
| `tests/test_prompt.py` | Unit tests with mocks: the prompt string, that the prompt never reads `sys.stdin`, the single flush per call, Ctrl-C/Ctrl-D, the non-tty path, and every path of `echo_suppressed()` (restore on exit and on exceptions, a Ctrl-C landing during the tty call, no tty) |
| `tests/test_turn_echo.py` | Runs one turn of the real `chat.main()` with a mocked client and an ordered event log, and asserts that suppression is on from the request to the trailing blank line and off at both prompts, for a normal turn and for each failure and interrupt path |
| `tests/test_prompt_pty.py` | The prompt and the held fake reply on a pty (below) |
| `tests/test_repl_integration.py` | `integration`: the real `chat.py` against the real server, one prompt, one reply, Ctrl-D, and the autosave. A round-trip smoke test; it does not detect the phantom character and asserts nothing about `ECHO` |

How the pty tests work (`tests/pty_helpers.py`, `tests/prompt_child.py`, `tests/stream_child.py`):

- The child gets the pty slave as stdin/stdout/stderr and as its controlling terminal (otherwise Ctrl-C and
  `ISIG` do not behave as in a real session), a throwaway `HOME` (your history and conversations are never
  touched), and the repository root as `cwd` and `PYTHONPATH` (a script run from `tests/` would otherwise
  put `tests/` at `sys.path[0]` and `import ui` would fail). The child scripts are not named `test_*.py`,
  so pytest does not collect them.
- **No sleeps.** The helper waits for the exact coloured prompt sequence readline writes
  (`\x1b[1;32m>>> \x1b[0m`; never the bare text `>>> `, which model output can contain) and counts the
  prompts it has seen. Readline sets up the terminal before it draws the prompt, so keys sent once the
  prompt is visible always arrive in the same state (the tests check this: ECHO and ICANON are off, ISIG is
  on, as soon as the prompt appears, and the prompt is not redrawn by itself).
- **Holds.** `stream_child.py` fakes a reply inside `echo_suppressed()`. After each token it tells the test
  it is at a named hold and blocks on a pipe until the test releases it. At a hold the test reads the tty
  flags with `tcgetattr` on the master and types keys, so the mid-reply state is checked at a known moment
  instead of by luck. A second hold after the reply and before the prompt is what makes "ECHO is back"
  checkable: once readline is running it clears ECHO itself, so a sample taken there says nothing.
- The typed text used during the reply (`qzxj`) shares no character with the fake reply, so any
  occurrence in the output can only be an echo.
- There is **no Ctrl-Z stop test**. A child started in its own session has a parent outside that session, so
  its process group is orphaned and the kernel ignores a terminal stop signal for it; the test could not
  observe a stop. What `fg` does to the prompt is tested without a stop: the redraw is a `SIGCONT` handler
  (`ui._redraw_after_resume`), so the tests send `SIGCONT` to the child at an idle prompt and with text
  typed, and check that the prompt (and the text) is drawn again and that editing carries on. Whether it looks
  right after a real Ctrl-Z and `fg` in your terminal is checked by hand.

What the pty tests cannot tell you is how your own terminal and shell behave (keyboard protocols, bracketed
paste details, `fg` after Ctrl-Z, what `stty -a` shows). Check those by hand; `docs/input.md` lists what to expect.

## The code

| File | Role |
|---|---|
| `tests/conftest.py` | The fixtures, the `--llama-model` option, the per-profile parametrization, and the log-directory summary |
| `tests/llama_server_config.py` | Loading and checking the config files, picking the profile, building the command line, the port check |
| `tests/llama_server_process.py` | Starting the process, waiting for it, stopping it |
| `tests/test_llama_server_fixture.py` | Offline tests of all of the above, and the integration smoke test |
| `tests/pty_helpers.py` | Runs a child process on a pty: controlling terminal, throwaway `HOME`, prompt counting, holds, tty flags |
| `tests/test_prompt.py`, `tests/test_turn_echo.py`, `tests/test_prompt_pty.py`, `tests/test_repl_integration.py` | The prompt and echo tests (see [Pseudo-terminal tests](#pseudo-terminal-tests-the-prompt-and-the-tty)) |
| `tests/test_think_tags_integration.py` | Thinking-tag detection and a real reply, checked against each real model (see [Thinking-tag tests](#thinking-tag-tests)) |

The offline tests never touch port 8001, your real `tests/llama-server.local.toml` or the
real environment, so they give the same result on every machine.
