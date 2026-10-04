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
| `llama_server` | session | A llama-server started once for the whole run. An object with `.url` (`http://127.0.0.1:8001`), `.port`, `.argv` (the full command line), `.profile` (the profile name) and `.log_dir` |
| `llama_client` | function | A new `LlamaClient` pointed at that server, already refreshed from `/props` |

A test marks itself `integration` and asks for the fixture it needs:

```python
import pytest

@pytest.mark.integration
def test_server_reports_a_context_size(llama_client):
    assert llama_client.server_n_ctx > 0
```

The server starts when the first test that needs it runs, and stops at the end of the
session. Integration tests talk to the real network, so they must not run inside
`FakeServer.patched()` from `tests/helpers.py`.

## Choosing a model

One profile (one model) runs per pytest run, because loading a model is slow. Pick it
with, in order of precedence:

1. `--llama-model NAME` on the command line,
2. the `LLAMA_TEST_MODEL` environment variable,
3. otherwise the first profile in `tests/llama-server.toml`.

```bash
venv/bin/python -m pytest -m integration --llama-model <profile>
```

To cover several models, run pytest once per profile. Run it from the repository root:
the option is registered by `tests/conftest.py`, which pytest loads at startup only for
runs that use the configured `testpaths` or a `tests/` path.

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
file = "My-Model-Q4_K_M.gguf"      # a filename inside models_dir
vram_mb = 6200                     # optional: measured memory use with these args
startup_timeout = 180              # optional: seconds to wait for this model to load
chat_template = "my-model.jinja"   # optional: a file in tests/chat-templates/
args = [                           # flags for this model, merged over the defaults
    "--ctx-size 8192",             # replaces the default above
    "--reasoning-format none",
]
remove = ["--log-prefix"]          # optional: drop a default (see below)
```

| Key | Where | Meaning |
|---|---|---|
| `args` | `[defaults]` and each profile | A list of strings, one llama-server flag per element |
| `file` | profile, required | GGUF filename, relative to your models directory |
| `remove` | profile | Defaults to leave out for this profile; see [Overriding a default](#overriding-a-default) |
| `chat_template` | profile | Template file in `tests/chat-templates/`; see [Chat templates](#chat-templates) |
| `vram_mb` | profile | Positive integer. Accepted and checked, not used yet (see [Parallel servers](#parallel-servers-not-supported-yet)) |
| `startup_timeout` | profile | Positive number of seconds. Beats the local file's value |

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

To add a profile, add a `[models.<name>]` table with `file`, add any flags it needs under
`args`, and run it once with `--llama-model <name>`.

## Chat templates

Some models need a modified chat template to work with llama.cpp. Keep that template in
git too: put it in `tests/chat-templates/` and name it in the profile.

```toml
[models.qwen3-5]
file = "Qwen3.5-9B-Q4_K_M.gguf"
chat_template = "qwen3.5.jinja"
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

## The code

| File | Role |
|---|---|
| `tests/conftest.py` | The fixtures, the `--llama-model` option, and the log-directory summary |
| `tests/llama_server_config.py` | Loading and checking the config files, picking the profile, building the command line, the port check |
| `tests/llama_server_process.py` | Starting the process, waiting for it, stopping it |
| `tests/test_llama_server_fixture.py` | Offline tests of all of the above, and the integration smoke test |

The offline tests never touch port 8001, your real `tests/llama-server.local.toml` or the
real environment, so they give the same result on every machine.
