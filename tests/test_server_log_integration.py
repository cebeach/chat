"""The server-log parser against a real llama-server's own log (guards against log-format drift).

The fixture launches the server with stdout and stderr in one file, as `2>&1 | tee` does.
Skipped unless a models directory is configured (docs/testing.md).
"""

import pytest

import server_log

pytestmark = pytest.mark.integration


def test_the_real_log_is_found_and_parsed(llama_server, llama_client):
    props = llama_client.props()
    logged = server_log.find_log(str(llama_server.log_dir), "", props, llama_server.url)
    assert logged, "the startup block did not match /props; has the log format changed?"
    path, summary = logged
    assert path.endswith("llama-server.log")
    block = server_log.parse_block(server_log.last_startup_block(path))
    # /props and the log print the same n_ctx_slot() value.
    assert block["n_ctx_slot"] == props["default_generation_settings"]["n_ctx"]
    assert block["n_slots"] == props["total_slots"]
    # One device buffer name (a CPU-only profile would have none and no figure, hence the guard).
    assert summary.get("device"), summary
    assert summary["model"] > 0
