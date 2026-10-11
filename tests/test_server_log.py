"""Load-time VRAM from the llama-server log: parsing, matching, finding, and the one-shot rule."""

import glob
import os
import time
from pathlib import Path
from unittest import mock

import pytest

import chat
import server_log
from tests.test_turn_echo import FakeStream, run_session

FIXTURE = Path(__file__).parent / "server_log_gemma.txt"  # captured from a real run, paths scrubbed
MODEL = "/models/gemma-4-12B-it-uncensored-Q8_0.gguf"
PROPS = {
    "model_path": MODEL,
    "total_slots": 2,
    "default_generation_settings": {"n_ctx": 262144},
}
URL = "http://127.0.0.1:8001"


def load(path):
    return f"0.00.000.001 I llama_model_loader: loaded meta data with 52 key-value pairs and 667 tensors from {path} (version GGUF V3 (latest))"


def buf(kind, name, mib, tag="x"):
    return f"0.00.000.002 I {tag}: {name:>10} {kind} buffer size = {mib:8.2f} MiB"


def tail(slots=2, ctx=262144, port=8001, host="127.0.0.1"):
    return [
        f"0.04.000.001 I srv    load_model: initializing, n_slots = {slots}, n_ctx_slot = {ctx}, kv_unified = 'true'",
        f"0.04.000.002 I srv  llama_server: listening on http://{host}:{port}",
    ]


def write(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def block(*lines, **kw):
    return list(lines) + tail(**kw)


class TestRealLog:
    def test_summary(self):
        lines = server_log.last_startup_block(FIXTURE)
        assert server_log.matches(lines, PROPS, 8001)
        v = server_log.vram_summary(lines, PROPS)
        assert v["device"] == "CUDA0"
        assert v["model"] == pytest.approx(12067.72)  # CPU_Mapped (1020 MiB) is host, the fit probe is 0
        assert v["kv"] == pytest.approx(4352 + 510)  # iSWA: two KV lines summed
        assert v["compute"] == pytest.approx(2149.32)  # CUDA_Host compute is host memory


class TestParsing:
    def test_partial_offload_cpu_mapped_is_not_vram(self):
        lines = block(load(MODEL), buf("model", "CPU_Mapped", 1020), buf("model", "CUDA0", 100))
        assert server_log.vram_summary(lines, PROPS)["device"] == "CUDA0"

    def test_second_device_gives_no_figure(self):
        lines = block(load(MODEL), buf("model", "CUDA0", 100), buf("model", "CUDA1", 100))
        assert server_log.vram_summary(lines, PROPS) == {"multi_gpu": True}

    def test_no_device_buffer(self):
        lines = block(load(MODEL), buf("model", "CPU", 100), buf("KV", "CUDA_Host", 5))
        assert server_log.vram_summary(lines, PROPS) is None

    def test_fit_probe_and_draft_segments_are_ignored(self):
        lines = block(
            load(MODEL),
            buf("model", "CUDA0", 1),
            load(MODEL),
            buf("model", "CUDA0", 500),
            load("/models/draft.gguf"),
            buf("model", "CUDA0", 99),
        )
        assert server_log.vram_summary(lines, PROPS)["model"] == 500

    def test_compute_last_wins_and_kv_sums(self):
        lines = block(
            load(MODEL),
            buf("compute", "CUDA0", 10),
            buf("compute", "CUDA0", 30),
            buf("KV", "CUDA0", 4),
            buf("KV", "CUDA0", 5),
            buf("model", "CUDA0", 1),
            buf("model", "CUDA0", 2),
        )
        v = server_log.vram_summary(lines, PROPS)
        assert (v["compute"], v["kv"], v["model"]) == (30, 9, 3)

    def test_prefix_free_and_prefixed_lines(self):
        lines = block(
            load(MODEL),
            "llama_kv_cache:      CUDA0 KV buffer size =    64.00 MiB",
            "garbled buffer size = x MiB",
        )
        assert server_log.vram_summary(lines, PROPS)["kv"] == 64

    @pytest.mark.parametrize("lines", [[], [""], ["garbage"], tail()])
    def test_garbage_does_not_raise(self, lines):
        assert server_log.vram_summary(lines, PROPS) is None
        assert not server_log.matches(lines, PROPS, 8001)


class TestBlocks:
    def test_last_block_wins_and_unfinished_run_is_skipped(self, tmp_path):
        lines = block(load(MODEL), buf("model", "CUDA0", 1)) + block(load(MODEL), buf("model", "CUDA0", 2))
        lines += [load(MODEL), buf("model", "CUDA0", 3)]  # a run that never came up
        got = server_log.last_startup_block(write(tmp_path / "a.txt", lines))
        assert server_log.vram_summary(got, PROPS)["model"] == 2

    def test_no_complete_block(self, tmp_path):
        assert server_log.last_startup_block(write(tmp_path / "a.txt", [load(MODEL)])) is None
        assert server_log.last_startup_block(tmp_path / "missing.txt") is None


class TestMatches:
    @pytest.mark.parametrize(
        "kw",
        [{"slots": 4}, {"ctx": 1}, {"port": 8002}, {"host": "10.0.0.5"}],
    )
    def test_mismatch_rejects(self, kw):
        assert not server_log.matches(block(load(MODEL), **kw), PROPS, 8001)

    def test_other_model_rejects(self):
        assert not server_log.matches(block(load("/models/other.gguf")), PROPS, 8001)

    def test_basename_equal_paths_match(self):
        assert server_log.matches(block(load("rel/" + os.path.basename(MODEL))), PROPS, 8001)

    @pytest.mark.parametrize("host", ["localhost", "[::1]"])
    def test_loopback_hosts(self, host):
        assert server_log.matches(block(load(MODEL), host=host), PROPS, 8001)


class TestFindLog:
    def make(self, tmp_path, name, mtime, model=500, **kw):
        path = write(tmp_path / name, block(load(MODEL), buf("model", "CUDA0", model), **kw))
        os.utime(path, (mtime, mtime))
        return path

    def test_newest_matching_file_wins(self, tmp_path):
        now = time.time()
        self.make(tmp_path, "old.txt", now - 100, model=1)
        self.make(tmp_path, "new.log", now - 10, model=2)
        self.make(tmp_path, "newest.txt", now, model=3, port=9999)  # another server
        self.make(tmp_path, "ignored.md", now + 5, model=4)
        path, summary = server_log.find_log(str(tmp_path), "", PROPS, URL)
        assert (os.path.basename(path), summary["model"]) == ("new.log", 2)

    def test_env_and_home_expansion(self, tmp_path, monkeypatch):
        self.make(tmp_path, "a.txt", time.time())
        monkeypatch.setenv("CHAT_TEST_TMP", str(tmp_path.parent))
        got = server_log.find_log(f"${{CHAT_TEST_TMP}}/{tmp_path.name}", "", PROPS, URL)
        assert got and got[1]["model"] == 500

    def test_explicit_file_is_still_checked(self, tmp_path):
        path = self.make(tmp_path, "a.txt", time.time(), slots=9)
        assert server_log.find_log("", str(path), PROPS, URL) is None

    @pytest.mark.parametrize("url", ["http://192.168.1.5:8001", "http://example.com:8001"])
    def test_remote_server_is_never_looked_up(self, tmp_path, url):
        self.make(tmp_path, "a.txt", time.time())
        with mock.patch("os.scandir") as scandir:
            assert server_log.find_log(str(tmp_path), "", PROPS, url) is None
        scandir.assert_not_called()

    def test_not_configured_and_missing_dir(self, tmp_path):
        assert server_log.find_log("", "", PROPS, URL) is None
        assert server_log.find_log(str(tmp_path / "nope"), "", PROPS, URL) is None

    def test_only_the_newest_files_are_read(self, tmp_path):
        now = time.time()
        for i in range(server_log.MAX_FILES + 5):
            self.make(tmp_path, f"f{i:02}.txt", now - i, model=i + 1, port=9999)
        self.make(tmp_path, "match.txt", now - 1000)  # older than the cap
        assert server_log.find_log(str(tmp_path), "", PROPS, URL) is None


class TestOneShot:
    def run(self, tmp_path, inputs):
        logdir = tmp_path / "logs"
        logdir.mkdir()
        os.utime(write(logdir / "a.txt", FIXTURE.read_text().splitlines()), None)
        touched, startup = [], []

        def spy(name, real):
            def wrapper(path, *a, **kw):
                if str(logdir) in str(path):
                    touched.append(name)
                return real(path, *a, **kw)

            return wrapper

        real_load = chat.load_vram

        def load_then_mark(*a, **kw):
            real_load(*a, **kw)
            startup.append(len(touched))

        shown = []

        def setup(client):
            client.base_url = URL
            client.props.return_value = PROPS

        with (
            mock.patch("builtins.open", spy("open", open)),
            mock.patch("os.scandir", spy("scandir", os.scandir)),
            mock.patch("os.listdir", spy("listdir", os.listdir)),
            mock.patch("os.stat", spy("stat", os.stat)),
            mock.patch("glob.glob", spy("glob", glob.glob)),
            mock.patch.object(Path, "iterdir", lambda self: touched.append("iterdir") or iter(())),
            mock.patch.object(chat, "load_vram", load_then_mark),
            mock.patch.object(chat, "display_conversation_info", lambda *a, **kw: shown.append(kw["vram"])),
            mock.patch.object(chat.tokens, "breakdown", side_effect=ValueError),
        ):
            run_session(
                tmp_path,
                lambda: FakeStream(["r"]),
                inputs=inputs,
                config={"server_log_dir": str(logdir)},
                setup=setup,
            )
        return touched, startup, shown

    def test_log_is_read_at_startup_and_never_again(self, tmp_path):
        touched, startup, shown = self.run(tmp_path, ("/info", "hello", "/info", None))
        assert startup[0] > 0
        assert len(touched) == startup[0]
        assert len(shown) == 2 and shown[0] == shown[1]
        assert shown[0].startswith("CUDA0 model 11.8 GiB, KV 4.7, compute 2.1")
        assert "may belong to an earlier run" in shown[0]

    def test_off_when_not_configured(self, tmp_path):
        shown = []
        with (
            mock.patch.object(chat, "display_conversation_info", lambda *a, **kw: shown.append(kw["vram"])),
            mock.patch.object(chat.tokens, "breakdown", side_effect=ValueError),
        ):
            session = run_session(tmp_path, lambda: FakeStream(["r"]), inputs=("/info", None))
        assert shown == [None]
        session.client.props.assert_not_called()


class TestModelChange:
    def test_figure_is_dropped_not_rescraped(self):
        state = chat.State(model="a", config={}, context_length=1)
        state.vram, state.vram_log = {"device": "CUDA0"}, "x.txt"
        chat._sync_server_info(state, "b", 1, announce=False)
        assert state.vram is None
        assert chat._vram_row(state) == "not available (model changed since startup)"

    def test_no_row_when_there_was_never_a_figure(self):
        state = chat.State(model="a", config={}, context_length=1)
        chat._sync_server_info(state, "b", 1, announce=False)
        assert chat._vram_row(state) is None

    def test_multi_gpu_row(self):
        state = chat.State(model="a", config={}, context_length=1, vram={"multi_gpu": True})
        assert chat._vram_row(state) == "multi-GPU not supported"
