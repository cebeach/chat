"""Unknown keys in config.toml are reported, with the closest known key when there is one."""

import re
import sys
from unittest import mock

import pytest

import chat
import config
import ui
from config import DEFAULTS


def write_config(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def load_checked(tmp_path, text):
    with mock.patch.object(config, "CONFIG_FILE", write_config(tmp_path, text)):
        return config.load_config_checked()


class TestUnknownKeys:
    def test_the_hyphenated_spelling_suggests_the_underscore_key(self):
        # The reported case: `server-log-dir` in config.toml silently did nothing.
        assert config.unknown_keys({"server-log-dir": "~/x"}) == [("server-log-dir", "server_log_dir")]

    @pytest.mark.parametrize(
        ("key", "suggestion"),
        [
            ("temprature", "temperature"),
            ("Temperature", "temperature"),
            ("Server-Log-Dir", "server_log_dir"),
            ("zzqx", None),
        ],
    )
    def test_the_suggestion(self, key, suggestion):
        assert config.unknown_keys({key: 1}) == [(key, suggestion)]

    def test_known_keys_and_an_empty_file_report_nothing(self):
        assert config.unknown_keys(dict.fromkeys(DEFAULTS, 1)) == []
        assert config.unknown_keys({}) == []


class TestLoadConfigChecked:
    def test_a_bad_key_is_named_with_its_suggestion(self, tmp_path):
        loaded, problems = load_checked(tmp_path, 'server-log-dir = "~/x"\ntemperature = 0.2\n')
        assert problems == [
            'config.toml: unknown key "server-log-dir" is ignored; did you mean "server_log_dir"?'
        ]
        assert loaded["temperature"] == 0.2
        assert "server-log-dir" not in loaded
        assert loaded["server_log_dir"] == ""

    def test_a_key_with_no_close_match_has_no_suggestion(self, tmp_path):
        _, problems = load_checked(tmp_path, "zzqx = 1\n")
        assert problems == ['config.toml: unknown key "zzqx" is ignored.']

    def test_every_bad_key_is_reported_in_file_order(self, tmp_path):
        _, problems = load_checked(tmp_path, "zzqx = 1\ntemprature = 0.2\nseed = 3\n")
        assert [p.split('"')[1] for p in problems] == ["zzqx", "temprature"]

    def test_a_missing_file_gives_the_defaults(self, tmp_path):
        with mock.patch.object(config, "CONFIG_FILE", tmp_path / "none.toml"):
            assert config.load_config_checked() == (DEFAULTS, [])

    def test_load_config_is_the_config_of_load_config_checked(self, tmp_path):
        path = write_config(tmp_path, "zzqx = 1\ntop_p = 0.5\n")
        with mock.patch.object(config, "CONFIG_FILE", path):
            assert config.load_config() == config.load_config_checked()[0]


class TestMainReportsThem:
    def run_main(self, tmp_path, text):
        parent = mock.Mock()
        client = mock.Mock()
        client.is_available.return_value = False
        with (
            mock.patch.object(config, "CONFIG_FILE", write_config(tmp_path, text)),
            mock.patch.object(chat, "LlamaClient", return_value=client),
            mock.patch.object(sys, "argv", ["chat.py"]),
            mock.patch.object(chat, "display_warning") as warning,
            mock.patch.object(chat, "display_error") as error,
        ):
            parent.attach_mock(warning, "warning")
            parent.attach_mock(error, "error")
            with pytest.raises(SystemExit) as exit_info:
                chat.main()
        assert exit_info.value.code == 1
        return parent.mock_calls

    def test_the_warning_comes_before_the_connection_error(self, tmp_path):
        calls = self.run_main(tmp_path, 'server-log-dir = "~/x"\n')
        assert [c[0] for c in calls] == ["warning", "error"]
        assert 'did you mean "server_log_dir"?' in calls[0].args[0]

    def test_a_clean_file_warns_about_nothing(self, tmp_path):
        calls = self.run_main(tmp_path, "temperature = 0.2\n")
        assert [c[0] for c in calls] == ["error"]


def test_display_warning_prints_the_text():
    with ui.console.capture() as cap:
        ui.display_warning("config.toml: unknown key [x]")
    assert "config.toml: unknown key" in re.sub(r"\x1b\[[0-9;]*m", "", cap.get())
