"""/set and the session's sampling options: the server's values, this session's overrides, validation.

/props reports the server's launch defaults, never what a request sent, so State.options holds only
the keys the user set; every value /set shows for the server is read from /props on each call.
"""

from unittest import mock

import pytest
import requests

import chat
import ui
from chat import State, _initial_options, _validated, handle_command
from config import DEFAULTS
from conversation import Conversation
from llama_client import LlamaClient
from tests.helpers import qwen_server, words_template
from tests.test_conversation_info import render_fn

OPTIONS = ["seed", "temperature", "top_p", "min_p", "repeat_penalty", "n_predict"]


@pytest.fixture
def server():
    s = qwen_server()
    s.render = words_template
    return s


@pytest.fixture
def client(server):
    with server.patched():
        yield LlamaClient("http://llama.test")


def make_state(**config):
    return State(model="/m/qwen.gguf", config={**DEFAULTS, **config}, context_length=4096)


def run(client, state, cmd, args=""):
    return render_fn(lambda: handle_command(cmd, args, client, Conversation(), state))


def set_(client, state, args=""):
    return run(client, state, "/set", args)


class TestListing:
    def test_every_option_is_listed_even_when_none_is_set(self, client):
        out = set_(client, make_state())
        for key in OPTIONS:
            assert key in out
        assert "Server value" in out and "Session override" in out

    def test_server_values_are_formatted(self, client):
        out = set_(client, make_state())
        # float32 noise is rounded and "random" is the unsigned 4294967295
        assert "0.8 -" in out and "0.95 -" in out and "0.05 -" in out
        assert "seed random -" in out
        assert "4294967295" not in out and "0.80000001" not in out

    def test_n_predict_is_never_reported_for_the_server(self, server, client):
        # /props reports a constant -1 here, whatever --predict the server was started with,
        # so showing it would claim "unlimited" for a server that limits replies.
        out = set_(client, make_state())
        assert "n_predict not reported -" in out and "unlimited" not in out
        server.params["n_predict"] = 64
        assert "n_predict not reported -" in set_(client, make_state())
        state = make_state()
        state.options["n_predict"] = 5
        assert "n_predict not reported 5" in set_(client, state)

    def test_an_override_is_shown_next_to_the_server_value(self, client):
        state = make_state()
        state.options["temperature"] = 0.2
        assert "temperature 0.8 0.2" in set_(client, state)

    def test_the_server_is_read_on_every_call(self, server, client):
        state = make_state()
        assert "temperature 0.8 -" in set_(client, state)
        server.params["temperature"] = 0.6000000238418579  # llama-server restarted with --temp 0.6
        assert "temperature 0.6 -" in set_(client, state)
        assert server.props_calls == 2

    def test_a_server_that_cannot_be_read_is_unavailable_never_guessed(self, server, client):
        server.fail = True
        out = set_(client, make_state())
        assert "temperature unavailable -" in out
        assert "0.8" not in out

    def test_no_hard_coded_defaults_remain(self):
        assert not hasattr(ui, "LLAMA_DEFAULTS")


class TestSetAndQuery:
    def test_setting_an_option_adds_only_that_key(self, client):
        state = make_state()
        assert "min_p set to 0.1." in set_(client, state, "min_p 0.1")
        assert state.options == {"min_p": 0.1}

    @pytest.mark.parametrize(
        "key, text, value",
        [
            ("seed", "7", 7),
            ("temperature", "0", 0.0),
            ("top_p", "0.9", 0.9),
            ("min_p", "0.05", 0.05),
            ("repeat_penalty", "1.1", 1.1),
            ("n_predict", "64", 64),
        ],
    )
    def test_each_option_parses_to_its_type(self, client, key, text, value):
        state = make_state()
        set_(client, state, f"{key} {text}")
        assert state.options == {key: value}
        assert type(state.options[key]) is type(value)

    def test_a_query_shows_the_server_value_when_nothing_is_set(self, client):
        assert "temperature: 0.8 (server)" in set_(client, make_state(), "temperature")

    def test_a_query_shows_both_when_overridden(self, client):
        state = make_state()
        state.options["temperature"] = 0.2
        out = set_(client, state, "temperature")
        assert "temperature: 0.2 (set for this session; the server's is 0.8)" in out

    def test_default_removes_the_override_and_names_the_servers_value(self, client):
        state = make_state()
        state.options.update(temperature=0.2, top_p=0.5)
        out = set_(client, state, "temperature default")
        assert state.options == {"top_p": 0.5}
        assert "the server's value (0.8) applies" in out

    def test_default_on_an_unset_option_is_harmless(self, client):
        state = make_state()
        set_(client, state, "temperature default")
        assert state.options == {}

    def test_an_unknown_option_lists_the_available_ones(self, client):
        out = set_(client, make_state(), "stop x")
        assert "Unknown option: stop" in out and "n_predict" in out

    def test_a_value_of_the_wrong_type_is_refused(self, client):
        state = make_state()
        assert "n_predict must be int (or 'default')." in set_(client, state, "n_predict many")
        assert state.options == {}

    def test_setting_needs_no_server(self, server, client):
        server.fail = True
        state = make_state()
        set_(client, state, "temperature 0.3")
        assert state.options == {"temperature": 0.3}


class TestValidation:
    @pytest.mark.parametrize(
        "key, text",
        [
            ("temperature", "-0.1"),
            ("temperature", "nan"),
            ("temperature", "inf"),
            ("top_p", "1.5"),
            ("top_p", "-0.1"),
            ("min_p", "2"),
            ("repeat_penalty", "0"),
            ("repeat_penalty", "-1"),
            ("n_predict", "0"),
            ("n_predict", "-1"),  # llama-server reads a request's -1 as "use the launch --predict"
            ("n_predict", "-2"),
            ("seed", "-2"),
            ("seed", "4294967296"),
        ],
    )
    def test_out_of_range_values_are_refused_and_nothing_is_stored(self, client, key, text):
        state = make_state()
        out = set_(client, state, f"{key} {text}")
        assert f"{key} must be" in out
        assert state.options == {}

    @pytest.mark.parametrize(
        "key, value",
        [
            ("temperature", 0),
            ("top_p", 1),
            ("top_p", 0),
            ("min_p", 1),
            ("repeat_penalty", 0.01),
            ("n_predict", 1),
            ("seed", -1),
            ("seed", 4294967295),
        ],
    )
    def test_the_boundaries_are_accepted(self, key, value):
        assert _validated(key, value) == value

    def test_a_refusal_keeps_the_previous_override(self, client):
        state = make_state()
        state.options["top_p"] = 0.5
        set_(client, state, "top_p 7")
        assert state.options == {"top_p": 0.5}


def initial(**config):
    """(options, what was printed) for _initial_options with the given config.toml values."""
    box = []
    out = render_fn(lambda: box.append(_initial_options({**DEFAULTS, **config})))
    return box[0], out


class TestInitialOptions:
    def test_unset_keys_are_absent(self):
        assert initial() == ({}, "")

    def test_valid_config_values_are_kept_with_their_type(self):
        options, out = initial(temperature=1, n_predict=128, seed=5)
        assert options == {"temperature": 1.0, "n_predict": 128, "seed": 5}
        assert type(options["temperature"]) is float
        assert out == ""

    @pytest.mark.parametrize(
        "key, value",
        [
            ("min_p", 5),
            ("temperature", "hot"),
            ("n_predict", "ten"),
            ("n_predict", 5.5),
            ("seed", True),
            ("top_p", False),
            ("repeat_penalty", 0),
            ("n_predict", 0),
            ("n_predict", -1),
        ],
    )
    def test_invalid_values_are_reported_by_key_and_dropped(self, key, value):
        options, out = initial(**{key: value})
        assert options == {}
        assert f"config.toml: {key} must be" in out and "ignored" in out

    def test_one_bad_key_does_not_lose_the_others(self):
        options, out = initial(min_p=5, top_p=0.9)
        assert options == {"top_p": 0.9}
        assert "min_p" in out


class TestConfigDisplay:
    def test_config_shows_the_same_rows_as_set(self, client):
        state = make_state(system_prompt="", llama_url="u", conversations_dir="d")
        state.options["temperature"] = 0.2
        out = run(client, state, "/config")
        assert "temperature 0.2 (set; server 0.8)" in out
        assert "top_p 0.95 (server)" in out and "seed random (server)" in out

    def test_config_without_a_client_does_not_guess(self):
        state = make_state(system_prompt="", llama_url="u", conversations_dir="d")
        out = run(None, state, "/config")
        assert "top_p unavailable (server)" in out


class TestPayload:
    def test_only_the_keys_the_user_set_are_sent(self, client):
        state = make_state()
        state.options.update(min_p=0.1, n_predict=5)
        with mock.patch("requests.post", wraps=requests.post) as post:
            client.chat("/m/qwen.gguf", [{"role": "user", "content": "hi"}], state.options)
        (sent,) = [c.kwargs["json"] for c in post.call_args_list if c.args[0].endswith("/completion")]
        assert {k: v for k, v in sent.items() if k not in ("model", "prompt")} == {
            "stream": True,
            "min_p": 0.1,
            "n_predict": 5,
        }


class TestReserve:
    def make(self, server, **options):
        state = make_state()
        state.options.update(options)
        return state

    def test_an_n_predict_override_is_kept_free_for_the_reply(self, server, client):
        server.n_ctx = 100
        messages = [{"role": "user", "content": " ".join(["w"] * 60)}]
        needed, _ = client.check_fit(messages)
        assert needed < 100
        assert chat._fits_window(client, messages, self.make(server))[0]
        ok, _ = chat._fits_window(client, messages, self.make(server, n_predict=100 - needed))
        assert not ok

    def test_the_larger_of_the_two_reserves_wins(self, server, client):
        server.n_ctx = 100
        messages = [{"role": "user", "content": " ".join(["w"] * 60)}]
        needed, _ = client.check_fit(messages)
        state = self.make(server, n_predict=5)
        state.config["reserve_output_tokens"] = 100 - needed
        assert not chat._fits_window(client, messages, state)[0]


class TestConfigFile:
    def test_every_option_defaults_to_unset(self):
        assert all(DEFAULTS[key] is None for key in OPTIONS)

    def test_toml_values_become_the_sessions_starting_overrides(self, tmp_path):
        import config

        path = tmp_path / "config.toml"
        path.write_text("min_p = 0.1\nrepeat_penalty = 1.1\nn_predict = 200\n")
        with mock.patch.object(config, "CONFIG_FILE", path):
            loaded = config.load_config()
        assert _initial_options(loaded) == {"min_p": 0.1, "repeat_penalty": 1.1, "n_predict": 200}
