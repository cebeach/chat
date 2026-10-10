"""The /duel session settings: config keys, validation and the reset on a project switch."""

import pytest

import duel
from chat import State, _initial_duel, _reset_session
from config import DEFAULTS
from tests.test_conversation_info import render_fn


def initial(**config):
    """(settings, what was printed) for _initial_duel with the given config.toml values."""
    box = []
    out = render_fn(lambda: box.append(_initial_duel({**DEFAULTS, **config})))
    return box[0], out


class TestInitialDuel:
    def test_defaults_are_opposed_non_empty_stances_and_the_five_axes(self):
        settings, out = initial()
        assert out == ""
        assert settings["axes"] == ["plot", "character", "dialogue", "pacing", "voice"]
        assert settings["stance_a"] and settings["stance_b"]
        assert settings["stance_a"] != settings["stance_b"]
        assert settings["context"] == "shared"
        assert all(settings[k] is None for k in ("temperature_a", "temperature_b", "seed_a", "seed_b"))

    def test_auto_starts_off(self):
        assert initial()[0]["auto"] is False

    def test_valid_values_pass(self):
        settings, out = initial(
            duel_axes=["pace", " tone "],
            duel_stance_a="Be terse.",
            duel_context="split",
            duel_temperature_a=0.2,
            duel_temperature_b=1,
            duel_seed_a=7,
        )
        assert out == ""
        assert settings["axes"] == ["pace", "tone"]
        assert settings["stance_a"] == "Be terse."
        assert settings["context"] == "split"
        assert (settings["temperature_a"], settings["temperature_b"], settings["seed_a"]) == (0.2, 1.0, 7)

    def test_a_random_seed_counts_as_unset(self):
        assert initial(duel_seed_b=-1)[0]["seed_b"] is None

    @pytest.mark.parametrize(
        "key, value, fallback_key",
        [
            ("duel_axes", [], "axes"),
            ("duel_axes", ["plot", ""], "axes"),
            ("duel_axes", "plot", "axes"),
            ("duel_stance_a", "  ", "stance_a"),
            ("duel_stance_b", 5, "stance_b"),
            ("duel_context", "pooled", "context"),
            ("duel_temperature_a", -1, "temperature_a"),
            ("duel_temperature_b", "hot", "temperature_b"),
            ("duel_seed_a", 1.5, "seed_a"),
            ("duel_seed_b", True, "seed_b"),
        ],
    )
    def test_each_bad_value_warns_and_falls_back(self, key, value, fallback_key):
        settings, out = initial(**{key: value})
        assert key in out and "ignored" in out
        assert settings[fallback_key] == initial()[0][fallback_key]

    def test_a_bad_value_does_not_spoil_the_others(self):
        settings, _ = initial(duel_axes=[], duel_context="split")
        assert settings["axes"] == duel.DEFAULT_AXES and settings["context"] == "split"

    def test_quiet_prints_nothing(self):
        out = render_fn(lambda: _initial_duel({**DEFAULTS, "duel_context": "x"}, quiet=True))
        assert out == ""


class TestSessionReset:
    def make_state(self, **config):
        config = {**DEFAULTS, **config}
        return State(model="m", config=config, context_length=4096, global_config=dict(config))

    def test_a_new_state_has_the_defaults(self):
        state = State(model="m", config={**DEFAULTS}, context_length=None)
        assert state.duel == initial()[0] and state.duel_last is None

    def test_a_project_switch_resets_duel_settings_and_the_last_record(self):
        state = self.make_state(duel_context="split")
        state.duel["context"] = "shared"
        state.duel["auto"] = True
        state.duel_last = {"ended": "normal"}
        _reset_session(state, {})
        assert state.duel["context"] == "split" and state.duel["auto"] is False
        assert state.duel_last is None

    def test_a_project_switch_does_not_repeat_the_warnings(self):
        state = self.make_state(duel_context="nonsense")
        assert render_fn(lambda: _reset_session(state, {})) == ""
