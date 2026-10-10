"""display_assistant_stream's label and interrupted parameters, and the ledger table."""

import threading

import duel
from tests.test_conversation_info import render_fn
from ui import COMMANDS, display_assistant_stream, display_ledger, print_help


def shown(tokens, **kwargs):
    box = []
    out = render_fn(lambda: box.append(display_assistant_stream(iter(tokens), **kwargs)))
    return out, box[0]


def interrupting(*tokens):
    yield from tokens
    raise KeyboardInterrupt


class TestStreamHeader:
    def test_the_default_header_is_assistant(self):
        out, text = shown(["hi"])
        assert out.startswith("Assistant:") and text == "hi"

    def test_a_label_replaces_it(self):
        out, _ = shown(["hi"], label="Side A:")
        assert out.startswith("Side A:") and "Assistant:" not in out


class TestInterrupt:
    def test_without_an_event_the_marker_is_appended(self):
        _, text = shown(interrupting("part"))
        assert text == "part [interrupted]"

    def test_with_an_event_it_is_set_and_nothing_is_appended(self):
        event = threading.Event()
        _, text = shown(interrupting("part"), interrupted=event)
        assert event.is_set() and text == "part"

    def test_an_unset_event_stays_unset_without_an_interrupt(self):
        event = threading.Event()
        shown(["a", "b"], interrupted=event)
        assert not event.is_set()


class TestLedgerTable:
    def test_empty(self):
        assert "No issues" in render_fn(lambda: display_ledger([]))

    def test_rows(self):
        ledger = duel.Ledger()
        ledger.raise_issue("a", "plot", "the quote", "too abrupt")
        ledger.raise_issue("b", "grammar", axis_known=False, unparsed=True)
        ledger.override_drop(1)
        out = render_fn(lambda: display_ledger(ledger.issues))
        assert "the quote" in out and "too abrupt" in out
        assert "Side B" in out and "Side A" in out
        assert "withdrawn (you)" in out and "grammar?" in out and "raised (no answer)" in out


def test_duel_is_a_command_and_in_the_help():
    assert "/duel" in COMMANDS
    assert "/duel" in render_fn(print_help)
