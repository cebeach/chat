"""Token accounting: fits/near_limit, the /info breakdown, the client's counting calls and /system."""

import json
from unittest import mock

import pytest

import chat
import tokens
from chat import State, handle_command
from config import DEFAULTS
from conversation import Conversation
from llama_client import LlamaClient
from tests.helpers import qwen_server, render, words_template
from tests.test_conversation_info import render_fn


class TestFits:
    def test_the_prompt_must_leave_room_for_one_token(self):
        # llama-server rejects a prompt of n_ctx tokens or more
        assert tokens.fits(4095, 4096)
        assert not tokens.fits(4096, 4096)
        assert not tokens.fits(5000, 4096)

    def test_the_reserve_is_kept_free(self):
        assert tokens.fits(3095, 4096, reserve=1000)
        assert not tokens.fits(3096, 4096, reserve=1000)

    def test_near_limit_is_above_80_percent(self):
        assert not tokens.near_limit(209715, 262144)
        assert tokens.near_limit(209716, 262144)
        assert not tokens.near_limit(0, 262144)

    def test_a_swap_to_a_smaller_window_makes_the_same_prompt_warn(self):
        assert not tokens.near_limit(150000, 262144)
        assert tokens.near_limit(150000, 131072)

    def test_an_unknown_window_never_warns(self):
        assert not tokens.near_limit(10**9, None)
        assert not tokens.near_limit(10**9, 0)

    def test_refusal_names_the_numbers_and_the_reserve(self):
        text = tokens.refusal(5000, 4096)
        assert "5,000" in text and "4,096" in text and "kept free" not in text
        assert "1,000 kept free" in tokens.refusal(5000, 4096, reserve=1000)
        assert tokens.refusal(5000, 4096, outcome="System prompt not changed").startswith(
            "System prompt not changed"
        )


@pytest.fixture
def server():
    s = qwen_server()
    s.render = words_template
    return s


@pytest.fixture
def client(server):
    with server.patched():
        yield LlamaClient("http://llama.test")


class TestClientCounting:
    def test_count_tokens_sends_the_flags_the_completion_path_uses(self, server, client):
        assert client.count_tokens("one two three") == 4  # three words and the BOS
        assert server.tokenize_calls == [
            {"content": "one two three", "add_special": True, "parse_special": True}
        ]

    def test_a_fragment_can_be_counted_without_the_bos(self, server, client):
        assert client.count_tokens("one two three", add_special=False) == 3

    def test_prompt_tokens_counts_the_rendered_template(self, server, client):
        messages = [{"role": "user", "content": "hello there"}]
        assert client.prompt_tokens(messages) == len(words_template(messages).split()) + 1
        assert server.template_calls[-1]["messages"] == messages

    def test_the_model_is_passed_to_the_template_like_chat_does(self, server, client):
        client.prompt_tokens([{"role": "user", "content": "x"}], model="/m/qwen.gguf")
        assert server.template_calls[-1]["model"] == "/m/qwen.gguf"

    def test_check_fit_reads_the_server_once_and_returns_the_window(self, server, client):
        server.n_ctx = 2048
        needed, n_ctx = client.check_fit([{"role": "user", "content": "a b c"}])
        assert (needed, n_ctx) == (
            len(words_template([{"role": "user", "content": "a b c"}]).split()) + 1,
            2048,
        )
        assert server.props_calls == 1

    def test_a_following_chat_skips_its_own_refresh(self, server, client):
        messages = [{"role": "user", "content": "hello"}]
        client.check_fit(messages)
        client.chat("/m/qwen.gguf", messages, refreshed=True)
        assert server.props_calls == 1
        client.chat("/m/qwen.gguf", messages)
        assert server.props_calls == 2

    def test_a_server_that_is_down_raises_for_the_caller_to_handle(self, server, client):
        server.fail = True
        with pytest.raises(Exception, match="server down"):
            client.check_fit([{"role": "user", "content": "x"}])


class TestBreakdown:
    def test_parts_overhead_and_total(self, server, client):
        conv = Conversation(system_prompt="be brief")
        conv.add_user("one two three")
        conv.add_assistant("four five")
        counts = tokens.breakdown(client, conv)
        messages = conv.get_messages()
        total = len(words_template(messages).split()) + 1
        assert counts["system"] == 2 and counts["user"] == 3 and counts["assistant"] == 2
        assert counts["total"] == total
        assert counts["system"] + counts["user"] + counts["assistant"] + counts["overhead"] == total
        assert counts["n_ctx"] == 4096

    def test_the_parts_are_counted_without_the_bos_and_the_total_with_it(self, server, client):
        conv = Conversation()
        conv.add_user("a b")
        tokens.breakdown(client, conv)
        flags = {c["content"]: c["add_special"] for c in server.tokenize_calls}
        assert flags["a b"] is False
        assert [v for k, v in flags.items() if k.startswith("<user>")] == [True]

    def test_overhead_never_goes_below_zero(self, server, client):
        server.render = lambda messages: "x"  # a prompt smaller than its parts
        conv = Conversation()
        conv.add_user("a b c d")
        assert tokens.breakdown(client, conv)["overhead"] == 0

    def test_thinking_is_not_in_the_assistant_share(self, server, client):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant("answer", thinking="a very long chain of reasoning about the question")
        counts = tokens.breakdown(client, conv)
        assert counts["assistant"] == 1
        assert "reasoning" not in json.dumps(server.template_calls[-1])

    def test_an_empty_conversation_has_no_breakdown(self, client):
        assert tokens.breakdown(client, Conversation()) is None


def make_state(**config):
    return State(
        model="/m/qwen.gguf",
        config={**DEFAULTS, "conversations_dir": "/nonexistent", **config},
        context_length=4096,
    )


class TestInfoCommand:
    def test_info_shows_the_breakdown_and_the_window(self, server, client):
        conv = Conversation(system_prompt="be brief")
        conv.add_user("one two three")
        conv.add_assistant("four five")
        out = render_fn(lambda: handle_command("/info", "", client, conv, make_state()))
        assert "Tokens: system prompt 2" in out and "Tokens: your messages 3" in out
        assert "Tokens: AI replies 2" in out and "Prompt tokens" in out
        assert "Context window 4,096 tokens" in out and "Window used" in out

    def test_info_still_counts_with_context_check_off(self, server, client):
        conv = Conversation()
        conv.add_user("one two")
        out = render_fn(lambda: handle_command("/info", "", client, conv, make_state(context_check=False)))
        assert "Prompt tokens" in out

    def test_info_without_a_reachable_server_shows_the_rest(self, server, client):
        server.fail = True
        conv = Conversation()
        conv.add_user("one two")
        out = render_fn(lambda: handle_command("/info", "", client, conv, make_state()))
        assert "Messages 1" in out and "Prompt tokens" not in out and "Context window 4,096" in out


class TestSystemCommand:
    def run(self, client, conv, args, **config):
        state = make_state(**config)
        out = render(lambda: handle_command("/system", args, client, conv, state))
        return out

    def test_a_system_prompt_that_fits_is_set(self, server, client):
        conv = Conversation(system_prompt="old")
        assert "System prompt set." in self.run(client, conv, "new words here")
        assert conv.system_prompt == "new words here"

    def test_the_candidate_is_the_new_prompt_plus_the_stored_messages(self, server, client):
        conv = Conversation(system_prompt="old")
        conv.add_user("hello")
        self.run(client, conv, "brand new")
        assert server.template_calls[-1]["messages"] == [
            {"role": "system", "content": "brand new"},
            {"role": "user", "content": "hello"},
        ]

    def test_one_that_does_not_fit_is_refused_and_the_old_one_stays(self, server, client):
        server.n_ctx = 20
        conv = Conversation(system_prompt="old")
        out = self.run(client, conv, " ".join(["word"] * 40))
        assert "System prompt not changed" in out and "System prompt set" not in out
        assert conv.system_prompt == "old"

    def test_a_file_that_does_not_fit_is_refused_too(self, server, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "sys.txt").write_text(" ".join(["word"] * 40))
        server.n_ctx = 20
        conv = Conversation(system_prompt="old")
        out = self.run(client, conv, "sys.txt")
        assert "System prompt not changed" in out
        assert conv.system_prompt == "old" and conv.source_file is None

    def test_context_check_off_sets_it_anyway(self, server, client):
        server.n_ctx = 20
        conv = Conversation()
        self.run(client, conv, " ".join(["word"] * 40), context_check=False)
        assert conv.system_prompt.startswith("word")

    def test_a_server_that_cannot_be_asked_does_not_block_the_command(self, server, client):
        server.fail = True
        conv = Conversation()
        self.run(client, conv, "new")
        assert conv.system_prompt == "new"

    def test_the_window_in_force_now_is_used(self, server, client):
        conv = Conversation()
        self.run(client, conv, "w " * 30)  # fits 4096
        assert conv.system_prompt
        server.n_ctx = 20  # the server was restarted smaller
        out = self.run(client, conv, "x " * 30)
        assert "System prompt not changed" in out


class TestConfig:
    def test_a_stale_read_file_max_kb_in_the_users_file_is_ignored(self, tmp_path):
        import config

        path = tmp_path / "config.toml"
        path.write_text("read_file_max_kb = 1\ncontext_check = false\n")
        with mock.patch.object(config, "CONFIG_FILE", path):
            loaded = config.load_config()
        assert "read_file_max_kb" not in loaded
        assert loaded["context_check"] is False and loaded["reserve_output_tokens"] == 0

    def test_defaults(self):
        assert DEFAULTS["context_check"] is True and DEFAULTS["reserve_output_tokens"] == 0
        assert "read_file_max_kb" not in DEFAULTS

    def test_context_check_can_be_toggled_from_the_repl(self, server, client):
        state = make_state()
        render(lambda: handle_command("/config", "context_check off", client, Conversation(), state))
        assert state.config["context_check"] is False
        render(lambda: handle_command("/config", "context_check on", client, Conversation(), state))
        assert state.config["context_check"] is True
