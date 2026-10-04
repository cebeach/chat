"""Tests for thinking-tag detection (see docs/thinking-tags.md).

Fixtures are real /apply-template renders captured from llama-server where we
had the model loaded (gpt-oss, DeepSeek-R1-distill, Devstral, Gemma and Qwen,
all with the actual sentinels; the Qwen and Gemma renders were re-checked
against the live servers). Template sources are verbatim excerpts, except where
a name says "synthetic". Cases with "synthetic" in the name are built by hand
because no such model was available.
"""

from unittest import mock

import pytest

import chat
import ui
from chat import State, _store_reply, _think_override, _update_think_tags, handle_command
from config import DEFAULTS
from conversation import Conversation, split_think
from llama_client import (
    INCONCLUSIVE,
    NONE,
    PAIR,
    LlamaClient,
    ThinkTagsError,
    find_think_tags,
    is_valid_think_pair,
)
from tests.helpers import (
    FakeServer,
    DEVSTRAL_CONT,
    DEVSTRAL_GEN,
    DEVSTRAL_SRC,
    GEMMA_CONT,
    GEMMA_GEN,
    GEMMA_PAIR,
    GEMMA_SRC,
    QWEN_CONT,
    QWEN_GEN,
    QWEN_PAIR,
    QWEN_SRC,
    A,
    R,
    U,
    devstral_server,
    gemma_server,
    qwen_server,
    render,
)

GPTOSS_HEAD = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\nCurrent date: 2026-10-01\n\nReasoning: medium\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
    f"<|start|>user<|message|>{U}<|end|><|start|>assistant"
)
GPTOSS_CONT = (
    f"{GPTOSS_HEAD}<|channel|>analysis<|message|>{R}<|end|><|start|>assistant<|channel|>final<|message|>{A}"
)
GPTOSS_GEN = GPTOSS_HEAD
GPTOSS_SRC = "# Valid channels: analysis, commentary, final. Channel must be included for every message."
GPTOSS_PAIR = ("<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>")
GPTOSS_REPLY = (
    '<|channel|>analysis<|message|>The user says: "Say hi". That is one word.<|end|>'
    "<|start|>assistant<|channel|>final<|message|>Hi"
)

DEEPSEEK_CONT = f"<｜User｜>{U}<｜Assistant｜><think>{R}</think>{A}"
DEEPSEEK_GEN = f"<｜User｜>{U}<｜Assistant｜>"
DEEPSEEK_SRC = "synthetic source excerpt: ... <think> ..."  # the real one contains "think"


class TestFindThinkTags:
    def check(self, cont, gen, src, expected):
        status, pair = find_think_tags(cont, gen, src)
        assert (status, pair) == expected

    def test_qwen_forced_open_falls_back_to_the_prompts_trailing_marker(self):
        self.check(QWEN_CONT, QWEN_GEN, QWEN_SRC, (PAIR, QWEN_PAIR))

    def test_gemma_start_is_marker_plus_word(self):
        self.check(GEMMA_CONT, GEMMA_GEN, GEMMA_SRC, (PAIR, GEMMA_PAIR))

    def test_gpt_oss_start_is_the_full_opener_and_end_is_several_markers(self):
        self.check(GPTOSS_CONT, GPTOSS_GEN, GPTOSS_SRC, (PAIR, GPTOSS_PAIR))

    def test_deepseek_emits_its_own_opening_tag(self):
        self.check(DEEPSEEK_CONT, DEEPSEEK_GEN, DEEPSEEK_SRC, (PAIR, QWEN_PAIR))

    def test_synthetic_bracket_tags_accepted_when_template_mentions_thinking(self):
        self.check(DEVSTRAL_CONT, DEVSTRAL_GEN, DEVSTRAL_SRC + " think ", (PAIR, ("[THINK]", "[/THINK]")))

    def test_devstral_family_guess_rejected_by_thinking_word_guard(self):
        # The server writes [THINK] for this template, but the template never mentions thinking.
        self.check(DEVSTRAL_CONT, DEVSTRAL_GEN, DEVSTRAL_SRC, (NONE, None))

    def test_synthetic_prose_between_sentinels_rejected(self):
        cont = f"<s>[INST]{U}[/INST]<think>{R} so the answer is {A}"
        self.check(cont, f"<s>[INST]{U}[/INST]", "think", (NONE, None))

    def test_synthetic_label_style_render_rejected(self):
        cont = f"User: {U}\nAssistant: Reasoning: {R}\nAnswer: {A}"
        self.check(cont, f"User: {U}\nAssistant:", "reasoning", (NONE, None))

    def test_missing_or_misordered_sentinels_are_a_conclusive_none(self):
        self.check(f"[INST]{U}[/INST]{A}", f"[INST]{U}[/INST]", "think", (NONE, None))
        self.check(f"x{A}y{R}z", "x", "think", (NONE, None))

    def test_failed_anchor_is_inconclusive(self):
        # e.g. a timestamp that changed between the two renders
        cont = QWEN_CONT.replace("<|im_start|>user", "<|im_start|>USER")
        self.check(cont, QWEN_GEN, "no thinking words here", (INCONCLUSIVE, None))

    def test_layer2_rescues_when_the_anchor_fails(self):
        cont = QWEN_CONT.replace("<|im_start|>user", "<|im_start|>USER")
        self.check(cont, QWEN_GEN, QWEN_SRC, (PAIR, QWEN_PAIR))

    def test_layer2_fallback_for_a_render_without_a_sentinel(self):
        # What a delimiter-style template produces: the continuation writes no reasoning.
        cont = f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n{A}"
        self.check(cont, QWEN_GEN, QWEN_SRC, (PAIR, QWEN_PAIR))
        self.check(cont, QWEN_GEN, "no mirror here", (NONE, None))

    def test_synthetic_role_markers_are_not_thinking_openers(self, subtests):
        for gen in [
            f"<|user|>\n{U}<|end|>\n<|assistant|>",
            f"<|start_header_id|>user<|end_header_id|>\n\n{U}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ]:
            with subtests.test(gen=gen):
                self.check(f"{gen}{A}", gen, "<|assistant|> <|end_header_id|>", (NONE, None))

    def test_synthetic_closing_marker_at_the_end_is_not_an_opener(self):
        self.check(f"[INST]{U}[/INST]{A}", f"[INST]{U}[/INST]", "[//INST] think", (NONE, None))

    def test_synthetic_inst_style_opener_needs_its_mirror_in_the_source(self):
        cont = f"[INST]{U}{A}"
        self.check(cont, f"[INST]{U}[INST]", "[/INST] think", (PAIR, ("[INST]", "[/INST]")))
        self.check(cont, f"[INST]{U}[INST]", "think", (NONE, None))


class TestIsValidThinkPair:
    def test_accepts_the_pairs_we_know(self, subtests):
        for pair in [QWEN_PAIR, GEMMA_PAIR, GPTOSS_PAIR, ("[THINK]", "[/THINK]")]:
            with subtests.test(pair=pair):
                assert is_valid_think_pair(*pair)

    def test_rejects_empty_whitespace_prose_and_non_strings(self, subtests):
        for pair in [
            ("", ""),
            ("<think>", ""),
            ("", "</think>"),
            ("a b", "</x>"),  # whitespace inside
            ("<think>", "so the answer is"),  # prose
            ("<think>", "Answer:"),  # a label, no marker
            ("plain", "</think>"),  # does not start with a marker
            (1, 2),
            (None, "</x>"),
            (["<think>"], "</think>"),
        ]:
            with subtests.test(pair=pair):
                assert not is_valid_think_pair(*pair)

    def test_length_limits(self):
        assert is_valid_think_pair("<" + "a" * 58 + ">", "<" + "b" * 78 + ">")  # 60 and 80
        assert not is_valid_think_pair("<" + "a" * 59 + ">", "</x>")  # 61
        assert not is_valid_think_pair("<x>", "<" + "b" * 79 + ">")  # 81


class TestSplitWithDetectedPairs:
    def test_gpt_oss_reply_becomes_thinking_and_the_answer(self):
        thinking, content = split_think(GPTOSS_REPLY, GPTOSS_PAIR)
        assert thinking == 'The user says: "Say hi". That is one word.'
        assert content == "Hi"

    def test_gemma_pair(self):
        text = "<|channel>thought\nreasoning\n<channel|>The answer."
        assert split_think(text, GEMMA_PAIR) == ("reasoning", "The answer.")
        assert split_think("a<channel|>b", GEMMA_PAIR) == ("", "a<channel|>b")  # a lone closer
        assert split_think("<|channel>thought only", GEMMA_PAIR) == ("only", "")  # never closed

    def test_no_tags_leave_the_text_untouched(self):
        assert split_think(GPTOSS_REPLY, None) == ("", GPTOSS_REPLY)

    def test_each_pair_only_splits_its_own_tags(self):
        qwen_reply = "<think>\nq\n</think>\nQ answer"
        assert split_think(qwen_reply, GEMMA_PAIR) == ("", qwen_reply)
        assert split_think(qwen_reply, QWEN_PAIR) == ("q", "Q answer")


class FakeStream:
    def __init__(self, think_tags, model="m"):
        self.think_tags = think_tags
        self.model = model


class TestStoreReply:
    """The turn stores the reply split with the tags detected for that turn."""

    def store(self, response, tags, conv=None):
        conv = conv or Conversation()
        conv.add_user("q")
        _store_reply(conv, response, FakeStream(tags), make_state())
        return conv.messages[-1]

    def test_a_reply_with_detected_tags_is_stored_split(self):
        msg = self.store(GPTOSS_REPLY, (*GPTOSS_PAIR, "detected"))
        assert (msg["thinking"], msg["content"]) == ('The user says: "Say hi". That is one word.', "Hi")
        assert msg["role"] == "assistant"

    def test_without_tags_the_reply_is_stored_whole(self):
        msg = self.store(GPTOSS_REPLY, None)
        assert msg["content"] == GPTOSS_REPLY
        assert "thinking" not in msg

    def test_an_unclosed_block_leaves_empty_content_and_the_rest_as_thinking(self):
        msg = self.store("<think>\nran out of tokens", (*QWEN_PAIR, "detected"))
        assert (msg["thinking"], msg["content"]) == ("ran out of tokens", "")

    def test_a_reply_without_thinking_has_no_thinking_key(self):
        msg = self.store("just an answer", (*QWEN_PAIR, "detected"))
        assert msg["content"] == "just an answer"
        assert "thinking" not in msg

    def test_a_conversation_spanning_models_is_split_per_turn(self, tmp_path):
        conv = Conversation()
        self.store("<think>\nq\n</think>\nQwen answer", (*QWEN_PAIR, "detected"), conv)
        self.store("<|channel>thought\ng\n<channel|>Gemma answer", (*GEMMA_PAIR, "detected"), conv)
        conv.save(str(tmp_path), name="mixed")
        loaded, _ = Conversation.load(str(tmp_path), "mixed")
        replies = [(m["thinking"], m["content"]) for m in loaded.messages if m["role"] == "assistant"]
        assert replies == [("q", "Qwen answer"), ("g", "Gemma answer")]


class TestDetection:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.client = LlamaClient("http://llama.test")

    def test_detects_and_reports_the_source(self):
        server = qwen_server()
        with server.patched():
            assert self.client.detect_think_tags() == (*QWEN_PAIR, "detected")
        # The continuation render must ask for the continuation path explicitly.
        cont_call = next(c for c in server.template_calls if c.get("continue_final_message"))
        assert cont_call["add_generation_prompt"] is False
        assert cont_call["messages"][1]["reasoning_content"] == R

    def test_accepted_pairs_are_memoized_by_server_identity(self):
        server = qwen_server()
        with server.patched():
            self.client.detect_think_tags()
            calls = len(server.template_calls)
            assert self.client.detect_think_tags() == (*QWEN_PAIR, "detected")
            assert len(server.template_calls) == calls  # no extra /apply-template
            assert server.props_calls == 2  # but /props is checked every call
            server.become("/m/gemma.gguf", GEMMA_SRC, GEMMA_CONT, GEMMA_GEN)  # a different model
            assert self.client.detect_think_tags() == (*GEMMA_PAIR, "detected")
            assert len(server.template_calls) > calls

    def test_a_changed_template_with_the_same_model_path_is_noticed(self):
        server = qwen_server()
        with server.patched():
            self.client.detect_think_tags()
            calls = len(server.template_calls)
            server.source = QWEN_SRC + " edited"
            self.client.detect_think_tags()
            assert len(server.template_calls) > calls

    def test_a_none_result_is_not_memoized(self):
        server = devstral_server()
        with server.patched():
            assert self.client.detect_think_tags() is None
            calls = len(server.template_calls)
            assert self.client.detect_think_tags() is None
            assert len(server.template_calls) > calls  # recomputed

    def test_a_conclusive_none_clears_the_previous_pair(self):
        server = qwen_server()
        with server.patched():
            assert self.client.detect_think_tags() is not None
            server.become("/m/devstral.gguf", DEVSTRAL_SRC, DEVSTRAL_CONT, DEVSTRAL_GEN)
            assert self.client.detect_think_tags() is None

    def test_server_errors_keep_the_previous_value_and_never_raise(self):
        server = qwen_server()
        with server.patched():
            before = self.client.detect_think_tags()
            server.fail = True
            assert self.client.detect_think_tags() == before
            server.fail = False
            server.bad_json = True
            assert self.client.detect_think_tags() == before

    def test_a_failed_anchor_keeps_the_previous_value(self):
        server = qwen_server()
        with server.patched():
            before = self.client.detect_think_tags()
            # new model whose renders do not line up and with nothing for layer 2
            server.become("/m/odd.gguf", "no words", QWEN_CONT.replace("<|im_start|>user", "X"), QWEN_GEN)
            assert self.client.detect_think_tags() == before

    ODD = ("/m/odd.gguf", "no words", QWEN_CONT.replace("<|im_start|>user", "X"), QWEN_GEN)

    def test_chat_refuses_after_a_swap_whose_tags_cannot_be_determined(self):
        server = qwen_server()
        with server.patched():
            assert self.client.detect_think_tags() is not None
            server.become(*self.ODD)
            posted = []
            orig_post = server.post
            server.post = lambda url, **kw: posted.append(url) or orig_post(url, **kw)
            with pytest.raises(ThinkTagsError, match=r"/m/odd\.gguf.*think_start"):
                self.client.chat("m", [{"role": "user", "content": "hi"}])
            assert not any(u.endswith("/completion") for u in posted)

    def test_it_keeps_refusing_until_the_tags_are_known_again(self):
        server = qwen_server()
        with server.patched():
            self.client.detect_think_tags()
            server.become(*self.ODD)
            for _ in range(2):
                with pytest.raises(ThinkTagsError):
                    self.client.chat("m", [{"role": "user", "content": "hi"}])
            server.become("/m/gemma.gguf", GEMMA_SRC, GEMMA_CONT, GEMMA_GEN)
            stream = self.client.chat("m", [{"role": "user", "content": "hi"}])
            assert stream.think_tags[:2] == GEMMA_PAIR

    def test_a_swap_back_to_the_settled_model_does_not_refuse(self):
        server = qwen_server()
        with server.patched():
            self.client.detect_think_tags()
            server.become(*self.ODD)
            self.client.detect_think_tags()
            server.become("/m/qwen.gguf", QWEN_SRC, QWEN_CONT, QWEN_GEN)
            assert self.client.chat("m", [{"role": "user", "content": "hi"}]).think_tags[:2] == QWEN_PAIR

    def test_no_swap_means_no_refusal(self):
        # The first detection of all has no previous model whose tags could be stale.
        server = FakeServer(*self.ODD)
        with server.patched():
            stream = self.client.chat("m", [{"role": "user", "content": "hi"}])
        assert stream.think_tags is None

    def test_the_config_override_is_never_refused(self):
        client = LlamaClient("http://llama.test", think_override=("<a>", "</a>"))
        server = qwen_server()
        with server.patched():
            client.detect_think_tags()
            server.become(*self.ODD)
            assert client.chat("m", [{"role": "user", "content": "hi"}]).think_tags == (
                "<a>",
                "</a>",
                "config",
            )

    def test_the_config_override_wins_and_makes_no_template_calls(self):
        client = LlamaClient("http://llama.test", think_override=("[A]", "[/A]"))
        server = qwen_server()
        with server.patched():
            assert client.detect_think_tags() == ("[A]", "[/A]", "config")
        # The model and context length are still read from /props (one GET), but
        # nothing is rendered for detection.
        assert server.props_calls == 1
        assert server.template_calls == []

    def test_override_needs_both_keys(self):
        assert _think_override({"think_start": "<x>", "think_end": ""}) is None
        assert _think_override({"think_start": "", "think_end": "</x>"}) is None
        assert _think_override({}) is None
        assert _think_override({"think_start": "<x>", "think_end": "</x>"}) == ("<x>", "</x>")


class TestChatPrefix:
    def stream_text(self, client, server):
        with server.patched():
            stream = client.chat("m", [{"role": "user", "content": "hello"}])
            return list(stream), stream.think_tags

    def test_prefix_for_a_forced_open_template_but_not_for_self_opened_ones(self):
        client = LlamaClient("http://llama.test")
        assert self.stream_text(client, qwen_server())[0] == ["<think>\n", "x"]
        assert self.stream_text(client, gemma_server())[0] == ["x"]
        assert self.stream_text(client, devstral_server())[0] == ["x"]

    def test_detection_runs_before_sending_so_a_swapped_model_is_not_judged_by_stale_tags(self):
        client = LlamaClient("http://llama.test")
        server = qwen_server()
        pieces, tags = self.stream_text(client, server)
        assert (pieces, tags) == (["<think>\n", "x"], (*QWEN_PAIR, "detected"))
        # The server is restarted with Gemma; the same client and no new detection call in between.
        server.become("/m/gemma.gguf", GEMMA_SRC, GEMMA_CONT, GEMMA_GEN, GEMMA_GEN.replace(U, "hello"))
        pieces, tags = self.stream_text(client, server)
        assert (pieces, tags) == (["x"], (*GEMMA_PAIR, "detected"))

    def test_a_closed_block_in_history_is_not_an_opening_tag(self):
        client = LlamaClient("http://llama.test")
        server = qwen_server()
        server.real_prompt = "<think>a</think>\nassistant\n"
        assert self.stream_text(client, server)[0] == ["x"]


def make_state(**kw):
    config = {**DEFAULTS, "conversations_dir": "/nonexistent"}
    return State(model="m", config=config, context_length=None, **kw)


BRACKET = ("[THINK]", "[/THINK]", "detected")


class TestConfigRendering:
    def test_think_tags_row_for_each_kind_of_value(self, subtests):
        config = {"system_prompt": "", "llama_url": "u", "conversations_dir": "d"}
        cases = [
            ((*GEMMA_PAIR, "detected"), "<|channel>thought … <channel|> (detected)"),
            (BRACKET, "[THINK] … [/THINK] (detected)"),  # an unescaped [/THINK] would raise MarkupError
            (("<x>", "</x>", "config"), "<x> … </x> (config)"),
            (None, "none detected"),
        ]
        for tags, expected in cases:
            with subtests.test(tags=tags):
                out = render(lambda: ui.display_config(config, "m", None, tags))
                assert expected in out

    def test_config_command_shows_the_active_tags(self):
        state = make_state(think_tags=BRACKET)
        out = render(lambda: handle_command("/config", "", None, Conversation(), state))
        assert "think_tags" in out
        assert "[THINK] … [/THINK] (detected)" in out


class TestToggleWarning:
    def run_toggle(self, state, args="save_thinking off"):
        with mock.patch.object(chat, "display_info") as info:
            handle_command("/config", args, None, Conversation(), state)
        return info.call_args.args[0]

    def test_off_with_no_tags_warns(self):
        state = make_state()
        msg = self.run_toggle(state)
        assert (
            msg == "save_thinking: off (no thinking tags known for the current model, "
            "so its replies cannot be separated and are saved whole)"
        )

    def test_off_with_an_active_pair_is_plain(self):
        state = make_state(think_tags=(*QWEN_PAIR, "detected"))
        assert self.run_toggle(state) == "save_thinking: off"

    def test_on_never_warns(self):
        assert self.run_toggle(make_state(), "save_thinking on") == "save_thinking: on"

    def test_a_swap_to_a_model_without_tags_is_flagged_at_the_next_toggle(self):
        state = make_state(think_tags=(*QWEN_PAIR, "detected"))
        with mock.patch.object(chat, "display_info"):
            _update_think_tags(state, None)
        assert "saved whole" in self.run_toggle(state)


class TestChangeAnnouncement:
    def announce(self, state, tags, **kw):
        with mock.patch.object(chat, "display_info") as info:
            _update_think_tags(state, tags, **kw)
        return [c.args[0] for c in info.call_args_list]

    def test_a_pair_change_prints_exactly_one_line(self):
        state = make_state()
        assert self.announce(state, (*QWEN_PAIR, "detected"), announce=False) == []  # startup
        out = self.announce(state, (*GEMMA_PAIR, "detected"))
        assert len(out) == 1
        assert "thinking tags: <|channel>thought … <channel|> (detected)" in out[0]

    def test_unchanged_or_source_only_changes_print_nothing(self):
        state = make_state(think_tags=(*QWEN_PAIR, "detected"))
        assert self.announce(state, (*QWEN_PAIR, "detected")) == []
        assert self.announce(state, (*QWEN_PAIR, "config")) == []

    def test_going_to_none_is_announced(self):
        state = make_state(think_tags=(*QWEN_PAIR, "detected"))
        assert self.announce(state, None) == ["thinking tags: none detected for the current model"]
        assert self.announce(state, None) == []  # not again

    def test_bracket_tags_are_announced_without_raising(self):
        state = make_state()
        out = render(lambda: _update_think_tags(state, BRACKET))
        assert "thinking tags: [THINK] … [/THINK] (detected)" in out

    def test_an_inconclusive_result_keeps_the_pair_and_prints_nothing(self):
        client = LlamaClient("http://llama.test")
        server = qwen_server()
        state = make_state()
        with server.patched():
            self.announce(state, client.detect_think_tags(), announce=False)
            server.fail = True
            assert self.announce(state, client.detect_think_tags()) == []
        assert state.think_tags == (*QWEN_PAIR, "detected")
