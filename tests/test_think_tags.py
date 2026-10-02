"""Tests for thinking-tag detection (see docs/thinking-tags.md).

Fixtures are real /apply-template renders captured from llama-server where we
had the model loaded (gpt-oss, DeepSeek-R1-distill, Devstral with the actual
sentinels; Gemma and Qwen captured earlier with other sentinel words, which are
substituted textually here). Template sources are short excerpts. Cases with
"synthetic" in the name are built by hand because no such model was available.
"""

import re
import tempfile
import unittest
from unittest import mock

import requests

import chat
import ui
from chat import State, _auto_save, _think_override, _update_think_tags, handle_command
from config import DEFAULTS
from conversation import Conversation, strip_think
from llama_client import (
    INCONCLUSIVE,
    NONE,
    PAIR,
    LlamaClient,
    find_think_tags,
)

U, R, A = "QZUSERQZ", "QZREASONQZ", "QZANSWERQZ"

QWEN_CONT = f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>{R}</think>{A}"
QWEN_GEN = f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n"
# Excerpt of the Qwen template: the forced-open generation prompt and history stripping.
QWEN_SRC = (
    "{%- if '</think>' in content %}{%- set content = content.split('</think>')[-1] | trim %}{%- endif %}"
)

GEMMA_CONT = f"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{U}<turn|>\n<|turn>model\n<|channel>thought\n{R}<channel|>{A}"
GEMMA_GEN = f"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{U}<turn|>\n<|turn>model\n"
GEMMA_SRC = "synthetic source: {{ '<|channel>thought' }} ..."  # Gemma's real source was never captured

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

DEVSTRAL_CONT = f"[INST]{U}[/INST][THINK]{R}[/THINK]{A}"
DEVSTRAL_GEN = f"[INST]{U}[/INST]"
# Real head of Devstral's template: no thinking-related word anywhere in it.
DEVSTRAL_SRC = (
    "{#- Default system message if no system prompt is passed. #}\n"
    "{%- set default_system_message = '' %}\n\n{#- Begin of sequence token. #}\n{{- bos_token }}\n"
)

QWEN_PAIR = ("<think>", "</think>")
GEMMA_PAIR = ("<|channel>thought", "<channel|>")


class FindThinkTagsTests(unittest.TestCase):
    def check(self, cont, gen, src, expected):
        status, pair = find_think_tags(cont, gen, src)
        self.assertEqual((status, pair), expected)

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

    def test_synthetic_role_markers_are_not_thinking_openers(self):
        for gen in [
            f"<|user|>\n{U}<|end|>\n<|assistant|>",
            f"<|start_header_id|>user<|end_header_id|>\n\n{U}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ]:
            with self.subTest(gen=gen):
                self.check(f"{gen}{A}", gen, "<|assistant|> <|end_header_id|>", (NONE, None))

    def test_synthetic_closing_marker_at_the_end_is_not_an_opener(self):
        self.check(f"[INST]{U}[/INST]{A}", f"[INST]{U}[/INST]", "[//INST] think", (NONE, None))

    def test_synthetic_inst_style_opener_needs_its_mirror_in_the_source(self):
        cont = f"[INST]{U}{A}"
        self.check(cont, f"[INST]{U}[INST]", "[/INST] think", (PAIR, ("[INST]", "[/INST]")))
        self.check(cont, f"[INST]{U}[INST]", "think", (NONE, None))


class StripWithDetectedPairsTests(unittest.TestCase):
    def test_gpt_oss_reply_becomes_just_the_answer(self):
        self.assertEqual(strip_think(GPTOSS_REPLY, [GPTOSS_PAIR]), "Hi")

    def test_gemma_pair(self):
        text = "<|channel>thought\nreasoning\n<channel|>The answer."
        self.assertEqual(strip_think(text, [GEMMA_PAIR]), "The answer.")
        for untouched in ["a<channel|>b", "<|channel>thought only", "<channel|>x<|channel>thought"]:
            self.assertEqual(strip_think(untouched, [GEMMA_PAIR]), untouched)

    def test_empty_pairs_leave_text_untouched(self):
        self.assertEqual(strip_think(GPTOSS_REPLY, []), GPTOSS_REPLY)

    def test_each_pair_only_strips_its_own_tags(self):
        qwen_reply = "<think>\nq\n</think>\nQ answer"
        self.assertEqual(strip_think(qwen_reply, [GEMMA_PAIR]), qwen_reply)
        self.assertEqual(strip_think(qwen_reply, [GEMMA_PAIR, QWEN_PAIR]), "Q answer")


class MixedModelSaveTests(unittest.TestCase):
    def test_a_conversation_spanning_models_is_stripped_for_both(self):
        conv = Conversation()
        conv.add_user("q1")
        conv.add_assistant("<think>\nq\n</think>\nQwen answer")
        conv.add_user("q2")
        conv.add_assistant("<|channel>thought\ng\n<channel|>Gemma answer")
        conv.add_user("q3")
        conv.add_assistant("lone\n</think>\nstray")  # unbalanced: left alone
        conv.add_user("q4")
        conv.add_assistant("<|channel>thought\nno end")  # unbalanced: left alone
        with tempfile.TemporaryDirectory() as tmp:
            conv.save(tmp, name="mixed", omit_think=True, think_pairs=[QWEN_PAIR, GEMMA_PAIR])
            loaded, _ = Conversation.load(tmp, "mixed")
        replies = [m["content"] for m in loaded.messages if m["role"] == "assistant"]
        self.assertEqual(
            replies, ["Qwen answer", "Gemma answer", "lone\n</think>\nstray", "<|channel>thought\nno end"]
        )


class FakeResponse:
    def __init__(self, json_data=None, lines=()):
        self._json, self._lines = json_data, lines

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


class FakeServer:
    """Stands in for requests.get/post against llama-server's native endpoints."""

    def __init__(self, model_path, source, cont, gen, real_prompt="<real>"):
        self.model_path, self.source = model_path, source
        self.cont, self.gen, self.real_prompt = cont, gen, real_prompt
        self.fail = False
        self.bad_json = False
        self.props_calls = 0
        self.template_calls = []

    def become(self, model_path, source, cont, gen, real_prompt="<real>"):
        self.model_path, self.source = model_path, source
        self.cont, self.gen, self.real_prompt = cont, gen, real_prompt

    def get(self, url, **kw):
        assert url.endswith("/props"), url
        if self.fail:
            raise requests.ConnectionError("server down")
        self.props_calls += 1
        if self.bad_json:
            return FakeResponse(ValueError("not json"))
        return FakeResponse({"model_path": self.model_path, "chat_template": self.source})

    def post(self, url, json=None, **kw):
        if url.endswith("/completion"):
            return FakeResponse(lines=[b'data: {"content": "x", "stop": false}'])
        assert url.endswith("/apply-template"), url
        self.template_calls.append(json)
        if self.fail:
            raise requests.ConnectionError("server down")
        if json.get("continue_final_message"):
            return FakeResponse({"prompt": self.cont})
        messages = json["messages"]
        if len(messages) == 1 and messages[0]["content"] == U:
            return FakeResponse({"prompt": self.gen})
        return FakeResponse({"prompt": self.real_prompt})

    def patched(self):
        return mock.patch.multiple("requests", get=self.get, post=self.post)


def qwen_server():
    return FakeServer("/m/qwen.gguf", QWEN_SRC, QWEN_CONT, QWEN_GEN, QWEN_GEN.replace(U, "hello"))


def gemma_server():
    return FakeServer("/m/gemma.gguf", GEMMA_SRC, GEMMA_CONT, GEMMA_GEN, GEMMA_GEN.replace(U, "hello"))


def devstral_server():
    return FakeServer("/m/devstral.gguf", DEVSTRAL_SRC, DEVSTRAL_CONT, DEVSTRAL_GEN, DEVSTRAL_GEN)


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.client = LlamaClient("http://llama.test")

    def test_detects_and_reports_the_source(self):
        server = qwen_server()
        with server.patched():
            self.assertEqual(self.client.detect_think_tags(), (*QWEN_PAIR, "detected"))
        # The continuation render must ask for the continuation path explicitly.
        cont_call = next(c for c in server.template_calls if c.get("continue_final_message"))
        self.assertIs(cont_call["add_generation_prompt"], False)
        self.assertEqual(cont_call["messages"][1]["reasoning_content"], R)

    def test_accepted_pairs_are_memoized_by_server_identity(self):
        server = qwen_server()
        with server.patched():
            self.client.detect_think_tags()
            calls = len(server.template_calls)
            self.assertEqual(self.client.detect_think_tags(), (*QWEN_PAIR, "detected"))
            self.assertEqual(len(server.template_calls), calls)  # no extra /apply-template
            self.assertEqual(server.props_calls, 2)  # but /props is checked every call
            server.become("/m/gemma.gguf", GEMMA_SRC, GEMMA_CONT, GEMMA_GEN)  # a different model
            self.assertEqual(self.client.detect_think_tags(), (*GEMMA_PAIR, "detected"))
            self.assertGreater(len(server.template_calls), calls)

    def test_a_changed_template_with_the_same_model_path_is_noticed(self):
        server = qwen_server()
        with server.patched():
            self.client.detect_think_tags()
            calls = len(server.template_calls)
            server.source = QWEN_SRC + " edited"
            self.client.detect_think_tags()
            self.assertGreater(len(server.template_calls), calls)

    def test_a_none_result_is_not_memoized(self):
        server = devstral_server()
        with server.patched():
            self.assertIsNone(self.client.detect_think_tags())
            calls = len(server.template_calls)
            self.assertIsNone(self.client.detect_think_tags())
            self.assertGreater(len(server.template_calls), calls)  # recomputed

    def test_a_conclusive_none_clears_the_previous_pair(self):
        server = qwen_server()
        with server.patched():
            self.assertIsNotNone(self.client.detect_think_tags())
            server.become("/m/devstral.gguf", DEVSTRAL_SRC, DEVSTRAL_CONT, DEVSTRAL_GEN)
            self.assertIsNone(self.client.detect_think_tags())

    def test_server_errors_keep_the_previous_value_and_never_raise(self):
        server = qwen_server()
        with server.patched():
            before = self.client.detect_think_tags()
            server.fail = True
            self.assertEqual(self.client.detect_think_tags(), before)
            server.fail = False
            server.bad_json = True
            self.assertEqual(self.client.detect_think_tags(), before)

    def test_a_failed_anchor_keeps_the_previous_value(self):
        server = qwen_server()
        with server.patched():
            before = self.client.detect_think_tags()
            # new model whose renders do not line up and with nothing for layer 2
            server.become("/m/odd.gguf", "no words", QWEN_CONT.replace("<|im_start|>user", "X"), QWEN_GEN)
            self.assertEqual(self.client.detect_think_tags(), before)

    def test_the_config_override_wins_and_makes_no_server_calls(self):
        client = LlamaClient("http://llama.test", think_override=("[A]", "[/A]"))
        server = qwen_server()
        with server.patched():
            self.assertEqual(client.detect_think_tags(), ("[A]", "[/A]", "config"))
        self.assertEqual(server.props_calls, 0)
        self.assertEqual(server.template_calls, [])

    def test_override_needs_both_keys(self):
        self.assertIsNone(_think_override({"think_start": "<x>", "think_end": ""}))
        self.assertIsNone(_think_override({"think_start": "", "think_end": "</x>"}))
        self.assertIsNone(_think_override({}))
        self.assertEqual(_think_override({"think_start": "<x>", "think_end": "</x>"}), ("<x>", "</x>"))


class ChatPrefixTests(unittest.TestCase):
    def stream_text(self, client, server):
        with server.patched():
            stream = client.chat("m", [{"role": "user", "content": "hello"}])
            return list(stream), stream.think_tags

    def test_prefix_for_a_forced_open_template_but_not_for_self_opened_ones(self):
        client = LlamaClient("http://llama.test")
        self.assertEqual(self.stream_text(client, qwen_server())[0], ["<think>\n", "x"])
        self.assertEqual(self.stream_text(client, gemma_server())[0], ["x"])
        self.assertEqual(self.stream_text(client, devstral_server())[0], ["x"])

    def test_detection_runs_before_sending_so_a_swapped_model_is_not_judged_by_stale_tags(self):
        client = LlamaClient("http://llama.test")
        server = qwen_server()
        pieces, tags = self.stream_text(client, server)
        self.assertEqual((pieces, tags), (["<think>\n", "x"], (*QWEN_PAIR, "detected")))
        # The server is restarted with Gemma; the same client and no new detection call in between.
        server.become("/m/gemma.gguf", GEMMA_SRC, GEMMA_CONT, GEMMA_GEN, GEMMA_GEN.replace(U, "hello"))
        pieces, tags = self.stream_text(client, server)
        self.assertEqual((pieces, tags), (["x"], (*GEMMA_PAIR, "detected")))

    def test_a_closed_block_in_history_is_not_an_opening_tag(self):
        client = LlamaClient("http://llama.test")
        server = qwen_server()
        server.real_prompt = "<think>a</think>\nassistant\n"
        self.assertEqual(self.stream_text(client, server)[0], ["x"])


def render(callable_):
    with ui.console.capture() as cap:
        callable_()
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", cap.get()).split())


def make_state(**kw):
    config = {**DEFAULTS, "conversations_dir": "/nonexistent"}
    return State(model="m", config=config, context_length=None, **kw)


BRACKET = ("[THINK]", "[/THINK]", "detected")


class ConfigRenderingTests(unittest.TestCase):
    def test_think_tags_row_for_each_kind_of_value(self):
        config = {"default_model": "", "system_prompt": "", "llama_url": "u", "conversations_dir": "d"}
        cases = [
            ((*GEMMA_PAIR, "detected"), "<|channel>thought … <channel|> (detected)"),
            (BRACKET, "[THINK] … [/THINK] (detected)"),  # an unescaped [/THINK] would raise MarkupError
            (("<x>", "</x>", "config"), "<x> … </x> (config)"),
            (None, "none detected"),
        ]
        for tags, expected in cases:
            with self.subTest(tags=tags):
                out = render(lambda: ui.display_config(config, "m", None, tags))
                self.assertIn(expected, out)

    def test_config_command_shows_the_active_tags(self):
        state = make_state(think_tags=BRACKET)
        out = render(lambda: handle_command("/config", "", None, Conversation(), state))
        self.assertIn("think_tags", out)
        self.assertIn("[THINK] … [/THINK] (detected)", out)


class ToggleWarningTests(unittest.TestCase):
    def run_toggle(self, state, args="save_thinking off"):
        with mock.patch.object(chat, "display_info") as info:
            handle_command("/config", args, None, Conversation(), state)
        return info.call_args.args[0]

    def test_off_with_no_tags_warns(self):
        state = make_state()
        msg = self.run_toggle(state)
        self.assertEqual(
            msg,
            "save_thinking: off (no thinking tags known for the current model, so its replies will not be stripped)",
        )

    def test_the_warning_mentions_earlier_replies_when_pairs_were_seen(self):
        state = make_state(think_pairs_seen=[QWEN_PAIR])
        self.assertIn("; earlier replies with known tags still are)", self.run_toggle(state))

    def test_off_with_an_active_pair_is_plain(self):
        state = make_state(think_tags=(*QWEN_PAIR, "detected"), think_pairs_seen=[QWEN_PAIR])
        self.assertEqual(self.run_toggle(state), "save_thinking: off")

    def test_on_never_warns(self):
        self.assertEqual(self.run_toggle(make_state(), "save_thinking on"), "save_thinking: on")

    def test_a_swap_to_a_model_without_tags_is_flagged_at_the_next_toggle(self):
        state = make_state(think_tags=(*QWEN_PAIR, "detected"), think_pairs_seen=[QWEN_PAIR])
        with mock.patch.object(chat, "display_info"):
            _update_think_tags(state, None)
        self.assertIn("will not be stripped", self.run_toggle(state))


class ChangeAnnouncementTests(unittest.TestCase):
    def announce(self, state, tags, **kw):
        with mock.patch.object(chat, "display_info") as info:
            _update_think_tags(state, tags, **kw)
        return [c.args[0] for c in info.call_args_list]

    def test_a_pair_change_prints_exactly_one_line(self):
        state = make_state()
        self.assertEqual(self.announce(state, (*QWEN_PAIR, "detected"), announce=False), [])  # startup
        out = self.announce(state, (*GEMMA_PAIR, "detected"))
        self.assertEqual(len(out), 1)
        self.assertIn("thinking tags: <|channel>thought … <channel|> (detected)", out[0])

    def test_unchanged_or_source_only_changes_print_nothing(self):
        state = make_state(think_tags=(*QWEN_PAIR, "detected"), think_pairs_seen=[QWEN_PAIR])
        self.assertEqual(self.announce(state, (*QWEN_PAIR, "detected")), [])
        self.assertEqual(self.announce(state, (*QWEN_PAIR, "config")), [])

    def test_going_to_none_is_announced(self):
        state = make_state(think_tags=(*QWEN_PAIR, "detected"), think_pairs_seen=[QWEN_PAIR])
        self.assertEqual(self.announce(state, None), ["thinking tags: none detected for the current model"])
        self.assertEqual(self.announce(state, None), [])  # not again

    def test_pairs_seen_accumulate_without_duplicates(self):
        state = make_state()
        for tags in [(*QWEN_PAIR, "detected"), (*GEMMA_PAIR, "detected"), (*QWEN_PAIR, "config"), None]:
            self.announce(state, tags, announce=False)
        self.assertEqual(state.think_pairs_seen, [QWEN_PAIR, GEMMA_PAIR])

    def test_bracket_tags_are_announced_without_raising(self):
        state = make_state()
        out = render(lambda: _update_think_tags(state, BRACKET))
        self.assertIn("thinking tags: [THINK] … [/THINK] (detected)", out)

    def test_an_inconclusive_result_keeps_the_pair_and_prints_nothing(self):
        client = LlamaClient("http://llama.test")
        server = qwen_server()
        state = make_state()
        with server.patched():
            self.announce(state, client.detect_think_tags(), announce=False)
            server.fail = True
            self.assertEqual(self.announce(state, client.detect_think_tags()), [])
        self.assertEqual(state.think_tags, (*QWEN_PAIR, "detected"))


class SavePathsUseSeenPairsTests(unittest.TestCase):
    def test_save_and_autosave_strip_with_all_seen_pairs(self):
        conv = Conversation()
        conv.add_user("q")
        conv.add_assistant(GPTOSS_REPLY)
        with tempfile.TemporaryDirectory() as tmp:
            state = State(
                model="m",
                config={"conversations_dir": tmp, "auto_save": True, "save_thinking": False},
                context_length=None,
                auto_save_name="auto_t",
                think_pairs_seen=[GPTOSS_PAIR],
            )
            with mock.patch.object(chat, "display_info"):
                handle_command("/save", "manual", None, conv, state)
            _auto_save(conv, state)
            for name in ("manual", "auto_t"):
                loaded, _ = Conversation.load(tmp, name)
                self.assertEqual(loaded.messages[1]["content"], "Hi")
        self.assertEqual(conv.messages[1]["content"], GPTOSS_REPLY)  # memory untouched


if __name__ == "__main__":
    unittest.main()
