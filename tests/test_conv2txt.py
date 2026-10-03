import ast
import sys
from pathlib import Path

from conv2txt import convert, separate_thinking, think_ends, valid_think_pair
from llama_client import is_valid_think_pair
from ui import separate_thinking as ui_separate_thinking


def conversation(*assistant_models):
    messages = []
    for i, model in enumerate(assistant_models, 1):
        messages.append({"role": "user", "timestamp": "2026-10-01T10:00:00", "content": f"question {i}"})
        reply = {"role": "assistant", "timestamp": "2026-10-01T10:00:05", "content": f"answer {i}"}
        if model:
            reply = {
                **{k: reply[k] for k in ("role", "timestamp")},
                "model": model,
                "content": reply["content"],
            }
        messages.append(reply)
    return {"model": "last.gguf", "system_prompt": "", "messages": messages}


class TestModelAnnotation:
    def test_a_reply_with_a_model_gets_a_model_line_before_its_text(self):
        text = convert(conversation("a.gguf"))
        lines = text.splitlines()
        i = lines.index("[model: a.gguf]")
        assert lines[i - 1].startswith("Assistant (")
        assert lines[i + 1] == "answer 1"

    def test_a_conversation_spanning_models_names_each_reply(self):
        text = convert(conversation("a.gguf", "b.gguf"))
        assert "[model: a.gguf]" in text
        assert "[model: b.gguf]" in text
        assert text.index("[model: a.gguf]") < text.index("[model: b.gguf]")

    def test_replies_without_a_model_look_exactly_as_before(self):
        text = convert(conversation(None))
        assert "[model:" not in text
        assert "Model: last.gguf" in text  # the file-level header is unchanged

    def test_user_messages_never_get_a_model_line(self):
        text = convert(conversation("a.gguf"))
        assert text.count("[model:") == 1


GEMMA_PAIR = ["<|channel>thought", "<channel|>"]


def with_reply(reply, think_pairs=None):
    data = {
        "model": "m",
        "system_prompt": "",
        "messages": [
            {"role": "user", "content": "q mentioning <channel|> literally"},
            {"role": "assistant", "content": reply},
        ],
    }
    if think_pairs is not None:
        data["think_pairs"] = think_pairs
    return data


class TestThinkingSeparation:
    def test_a_newline_is_added_after_a_recorded_closing_tag_and_survives_wrapping(self):
        text = convert(with_reply("<|channel>thought\nwhy<channel|>The answer.", [GEMMA_PAIR]))
        assert "why<channel|>\nThe answer." in text

    def test_user_messages_are_never_touched(self):
        text = convert(with_reply("a<channel|>b", [GEMMA_PAIR]))
        assert "q mentioning <channel|> literally" in text

    def test_without_an_applicable_pair_the_output_is_exactly_as_before(self):
        base = convert(with_reply("why<channel|>The answer."))
        assert convert(with_reply("why<channel|>The answer.", [])) == base
        assert convert(with_reply("why<channel|>The answer.", [["<think>", "</think>"]])) == base
        assert "why<channel|>The answer." in base

    def test_a_reply_that_already_has_a_newline_or_ends_at_the_tag_is_unchanged(self, subtests):
        pairs = [["<think>", "</think>"]]
        for reply in ("why</think>\n\nanswer", "only thinking</think>"):
            with subtests.test(reply=reply):
                assert convert(with_reply(reply, pairs)) == convert(with_reply(reply))

    def test_malformed_think_pairs_are_ignored(self, subtests):
        base = convert(with_reply("why<channel|>The answer."))
        for raw in (
            "<channel|>",
            {"a": "b"},
            None,
            [["the", "The"]],
            [["", ""]],
            [["<a>", "so the answer is"]],
            ["x"],
            [[1, 2]],
        ):
            with subtests.test(raw=raw):
                data = with_reply("why<channel|>The answer.")
                data["think_pairs"] = raw
                assert convert(data) == base

    def test_at_most_sixteen_pairs_are_used(self):
        raw = [[f"<t{i}>", f"</t{i}>"] for i in range(40)]
        assert len(think_ends({"think_pairs": raw})) == 16


class TestStandaloneAndParity:
    def test_separate_thinking_agrees_with_the_one_in_ui(self, subtests):
        samples = [
            ("why<channel|>144", ["<channel|>"]),
            ("a</think>b</think>\nc</think>", ["</think>"]),
            ("a<channel|>b</think>c", ["<channel|>", "</think>"]),
            ("a[/THINK]b", ["[/THINK]"]),
            ("nothing here", ["<channel|>"]),
            ("x<channel|>", ["<channel|>", ""]),
        ]
        for text, ends in samples:
            with subtests.test(text=text):
                assert separate_thinking(text, ends) == ui_separate_thinking(text, ends)

    def test_the_pair_check_agrees_with_is_valid_think_pair(self, subtests):
        pairs = [
            ("<think>", "</think>"),
            ("<|channel>thought", "<channel|>"),
            ("<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>"),
            ("[THINK]", "[/THINK]"),
            ("the", "the"),  # would put a newline after every "the"
            ("", ""),
            ("<think>", ""),
            ("a b", "</x>"),
            ("<x>", "so the answer is"),
            ("<x>", "Answer:"),
            ("<" + "a" * 58 + ">", "<" + "b" * 78 + ">"),  # exactly at the limits
            ("<" + "a" * 59 + ">", "</x>"),
            ("<x>", "<" + "b" * 79 + ">"),
            (1, 2),
            (None, "</x>"),
        ]
        for pair in pairs:
            with subtests.test(pair=pair):
                assert valid_think_pair(*pair) == is_valid_think_pair(*pair)

    def test_the_script_still_imports_only_the_standard_library(self):
        tree = ast.parse((Path(__file__).resolve().parent.parent / "conv2txt.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add((node.module or "").split(".")[0])
        assert imported, "expected some imports"
        assert imported - set(sys.stdlib_module_names) == set()
