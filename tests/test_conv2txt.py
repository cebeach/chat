import ast
import sys
from pathlib import Path

import conv2txt
from conv2txt import convert
from ui import THINK_DELIMITER


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


def with_reply(reply, thinking=None, extra=None):
    assistant = {"role": "assistant", "content": reply}
    if thinking is not None:
        assistant["thinking"] = thinking
    data = {
        "model": "m",
        "system_prompt": "",
        "messages": [{"role": "user", "content": "q mentioning <channel|> literally"}, assistant],
    }
    data.update(extra or {})
    return data


class TestThinkingSeparation:
    def test_thinking_is_written_before_the_content_with_a_delimiter(self):
        text = convert(with_reply("The answer.", thinking="why"))
        assert "why\n\n***\n\nThe answer." in text

    def test_a_reply_that_was_only_thinking_ends_with_the_delimiter(self):
        assert "only thinking\n\n***" in convert(with_reply("", thinking="only thinking"))

    def test_user_messages_are_never_touched(self):
        text = convert(with_reply("b", thinking="a"))
        assert "q mentioning <channel|> literally" in text

    def test_without_thinking_the_output_is_exactly_the_content(self):
        text = convert(with_reply("why<channel|>The answer."))
        assert "why<channel|>The answer." in text
        assert "***" not in text

    def test_think_pairs_in_the_file_are_ignored(self):
        base = convert(with_reply("why<channel|>The answer."))
        extra = {"think_pairs": [["<|channel>thought", "<channel|>"]]}
        assert convert(with_reply("why<channel|>The answer.", extra=extra)) == base


class TestStandaloneAndParity:
    def test_the_delimiter_agrees_with_the_one_in_ui(self):
        assert conv2txt.THINK_DELIMITER == THINK_DELIMITER

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
