import ast
import json
import sys
from pathlib import Path

import conv2txt
from conv2txt import convert


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
    def test_thinking_is_omitted_by_default(self):
        text = convert(with_reply("The answer.", thinking="PRIVATE-REASONING"))
        assert "The answer." in text
        assert "PRIVATE-REASONING" not in text
        assert "END OF THINKING" not in text

    def test_kept_thinking_is_written_before_the_content_with_a_rule(self):
        text = convert(with_reply("The answer.", thinking="why"), keep_thinking=True)
        assert "why\n\n*** END OF THINKING ***\n\nThe answer." in text

    def test_a_kept_reply_that_was_only_thinking_ends_with_the_rule(self):
        text = convert(with_reply("", thinking="only thinking"), keep_thinking=True)
        assert text.rstrip().endswith("only thinking\n\n*** END OF THINKING ***")

    def test_a_reply_that_was_only_thinking_is_skipped_by_default(self):
        text = convert(with_reply("", thinking="only thinking"))
        assert "Assistant" not in text
        assert "only thinking" not in text
        assert "\n\n\n" not in text

    def test_an_empty_reply_without_thinking_is_still_listed(self):
        assert "Assistant:" in convert(with_reply(""))

    def test_the_rule_is_never_wrapped(self):
        text = convert(with_reply("The answer.", thinking="why"), keep_thinking=True, line_length=10)
        assert "\n*** END OF THINKING ***\n" in text

    def test_user_messages_are_never_touched(self):
        text = convert(with_reply("b", thinking="a"), keep_thinking=True)
        assert "q mentioning <channel|> literally" in text

    def test_without_thinking_the_output_is_exactly_the_content(self):
        text = convert(with_reply("why<channel|>The answer."), keep_thinking=True)
        assert "why<channel|>The answer." in text
        assert "***" not in text

    def test_think_pairs_in_the_file_are_ignored(self):
        base = convert(with_reply("why<channel|>The answer."))
        extra = {"think_pairs": [["<|channel>thought", "<channel|>"]]}
        assert convert(with_reply("why<channel|>The answer.", extra=extra)) == base


class TestKeepThinkingSwitch:
    def run(self, tmp_path, monkeypatch, capsys, *args):
        path = tmp_path / "c.json"
        path.write_text(json.dumps(with_reply("The answer.", thinking="PRIVATE-REASONING")))
        monkeypatch.setattr(sys, "argv", ["conv2txt.py", str(path), *args])
        conv2txt.main()
        return capsys.readouterr().out

    def test_off_by_default(self, tmp_path, monkeypatch, capsys):
        assert "PRIVATE-REASONING" not in self.run(tmp_path, monkeypatch, capsys)

    def test_the_switch_keeps_thinking(self, tmp_path, monkeypatch, capsys):
        out = self.run(tmp_path, monkeypatch, capsys, "--keep-thinking")
        assert "PRIVATE-REASONING\n\n*** END OF THINKING ***\n\nThe answer." in out


class TestStandaloneAndParity:
    def test_the_rule_is_the_documented_literal(self):
        assert conv2txt.THINK_RULE == "*** END OF THINKING ***"

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
