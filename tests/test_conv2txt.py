import unittest

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


class ModelAnnotationTests(unittest.TestCase):
    def test_a_reply_with_a_model_gets_a_model_line_before_its_text(self):
        text = convert(conversation("a.gguf"))
        lines = text.splitlines()
        i = lines.index("[model: a.gguf]")
        self.assertTrue(lines[i - 1].startswith("Assistant ("))
        self.assertEqual(lines[i + 1], "answer 1")

    def test_a_conversation_spanning_models_names_each_reply(self):
        text = convert(conversation("a.gguf", "b.gguf"))
        self.assertIn("[model: a.gguf]", text)
        self.assertIn("[model: b.gguf]", text)
        self.assertLess(text.index("[model: a.gguf]"), text.index("[model: b.gguf]"))

    def test_replies_without_a_model_look_exactly_as_before(self):
        text = convert(conversation(None))
        self.assertNotIn("[model:", text)
        self.assertIn("Model: last.gguf", text)  # the file-level header is unchanged

    def test_user_messages_never_get_a_model_line(self):
        text = convert(conversation("a.gguf"))
        self.assertEqual(text.count("[model:"), 1)


if __name__ == "__main__":
    unittest.main()
