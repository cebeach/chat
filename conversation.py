import json
from datetime import datetime
from pathlib import Path


def split_think(text, tags):
    """Split one reply into (thinking, content) using its (start, end) thinking tags.

    The tags are not part of either result. Several blocks are joined with a
    blank line. A block that is never closed (a reply cut off by the token limit
    or the end of the context) takes everything after its opener, which leaves
    content empty. With no tags, or no opener in the text, the reply is returned
    untouched as ("", text).
    """
    if not tags:
        return "", text
    start, end = tags[:2]
    if not start or not end or start not in text:
        return "", text
    thinking, content = [], []
    pos = 0
    while True:
        opened = text.find(start, pos)
        if opened == -1:
            content.append(text[pos:])
            break
        content.append(text[pos:opened])
        body = opened + len(start)
        closed = text.find(end, body)
        if closed == -1:
            thinking.append(text[body:])
            break
        thinking.append(text[body:closed])
        pos = closed + len(end)
    return "\n\n".join(t.strip() for t in thinking if t.strip()), "".join(content).strip()


class Conversation:
    def __init__(self, system_prompt=""):
        self.system_prompt = system_prompt
        self.messages = []
        self.source_file = None

    def _add(self, role, content, source_file=None, model=None, thinking=None):
        msg = {
            "role": role,
            "timestamp": datetime.now().isoformat(),
        }
        if source_file is not None:
            msg["source_file"] = source_file
        if model:
            msg["model"] = model
        if thinking:
            msg["thinking"] = thinking
        msg["content"] = content  # Add content last
        self.messages.append(msg)

    def add_user(self, content, source_file=None):
        self._add("user", content, source_file=source_file)

    def add_assistant(self, content, model=None, thinking=None):
        """Add an assistant reply.

        model is the model that produced it, if known. thinking is the reasoning
        split off the reply (see split_think); it is stored only when non-empty,
        and content then holds the answer alone.
        """
        self._add("assistant", content, model=model, thinking=thinking)

    def clear(self):
        self.messages.clear()

    def summary(self):
        """Return a dict of conversation statistics."""
        all_content = " ".join(m["content"] for m in self.messages)
        return {
            "messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m["role"] == "user"),
            "assistant_messages": sum(1 for m in self.messages if m["role"] == "assistant"),
            "words": len(all_content.split()) if all_content.strip() else 0,
            "characters": sum(len(m["content"]) for m in self.messages),
        }

    def get_pair(self, pair_index):
        """Return the (user, assistant) message pair at the given 1-based index.

        Raises:
            IndexError: If the pair index is out of range.
        """
        user_msgs = [(i, m) for i, m in enumerate(self.messages) if m["role"] == "user"]
        if pair_index < 1 or pair_index > len(user_msgs):
            raise IndexError(f"Pair {pair_index} out of range (1-{len(user_msgs)})")
        msg_idx = user_msgs[pair_index - 1][0]
        user_msg = self.messages[msg_idx]
        # The assistant response follows the user message
        asst_msg = None
        if msg_idx + 1 < len(self.messages):
            candidate = self.messages[msg_idx + 1]
            if candidate["role"] == "assistant":
                asst_msg = candidate
        return user_msg, asst_msg

    def recall(self, pair_index):
        """Re-inject a user+assistant pair into the end of the conversation.

        Inserts a context note followed by the pair's messages at the end
        of the message list so they fall within the model's context window.

        Raises:
            IndexError: If the pair index is out of range.
        """
        user_msg, asst_msg = self.get_pair(pair_index)
        note = {
            "role": "user",
            "content": ("[The following exchange is recalled from earlier in the conversation for context]"),
            "timestamp": datetime.now().isoformat(),
        }
        self.messages.append(note)
        self.messages.append({"role": "user", "content": user_msg["content"]})
        if asst_msg:
            self.messages.append({"role": "assistant", "content": asst_msg["content"]})

    def get_messages(self):
        """Return messages list with system prompt prepended if set.

        Only returns 'role' and 'content' fields (not metadata like source_file),
        so the reasoning kept in 'thinking' is never sent back to the model. An
        assistant message with empty content (a reply that was only thinking) is
        left out together with the user message just before it, so no empty
        assistant turn reaches the chat template.
        """
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        for msg in self.messages:
            if msg["role"] == "assistant" and not msg["content"]:
                if msgs and msgs[-1]["role"] == "user":
                    msgs.pop()
                continue
            msgs.append({"role": msg["role"], "content": msg["content"]})
        return msgs

    def save(self, conversations_dir, name=None, model="", omit_thinking=False):
        """Save conversation to a JSON file.

        Args:
            conversations_dir: Directory to save into (created if missing).
            name: Filename stem. Defaults to a timestamp.
            model: Current model name to store in the file.
            omit_thinking: Leave the "thinking" field out of the assistant
                messages in the saved file. The in-memory messages are never
                modified.

        Returns:
            The Path of the saved file.
        """
        dirpath = Path(conversations_dir)
        dirpath.mkdir(parents=True, exist_ok=True)

        if not name:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize name — keep only safe characters
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        filepath = dirpath / f"{safe_name}.json"

        # Build data dict with source_file positioned after model
        data = {"model": model}
        if self.source_file is not None:
            data["source_file"] = self.source_file
        data["system_prompt"] = self.system_prompt
        messages = self.messages
        if omit_thinking:
            messages = [{k: v for k, v in m.items() if k != "thinking"} for m in messages]
        data["messages"] = messages
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    @classmethod
    def load(cls, conversations_dir, name):
        """Load a conversation from a JSON file.

        Args:
            conversations_dir: Directory containing saved conversations.
            name: Filename stem (without .json extension).

        Returns:
            A tuple of (Conversation, model_name).

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        filepath = Path(conversations_dir) / f"{name}.json"
        with open(filepath) as f:
            data = json.load(f)

        conv = cls(system_prompt=data.get("system_prompt", ""))
        conv.messages = data.get("messages", [])
        conv.source_file = data.get("source_file")
        return conv, data.get("model", "")

    @staticmethod
    def list_saved(conversations_dir):
        """List saved conversation names (sorted newest first).

        Returns:
            A list of (name, filepath) tuples.
        """
        dirpath = Path(conversations_dir)
        if not dirpath.exists():
            return []

        files = sorted(dirpath.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [(f.stem, f) for f in files]
