import json
from datetime import datetime
from pathlib import Path


class Conversation:
    def __init__(self, system_prompt=""):
        self.system_prompt = system_prompt
        self.messages = []

    def add_user(self, content):
        self.messages.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def add_assistant(self, content):
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def clear(self):
        self.messages.clear()

    def summary(self):
        """Return a dict of conversation statistics."""
        user_msgs = [m for m in self.messages if m["role"] == "user"]
        asst_msgs = [m for m in self.messages if m["role"] == "assistant"]
        all_content = " ".join(m["content"] for m in self.messages)
        return {
            "messages": len(self.messages),
            "user_messages": len(user_msgs),
            "assistant_messages": len(asst_msgs),
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
            "content": (
                "[The following exchange is recalled from earlier "
                "in the conversation for context]"
            ),
            "timestamp": datetime.now().isoformat(),
        }
        self.messages.append(note)
        self.messages.append({"role": "user", "content": user_msg["content"]})
        if asst_msg:
            self.messages.append({"role": "assistant", "content": asst_msg["content"]})

    def get_messages(self):
        """Return messages list with system prompt prepended if set."""
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(self.messages)
        return msgs

    def save(self, conversations_dir, name=None, model=""):
        """Save conversation to a JSON file.

        Args:
            conversations_dir: Directory to save into (created if missing).
            name: Filename stem. Defaults to a timestamp.
            model: Current model name to store in the file.

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

        data = {
            "model": model,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
        }
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

        files = sorted(
            dirpath.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return [(f.stem, f) for f in files]
