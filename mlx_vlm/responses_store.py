"""LRU response store for OpenAI Responses API previous_response_id support."""

import threading
from collections import OrderedDict
from typing import Any, Optional


class ResponseStore:
    """Bounded LRU store mapping response IDs to (input_items, response_object) pairs.

    Used to support the ``previous_response_id`` parameter in the Responses API,
    which allows clients to chain responses without resending full conversation
    history.

    Args:
        maxsize: Maximum number of responses to store. When exceeded, the oldest
                 entry is evicted. Defaults to 256.
    """

    def __init__(self, maxsize: int = 256):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def save(
        self,
        response_id: str,
        input_items: Any,
        response_output: list,
    ) -> None:
        """Save a response for later replay.

        Args:
            response_id: The unique response ID (e.g., ``"resp_abc123"``).
            input_items: The original request input (string or list of input items).
            response_output: The response output items list (dicts or model instances).
        """
        with self._lock:
            if response_id in self._store:
                self._store.move_to_end(response_id)
            self._store[response_id] = {
                "input": input_items,
                "output": response_output,
            }
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def get(self, response_id: str) -> Optional[dict]:
        """Retrieve a stored response by ID.

        Args:
            response_id: The response ID to look up.

        Returns:
            Dict with ``"input"`` and ``"output"`` keys, or ``None`` if not found.
        """
        with self._lock:
            entry = self._store.get(response_id)
            if entry is not None:
                self._store.move_to_end(response_id)
            return entry

    def replay_input(self, response_id: str) -> Optional[list]:
        """Build conversation input by replaying a previous response.

        Reconstructs input items from the stored response: the original input
        items followed by the output items converted to input format.

        Args:
            response_id: The previous response ID to replay.

        Returns:
            List of input items suitable for prepending to the current request,
            or ``None`` if the response ID is not found.
        """
        entry = self.get(response_id)
        if entry is None:
            return None

        items = []

        # Add original input items
        original_input = entry["input"]
        if isinstance(original_input, str):
            items.append({"role": "user", "content": original_input})
        elif isinstance(original_input, list):
            items.extend(original_input)

        # Convert output items to input format
        for output_item in entry.get("output", []):
            if isinstance(output_item, dict):
                item_type = output_item.get("type", "")
                if item_type == "message":
                    content = output_item.get("content", [])
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "output_text"
                        ):
                            items.append(
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": part.get("text", ""),
                                        }
                                    ],
                                }
                            )
                elif item_type == "function_call":
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": output_item.get("call_id", ""),
                            "name": output_item.get("name", ""),
                            "arguments": output_item.get("arguments", ""),
                        }
                    )

        return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Remove all stored responses."""
        with self._lock:
            self._store.clear()
