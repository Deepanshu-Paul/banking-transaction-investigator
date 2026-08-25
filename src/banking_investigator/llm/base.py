from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):

    @abstractmethod
    def invoke(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
    ) -> Any:
        """Send messages to the configured LLM and return a standard response."""
        raise NotImplementedError