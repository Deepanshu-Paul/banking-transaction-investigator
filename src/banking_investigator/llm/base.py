from abc import ABC, abstractmethod
from typing import Any, TypeVar


T = TypeVar("T")


class LLMClient(ABC):

    @abstractmethod
    def invoke(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
    ) -> Any:
        """Send messages to the configured LLM and return an AIMessage."""
        raise NotImplementedError

    @abstractmethod
    def invoke_structured(
        self,
        messages: list[Any],
        output_schema: type[T],
    ) -> T:
        """Send messages and return validated structured output."""
        raise NotImplementedError