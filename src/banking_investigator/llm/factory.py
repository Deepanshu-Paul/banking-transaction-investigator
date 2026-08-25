from banking_investigator.config.settings import settings
from banking_investigator.llm.base import LLMClient
from banking_investigator.llm.groq_client import GroqLLMClient
from banking_investigator.llm.openai_client import OpenAILLMClient


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "groq":
        return GroqLLMClient()

    if settings.llm_provider == "openai":
        return OpenAILLMClient()

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider}"
    )