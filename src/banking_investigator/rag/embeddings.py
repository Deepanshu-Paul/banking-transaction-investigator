from langchain_openai import OpenAIEmbeddings

from banking_investigator.config.settings import settings


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.openai_api_key,
)


def embed_text(text: str) -> list[float]:
    return embedding_model.embed_query(text)