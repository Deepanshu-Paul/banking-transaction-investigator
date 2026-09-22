import sys

sys.path.insert(0, "src")

from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from banking_investigator.config.settings import settings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.openai_api_key,
)


with PostgresSaver.from_conn_string(
    settings.postgres_conn_string
) as checkpointer:
    checkpointer.setup()


with PostgresStore.from_conn_string(
    settings.postgres_conn_string,
    index={
        "dims": 1536,
        "embed": embeddings,
        "fields": ["text"],
    },
) as store:
    store.setup()


print(
    "LangGraph checkpoint and store tables initialized successfully."
)
