import sys

sys.path.insert(0, "src")

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from banking_investigator.config.settings import settings


with PostgresSaver.from_conn_string(
    settings.postgres_conn_string
) as checkpointer:
    checkpointer.setup()


with PostgresStore.from_conn_string(
    settings.postgres_conn_string
) as store:
    store.setup()


print(
    "LangGraph checkpoint and store tables initialized successfully."
)