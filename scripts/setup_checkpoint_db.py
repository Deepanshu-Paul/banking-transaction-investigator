import sys

sys.path.insert(0, "src")

from langgraph.checkpoint.postgres import PostgresSaver

from banking_investigator.config.settings import settings


with PostgresSaver.from_conn_string(
    settings.postgres_conn_string
) as checkpointer:
    checkpointer.setup()

print("LangGraph checkpoint tables initialized successfully.")