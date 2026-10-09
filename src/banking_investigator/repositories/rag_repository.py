from typing import Any
from uuid import UUID, uuid4

from banking_investigator.repositories.database import Database


class RagRepository:
    def __init__(self) -> None:
        self.db = Database()

    def create_document(
        self,
        document_id: str,
        version: int,
        title: str,
        source: str,
        document_type: str,
        status: str,
        effective_from,
        effective_to,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        query = """
            INSERT INTO rag_documents (
                document_id,
                version,
                title,
                source,
                document_type,
                status,
                region,
                effective_from,
                effective_to,
                content,
                metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
        """

        region = metadata.get("region")

        with __import__("psycopg").connect(
            self.db.database_url
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        document_id,
                        version,
                        title,
                        source,
                        document_type,
                        status,
                        region,
                        effective_from,
                        effective_to,
                        content,
                        __import__("psycopg").types.json.Jsonb(
                            metadata
                        ),
                    ),
                )

    def create_chunk(
        self,
        document_id: str,
        document_version: int,
        chunk_index: int,
        content: str,
        token_count: int,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        chunk_id = str(uuid4())

        query = """
            INSERT INTO rag_chunks (
                chunk_id,
                document_id,
                document_version,
                chunk_index,
                content,
                embedding,
                metadata,
                token_count
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            )
        """

        with __import__("psycopg").connect(
            self.db.database_url
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        chunk_id,
                        document_id,
                        document_version,
                        chunk_index,
                        content,
                        embedding,
                        __import__("psycopg").types.json.Jsonb(
                            metadata or {}
                        ),
                        token_count,
                    ),
                )

        return chunk_id