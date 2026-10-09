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
            %s, %s, %s, %s, %s, %s::vector, %s, %s
        )
        """
        embedding_value = None

        if embedding is not None:
            embedding_value = "[" + ",".join(
                str(value) for value in embedding
            ) + "]"
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
                        embedding_value,
                        __import__("psycopg").types.json.Jsonb(
                            metadata or {}
                        ),
                        token_count,
                    ),
                )

        return chunk_id

    def search_similar_chunks(
        self,
        embedding: list[float],
        top_k: int = 3,
        document_type: str | None = None,
        region: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(embedding) != 1536:
            raise ValueError(
                "Embedding must contain exactly 1536 dimensions"
            )

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        embedding_value = "[" + ",".join(
            str(value) for value in embedding
        ) + "]"

        conditions = [
            "c.embedding IS NOT NULL",
            "d.status = 'active'",
            "d.effective_from <= CURRENT_TIMESTAMP",
            "(d.effective_to IS NULL OR "
            "d.effective_to > CURRENT_TIMESTAMP)",
        ]

        params: list[Any] = [embedding_value]

        if document_type is not None:
            conditions.append("d.document_type = %s")
            params.append(document_type)

        if region is not None:
            conditions.append("d.region = %s")
            params.append(region)

        query = f"""
            SELECT
                c.chunk_id,
                c.document_id,
                c.document_version,
                d.title,
                c.chunk_index,
                c.content,
                c.metadata,
                1 - (c.embedding <=> %s::vector) AS similarity
            FROM rag_chunks AS c
            JOIN rag_documents AS d
              ON d.document_id = c.document_id
             AND d.version = c.document_version
            WHERE {" AND ".join(conditions)}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """

        params.extend([embedding_value, top_k])

        with __import__("psycopg").connect(
            self.db.database_url
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()

        return [
            {
                "chunk_id": str(row[0]),
                "document_id": row[1],
                "document_version": row[2],
                "title": row[3],
                "chunk_index": row[4],
                "content": row[5],
                "metadata": row[6],
                "similarity": float(row[7]),
            }
            for row in rows
        ]