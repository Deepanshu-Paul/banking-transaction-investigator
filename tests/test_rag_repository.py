from datetime import datetime, timezone
from uuid import uuid4

import psycopg

from banking_investigator.config.settings import settings
from banking_investigator.repositories.rag_repository import RagRepository


def test_rag_repository_persists_document_and_chunk() -> None:
    repository = RagRepository()

    document_id = f"TEST-{uuid4().hex[:8]}"
    version = 1

    try:
        repository.create_document(
            document_id=document_id,
            version=version,
            title="Test Policy",
            source="test",
            document_type="policy",
            status="active",
            effective_from=datetime.now(timezone.utc),
            effective_to=None,
            content="Test banking policy document.",
            metadata={"region": "UK"},
        )

        chunk_id = repository.create_chunk(
            document_id=document_id,
            document_version=version,
            chunk_index=0,
            content="Test banking policy chunk.",
            token_count=5,
            metadata={"region": "UK"},
        )

        with psycopg.connect(
            settings.postgres_conn_string
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document_id, title, metadata
                    FROM rag_documents
                    WHERE document_id = %s
                    """,
                    (document_id,),
                )

                document = cur.fetchone()

                cur.execute(
                    """
                    SELECT chunk_id, content, token_count
                    FROM rag_chunks
                    WHERE chunk_id = %s
                    """,
                    (chunk_id,),
                )

                chunk = cur.fetchone()

        assert document is not None
        assert document[0] == document_id
        assert document[1] == "Test Policy"
        assert document[2]["region"] == "UK"

        assert chunk is not None
        assert str(chunk[0]) == chunk_id
        assert chunk[1] == "Test banking policy chunk."
        assert chunk[2] == 5

    finally:
        with psycopg.connect(
            settings.postgres_conn_string
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM rag_documents
                    WHERE document_id = %s
                    """,
                    (document_id,),
                )