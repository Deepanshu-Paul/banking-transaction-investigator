"""create dedicated RAG document and chunk storage

Revision ID: f2d3b8a10c47
Revises: b4637492163b
Create Date: 2026-09-27

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f2d3b8a10c47"
down_revision: Union[str, Sequence[str], None] = "b4637492163b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    """Create isolated storage for RAG documents and chunks."""

    # LangGraph semantic memory already relies on pgvector.
    # The RAG layer uses the same PostgreSQL extension but separate tables.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "rag_documents",
        sa.Column("document_id", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("source", sa.String(length=500), nullable=False),
        sa.Column("document_type", sa.String(length=50), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("region", sa.String(length=20), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("document_id", "version"),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="ck_rag_documents_effective_window",
        ),
    )

    op.create_index(
        "ix_rag_documents_document_type",
        "rag_documents",
        ["document_type"],
    )
    op.create_index(
        "ix_rag_documents_region",
        "rag_documents",
        ["region"],
    )
    op.create_index(
        "ix_rag_documents_status",
        "rag_documents",
        ["status"],
    )

    op.create_table(
        "rag_chunks",
        sa.Column("chunk_id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.String(length=100), nullable=False),
        sa.Column("document_version", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # Created as text first so the project does not need the Python
        # pgvector package just to run this migration.
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.ForeignKeyConstraint(
            ["document_id", "document_version"],
            ["rag_documents.document_id", "rag_documents.version"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "document_id",
            "document_version",
            "chunk_index",
            name="uq_rag_chunks_document_version_index",
        ),
    )

    # Use the PostgreSQL vector type after the table exists.
    op.execute(
        f"""
        ALTER TABLE rag_chunks
        ALTER COLUMN embedding TYPE vector({EMBEDDING_DIMENSIONS})
        USING CASE
            WHEN embedding IS NULL THEN NULL
            ELSE embedding::vector
        END
        """
    )

    op.create_index(
        "ix_rag_chunks_document",
        "rag_chunks",
        ["document_id", "document_version"],
    )

    op.execute(
        """
        CREATE INDEX ix_rag_chunks_embedding_hnsw
        ON rag_chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    """Drop only the RAG tables; keep the shared vector extension."""

    op.execute("DROP INDEX IF EXISTS ix_rag_chunks_embedding_hnsw")

    op.drop_index(
        "ix_rag_chunks_document",
        table_name="rag_chunks",
    )
    op.drop_table("rag_chunks")

    op.drop_index(
        "ix_rag_documents_status",
        table_name="rag_documents",
    )
    op.drop_index(
        "ix_rag_documents_region",
        table_name="rag_documents",
    )
    op.drop_index(
        "ix_rag_documents_document_type",
        table_name="rag_documents",
    )
    op.drop_table("rag_documents")
