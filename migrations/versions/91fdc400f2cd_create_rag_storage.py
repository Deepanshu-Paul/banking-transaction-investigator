"""create rag storage

Revision ID: 91fdc400f2cd
Revises: b4637492163b
Create Date: 2026-10-06

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "91fdc400f2cd"
down_revision: Union[str, Sequence[str], None] = "b4637492163b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.execute(
        "CREATE EXTENSION IF NOT EXISTS vector"
    )

    op.create_table(
        "rag_documents",
        sa.Column(
            "document_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "title",
            sa.String(length=500),
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.String(length=1000),
            nullable=False,
        ),
        sa.Column(
            "document_type",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "region",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "effective_from",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "effective_to",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
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
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="ck_rag_documents_validity_window",
        ),
        sa.PrimaryKeyConstraint(
            "document_id",
            "version",
        ),
    )

    op.create_index(
        "ix_rag_documents_status",
        "rag_documents",
        ["status"],
        unique=False,
    )

    op.create_index(
        "ix_rag_documents_document_type",
        "rag_documents",
        ["document_type"],
        unique=False,
    )

    op.create_index(
        "ix_rag_documents_region",
        "rag_documents",
        ["region"],
        unique=False,
    )

    op.create_index(
        "ix_rag_documents_metadata",
        "rag_documents",
        ["metadata"],
        unique=False,
    )

    op.create_table(
        "rag_chunks",
        sa.Column(
            "chunk_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "document_version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "chunk_index",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "token_count",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "embedding",
            sa.Text(),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "document_version"],
            [
                "rag_documents.document_id",
                "rag_documents.version",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.UniqueConstraint(
            "document_id",
            "document_version",
            "chunk_index",
            name="uq_rag_chunks_document_position",
        ),
    )

    op.create_index(
        "ix_rag_chunks_document",
        "rag_chunks",
        ["document_id", "document_version"],
        unique=False,
    )

    op.execute(
        """
        ALTER TABLE rag_chunks
        ALTER COLUMN embedding TYPE vector(1536)
        USING embedding::vector
        """
    )

    op.execute(
        """
        CREATE INDEX ix_rag_chunks_embedding_hnsw
        ON rag_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (
            m = 16,
            ef_construction = 64
        )
        """
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.execute(
        "DROP INDEX IF EXISTS ix_rag_chunks_embedding_hnsw"
    )

    op.drop_index(
        "ix_rag_chunks_document",
        table_name="rag_chunks",
    )

    op.drop_table("rag_chunks")

    op.drop_index(
        "ix_rag_documents_metadata",
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

    op.drop_index(
        "ix_rag_documents_status",
        table_name="rag_documents",
    )

    op.drop_table("rag_documents")