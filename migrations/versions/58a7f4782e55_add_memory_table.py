"""add memory table

Revision ID: 58a7f4782e55
Revises: 9c81811439f2
Create Date: 2026-09-05 20:34:58.821586

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "58a7f4782e55"
down_revision: Union[str, Sequence[str], None] = "9c81811439f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "memories",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "namespace",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "key",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "value",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_memories_namespace",
        "memories",
        ["namespace"],
        unique=False,
    )

    op.create_index(
        "ix_memories_key",
        "memories",
        ["key"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_memories_key",
        table_name="memories",
    )

    op.drop_index(
        "ix_memories_namespace",
        table_name="memories",
    )

    op.drop_table("memories")
