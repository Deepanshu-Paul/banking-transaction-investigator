"""add idempotency records

Revision ID: 0a3794419275
Revises: 3c755620d964
Create Date: 2026-08-28 20:56:23.246905

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0a3794419275"
down_revision: Union[str, Sequence[str], None] = "3c755620d964"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "idempotency_records",
        sa.Column(
            "operation_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "operation_type",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=30),
            nullable=False,
        ),
        sa.Column(
            "result",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("operation_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("idempotency_records")