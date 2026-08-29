"""add idempotency recovery fields

Revision ID: 9c81811439f2
Revises: 0a3794419275
Create Date: 2026-08-28 21:38:56.088999

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9c81811439f2"
down_revision: Union[str, Sequence[str], None] = "0a3794419275"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column(
        "idempotency_records",
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=True,
        ),
    )

    op.add_column(
        "idempotency_records",
        sa.Column(
            "lease_until",
            sa.DateTime(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_column(
        "idempotency_records",
        "lease_until",
    )

    op.drop_column(
        "idempotency_records",
        "updated_at",
    )