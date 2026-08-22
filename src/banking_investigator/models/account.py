from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from banking_investigator.models.base import Base


class Account(Base):
    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )

    customer_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("customers.customer_id"),
        nullable=False,
    )

    account_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )