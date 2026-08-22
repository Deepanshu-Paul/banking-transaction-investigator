from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from banking_investigator.models.base import Base


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )

    account_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("accounts.account_id"),
        nullable=False,
    )

    amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    merchant: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    transaction_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )