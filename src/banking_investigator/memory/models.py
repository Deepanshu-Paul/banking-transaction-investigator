from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from banking_investigator.models.base import Base


@dataclass
class MemoryItem:
    """Domain representation of one memory item."""

    namespace: str
    key: str
    value: Any
    created_at: datetime
    updated_at: datetime


class Memory(Base):
    """SQLAlchemy model representing persisted agent memory."""

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    namespace: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    value: Mapped[Any] = mapped_column(
        JSON,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
