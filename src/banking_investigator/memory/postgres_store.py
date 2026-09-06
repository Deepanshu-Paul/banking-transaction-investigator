from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from banking_investigator.config.settings import settings
from banking_investigator.memory.models import Memory, MemoryItem
from banking_investigator.memory.store import MemoryStore


class PostgresMemoryStore(MemoryStore):
    """PostgreSQL-backed implementation of MemoryStore."""

    def __init__(self) -> None:
        self.engine = create_engine(settings.database_url)

    @staticmethod
    def _to_item(memory: Memory) -> MemoryItem:
        """Convert a database model into a domain memory item."""
        return MemoryItem(
            namespace=memory.namespace,
            key=memory.key,
            value=memory.value,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            expires_at=memory.expires_at,
        )

    def store(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> MemoryItem:
        """Create or update a memory with an optional TTL."""

        now = datetime.now(timezone.utc)

        expires_at = None

        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                raise ValueError("ttl_seconds must be greater than zero")

            expires_at = now + timedelta(seconds=ttl_seconds)

        with Session(self.engine) as session:
            statement = select(Memory).where(
                Memory.namespace == namespace,
                Memory.key == key,
            )

            memory = session.execute(statement).scalar_one_or_none()

            if memory is None:
                memory = Memory(
                    namespace=namespace,
                    key=key,
                    value=value,
                    created_at=now,
                    updated_at=now,
                    expires_at=expires_at,
                )
                session.add(memory)
            else:
                memory.value = value
                memory.updated_at = now

                if ttl_seconds is not None:
                    memory.expires_at = expires_at

            session.commit()
            session.refresh(memory)

            return self._to_item(memory)

    def retrieve(
        self,
        namespace: str,
        key: str,
    ) -> MemoryItem | None:
        """Retrieve a memory if it has not expired."""

        now = datetime.now(timezone.utc)

        with Session(self.engine) as session:
            statement = select(Memory).where(
                Memory.namespace == namespace,
                Memory.key == key,
            )

            memory = session.execute(statement).scalar_one_or_none()

            if memory is None:
                return None

            if (
                memory.expires_at is not None
                and memory.expires_at <= now
            ):
                return None

            return self._to_item(memory)

    def retrieve_namespace(
        self,
        namespace: str,
    ) -> list[MemoryItem]:
        """Retrieve all non-expired memories within a namespace."""

        now = datetime.now(timezone.utc)

        statement = (
            select(Memory)
            .where(
                Memory.namespace == namespace,
                (
                    (Memory.expires_at.is_(None))
                    | (Memory.expires_at > now)
                ),
            )
            .order_by(Memory.updated_at.desc())
        )

        with Session(self.engine) as session:
            memories = session.execute(statement).scalars().all()

            return [
                self._to_item(memory)
                for memory in memories
            ]

    def delete(
        self,
        namespace: str,
        key: str,
    ) -> None:
        """Delete a memory item."""

        with Session(self.engine) as session:
            statement = select(Memory).where(
                Memory.namespace == namespace,
                Memory.key == key,
            )

            memory = session.execute(statement).scalar_one_or_none()

            if memory is not None:
                session.delete(memory)
                session.commit()
