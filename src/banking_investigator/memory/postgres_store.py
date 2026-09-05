from datetime import datetime, timezone
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

    def store(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> MemoryItem:
        """Create a new memory or update an existing one."""

        now = datetime.now(timezone.utc)

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
                )
                session.add(memory)
            else:
                memory.value = value
                memory.updated_at = now

            session.commit()
            session.refresh(memory)

            return MemoryItem(
                namespace=memory.namespace,
                key=memory.key,
                value=memory.value,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
            )

    def retrieve(
        self,
        namespace: str,
        key: str,
    ) -> MemoryItem | None:
        """Retrieve a memory item by namespace and key."""

        with Session(self.engine) as session:
            statement = select(Memory).where(
                Memory.namespace == namespace,
                Memory.key == key,
            )

            memory = session.execute(statement).scalar_one_or_none()

            if memory is None:
                return None

            return MemoryItem(
                namespace=memory.namespace,
                key=memory.key,
                value=memory.value,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
            )

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
