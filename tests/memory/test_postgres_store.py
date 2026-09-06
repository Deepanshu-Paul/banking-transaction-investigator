from uuid import uuid4

from banking_investigator.memory.postgres_store import PostgresMemoryStore


def test_store_and_retrieve_memory() -> None:
    store = PostgresMemoryStore()

    namespace = f"test:{uuid4()}"
    key = "usual_location"
    value = {"city": "Delhi"}

    try:
        stored = store.store(
            namespace=namespace,
            key=key,
            value=value,
        )

        assert stored.namespace == namespace
        assert stored.key == key
        assert stored.value == value

        retrieved = store.retrieve(
            namespace=namespace,
            key=key,
        )

        assert retrieved is not None
        assert retrieved.namespace == namespace
        assert retrieved.key == key
        assert retrieved.value == value

    finally:
        store.delete(
            namespace=namespace,
            key=key,
        )


def test_store_updates_existing_memory() -> None:
    store = PostgresMemoryStore()

    namespace = f"test:{uuid4()}"
    key = "usual_location"

    try:
        first = store.store(
            namespace=namespace,
            key=key,
            value={"city": "Delhi"},
        )

        updated = store.store(
            namespace=namespace,
            key=key,
            value={"city": "Mumbai"},
        )

        assert updated.value == {"city": "Mumbai"}
        assert updated.created_at == first.created_at
        assert updated.updated_at >= first.updated_at

        retrieved = store.retrieve(
            namespace=namespace,
            key=key,
        )

        assert retrieved is not None
        assert retrieved.value == {"city": "Mumbai"}

    finally:
        store.delete(
            namespace=namespace,
            key=key,
        )


def test_retrieve_returns_none_when_memory_does_not_exist() -> None:
    store = PostgresMemoryStore()

    namespace = f"test:{uuid4()}"

    result = store.retrieve(
        namespace=namespace,
        key="does_not_exist",
    )

    assert result is None


def test_delete_removes_memory() -> None:
    store = PostgresMemoryStore()

    namespace = f"test:{uuid4()}"
    key = "temporary_memory"

    store.store(
        namespace=namespace,
        key=key,
        value={"status": "temporary"},
    )

    store.delete(
        namespace=namespace,
        key=key,
    )

    result = store.retrieve(
        namespace=namespace,
        key=key,
    )

    assert result is None
