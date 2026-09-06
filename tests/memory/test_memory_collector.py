from banking_investigator.memory.collector import MemoryCollector


def test_collect_account_profile_stores_verified_account_data() -> None:
    collector = MemoryCollector()

    account_data = {
        "account_id": "ACC1001",
        "customer_id": "CUST1001",
        "account_type": "checking",
        "status": "active",
        "created_at": "2026-09-06T10:00:00+00:00",
    }

    try:
        memory = collector.collect_account_profile(account_data)

        assert memory is not None
        assert memory.namespace == "customer:CUST1001"
        assert memory.key == "account_profile"
        assert memory.value == {
            "account_id": "ACC1001",
            "account_type": "checking",
            "status": "active",
        }

    finally:
        collector.memory_service.forget(
            namespace="customer:CUST1001",
            key="account_profile",
        )


def test_collect_account_profile_ignores_missing_customer_id() -> None:
    collector = MemoryCollector()

    account_data = {
        "account_id": "ACC1001",
        "account_type": "checking",
        "status": "active",
    }

    result = collector.collect_account_profile(account_data)

    assert result is None
