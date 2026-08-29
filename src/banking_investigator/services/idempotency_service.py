from typing import Any

from banking_investigator.repositories.idempotency_repository import (
    IdempotencyRepository,
)


class IdempotencyService:
    def __init__(self, repository: IdempotencyRepository):
        self.repository = repository

    def get_existing_operation(
        self,
        operation_id: str,
    ) -> dict[str, Any] | None:
        return self.repository.find_by_operation_id(
            operation_id
        )

    def claim_operation(
        self,
        operation_id: str,
        operation_type: str,
        lease_seconds: int = 60,
    ) -> bool:
        return self.repository.claim_operation(
            operation_id=operation_id,
            operation_type=operation_type,
            lease_seconds=lease_seconds,
        )

    def record_success(
        self,
        operation_id: str,
        result: str,
    ) -> None:
        self.repository.mark_success(
            operation_id=operation_id,
            result=result,
        )

    def record_failure(
        self,
        operation_id: str,
        result: str,
    ) -> None:
        self.repository.mark_failed(
            operation_id=operation_id,
            result=result,
        )