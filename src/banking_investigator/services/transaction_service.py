from typing import Any

from banking_investigator.repositories.transaction_repository import (
    TransactionRepository,
)


class TransactionService:
    def __init__(self, repository: TransactionRepository):
        self.repository = repository

    def get_transaction(self, transaction_id: str) -> dict[str, Any]:
        transaction = self.repository.find_by_id(transaction_id)

        if transaction is None:
            raise ValueError(
                f"Transaction '{transaction_id}' was not found."
            )

        return transaction