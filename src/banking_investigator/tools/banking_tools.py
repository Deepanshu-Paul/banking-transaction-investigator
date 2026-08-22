from typing import Any

from banking_investigator.repositories.transaction_repository import (
    TransactionRepository,
)
from banking_investigator.services.transaction_service import TransactionService


repository = TransactionRepository()
service = TransactionService(repository)


def get_transaction(transaction_id: str) -> dict[str, Any]:
    """
    Retrieve a banking transaction by its transaction ID.
    """
    return service.get_transaction(transaction_id)

from banking_investigator.repositories.account_repository import AccountRepository
from banking_investigator.services.account_service import AccountService


account_repository = AccountRepository()
account_service = AccountService(account_repository)


def get_account(account_id: str) -> dict[str, Any]:
    """
    Retrieve a banking account by its account ID.
    """
    return account_service.get_account(account_id)