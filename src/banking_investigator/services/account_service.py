from typing import Any

from banking_investigator.repositories.account_repository import (
    AccountRepository,
)
from banking_investigator.utils.deadline import Deadline


class AccountService:
    def __init__(self, repository: AccountRepository):
        self.repository = repository

    def get_account(
        self,
        account_id: str,
        deadline: Deadline | None = None,
    ) -> dict[str, Any]:
        account = self.repository.find_by_id(
            account_id,
            deadline=deadline,
        )

        if account is None:
            raise ValueError(
                f"Account '{account_id}' was not found."
            )

        return account