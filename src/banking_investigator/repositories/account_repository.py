from typing import Any

from banking_investigator.repositories.database import Database
from banking_investigator.utils.deadline import Deadline


class AccountRepository:
    def __init__(self):
        self.db = Database()

    def find_by_id(
        self,
        account_id: str,
        deadline: Deadline | None = None,
    ) -> dict[str, Any] | None:

        query = """
            SELECT
                account_id,
                customer_id,
                account_type,
                status,
                created_at
            FROM accounts
            WHERE account_id = %s
        """

        row = self.db.fetch_one(
            query,
            (account_id,),
            deadline=deadline,
        )

        if row is None:
            return None

        return {
            "account_id": row[0],
            "customer_id": row[1],
            "account_type": row[2],
            "status": row[3],
            "created_at": row[4],
        }