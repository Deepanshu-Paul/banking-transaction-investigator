from typing import Any

from banking_investigator.repositories.database import Database
from banking_investigator.utils.deadline import Deadline


class TransactionRepository:
    def __init__(self):
        self.db = Database()

    def find_by_id(
        self,
        transaction_id: str,
        deadline: Deadline | None = None,
    ) -> dict[str, Any] | None:

        query = """
            SELECT
                transaction_id,
                account_id,
                amount,
                currency,
                merchant,
                transaction_type,
                status,
                timestamp
            FROM transactions
            WHERE transaction_id = %s
        """

        row = self.db.fetch_one(
            query,
            (transaction_id,),
            deadline=deadline,
        )

        if row is None:
            return None

        return {
            "transaction_id": row[0],
            "account_id": row[1],
            "amount": row[2],
            "currency": row[3],
            "merchant": row[4],
            "transaction_type": row[5],
            "status": row[6],
            "timestamp": row[7],
        }