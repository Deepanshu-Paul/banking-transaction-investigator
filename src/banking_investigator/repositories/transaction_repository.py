from typing import Any

import psycopg

from banking_investigator.config.settings import settings


class TransactionRepository:
    def __init__(self):
        self.database_url = settings.database_url.replace("+psycopg", "")

    def find_by_id(self, transaction_id: str) -> dict[str, Any] | None:
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

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (transaction_id,))
                row = cur.fetchone()

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