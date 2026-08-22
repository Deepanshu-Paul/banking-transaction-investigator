from typing import Any

import psycopg

from banking_investigator.config.settings import settings


class AccountRepository:
    def __init__(self):
        self.database_url = settings.database_url.replace("+psycopg", "")

    def find_by_id(self, account_id: str) -> dict[str, Any] | None:
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

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (account_id,))
                row = cur.fetchone()

        if row is None:
            return None

        return {
            "account_id": row[0],
            "customer_id": row[1],
            "account_type": row[2],
            "status": row[3],
            "created_at": row[4],
        }