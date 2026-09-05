from typing import Any

import psycopg
from psycopg.errors import QueryCanceled
from banking_investigator.services.errors import RetryableError
from banking_investigator.config.settings import settings


class TransactionRepository:
    def __init__(self):
        self.database_url = settings.database_url.replace("+psycopg", "")

    def _fetch_one(
        self,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> tuple[Any, ...] | None:
        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SET statement_timeout = "
                        f"{settings.db_statement_timeout_ms}"
                    )

                    cur.execute(query, params)
                    return cur.fetchone()

        except QueryCanceled as exc:
            raise RetryableError(
                f"Database query timed out after "
                f"{settings.db_statement_timeout_ms} ms"
            ) from exc

    def find_by_id(
        self,
        transaction_id: str,
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

        row = self._fetch_one(query, (transaction_id,))

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