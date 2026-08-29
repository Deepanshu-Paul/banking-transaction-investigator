from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg

from banking_investigator.config.settings import settings


class IdempotencyRepository:
    def __init__(self):
        self.database_url = settings.database_url.replace(
            "+psycopg",
            "",
        )

    def find_by_operation_id(
        self,
        operation_id: str,
    ) -> dict[str, Any] | None:
        query = """
            SELECT
                operation_id,
                operation_type,
                status,
                result,
                created_at,
                updated_at,
                lease_until
            FROM idempotency_records
            WHERE operation_id = %s
        """

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (operation_id,))
                row = cur.fetchone()

        if row is None:
            return None

        return {
            "operation_id": row[0],
            "operation_type": row[1],
            "status": row[2],
            "result": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "lease_until": row[6],
        }

    def claim_operation(
        self,
        operation_id: str,
        operation_type: str,
        lease_seconds: int = 60,
    ) -> bool:
        now = datetime.now(timezone.utc)
        lease_until = now + timedelta(seconds=lease_seconds)

        query = """
            INSERT INTO idempotency_records (
                operation_id,
                operation_type,
                status,
                result,
                created_at,
                updated_at,
                lease_until
            )
            VALUES (
                %s,
                %s,
                'PROCESSING',
                NULL,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP,
                %s
            )

            ON CONFLICT (operation_id)
            DO UPDATE
            SET
                status = 'PROCESSING',
                result = NULL,
                updated_at = CURRENT_TIMESTAMP,
                lease_until = %s

            WHERE
                idempotency_records.status = 'PROCESSING'
                AND idempotency_records.lease_until < CURRENT_TIMESTAMP

            RETURNING operation_id
        """

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        operation_id,
                        operation_type,
                        lease_until,
                        lease_until,
                    ),
                )

                row = cur.fetchone()

        return row is not None

    def mark_success(
        self,
        operation_id: str,
        result: str,
    ) -> None:
        query = """
            UPDATE idempotency_records
            SET
                status = 'SUCCESS',
                result = %s,
                updated_at = CURRENT_TIMESTAMP,
                lease_until = NULL
            WHERE operation_id = %s
        """

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        result,
                        operation_id,
                    ),
                )

    def mark_failed(
        self,
        operation_id: str,
        result: str,
    ) -> None:
        query = """
            UPDATE idempotency_records
            SET
                status = 'FAILED',
                result = %s,
                updated_at = CURRENT_TIMESTAMP,
                lease_until = NULL
            WHERE operation_id = %s
        """

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        result,
                        operation_id,
                    ),
                )