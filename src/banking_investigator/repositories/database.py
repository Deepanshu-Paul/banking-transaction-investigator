from typing import Any

import psycopg
from psycopg.errors import QueryCanceled

from banking_investigator.config.settings import settings
from banking_investigator.services.errors import RetryableError
from banking_investigator.utils.deadline import Deadline


class Database:
    def __init__(self) -> None:
        self.database_url = settings.database_url.replace(
            "+psycopg",
            "",
        )

    def fetch_one(
        self,
        query: str,
        params: tuple[Any, ...] = (),
        deadline: Deadline | None = None,
    ) -> tuple[Any, ...] | None:

        if deadline is not None:
            remaining_seconds = deadline.remaining_seconds()

            if remaining_seconds <= 0:
                raise RetryableError(
                    "Database operation deadline exceeded"
                )

            timeout_ms = max(
                1,
                int(remaining_seconds * 1000),
            )
        else:
            timeout_ms = settings.db_statement_timeout_ms

        try:
            with psycopg.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SET statement_timeout = {timeout_ms}"
                    )

                    cur.execute(query, params)

                    return cur.fetchone()

        except QueryCanceled as exc:
            raise RetryableError(
                f"Database query timed out after "
                f"{timeout_ms} ms"
            ) from exc