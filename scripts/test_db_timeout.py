import sys
import time

sys.path.insert(0, "src")

import psycopg

from banking_investigator.config.settings import settings


database_url = settings.database_url.replace("+psycopg", "")

start = time.perf_counter()

try:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SET statement_timeout = {settings.db_statement_timeout_ms}"
            )

            print(
                f"Running pg_sleep(5) with timeout "
                f"{settings.db_statement_timeout_ms} ms..."
            )

            cur.execute("SELECT pg_sleep(5)")

except Exception as exc:
    elapsed = time.perf_counter() - start

    print("\nEXCEPTION:")
    print(type(exc).__name__)
    print(exc)

    print(f"\nElapsed time: {elapsed:.2f} seconds")