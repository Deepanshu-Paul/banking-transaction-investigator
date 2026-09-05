import sys

sys.path.insert(0, "src")

from banking_investigator.repositories.database import Database
from banking_investigator.services.errors import RetryableError


database = Database()

print("Running slow query through shared Database...")

try:
    database.fetch_one(
        "SELECT pg_sleep(5)"
    )
except RetryableError as exc:
    print("EXPECTED RetryableError:")
    print(exc)
else:
    print("ERROR: Query did not time out")