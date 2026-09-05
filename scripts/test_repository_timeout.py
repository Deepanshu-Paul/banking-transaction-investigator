import sys

sys.path.insert(0, "src")

from banking_investigator.repositories.transaction_repository import (
    TransactionRepository,
)
from banking_investigator.services.errors import RetryableError


repository = TransactionRepository()

print("Running slow query through TransactionRepository...")

try:
    repository._fetch_one("SELECT pg_sleep(5)")
except RetryableError as exc:
    print("EXPECTED RetryableError:")
    print(exc)
else:
    print("ERROR: Query did not raise RetryableError")