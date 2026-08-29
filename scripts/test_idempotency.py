import sys

sys.path.insert(0, "src")

from banking_investigator.repositories.idempotency_repository import (
    IdempotencyRepository,
)
from banking_investigator.services.idempotency_service import (
    IdempotencyService,
)


repository = IdempotencyRepository()
service = IdempotencyService(repository)


operation_id = "FREEZE-TXN1002"
operation_type = "FREEZE_ACCOUNT"


print("===== FIRST ATTEMPT =====")

claimed = service.claim_operation(
    operation_id=operation_id,
    operation_type=operation_type,
)

if claimed:
    print("Operation claimed successfully.")
    print("Status: PROCESSING")

    # Simulated banking side effect.
    result = "Account ACC1002 frozen"

    print("Executing side effect...")

    service.record_success(
        operation_id=operation_id,
        result=result,
    )

    print("Status: SUCCESS")
    print("Result:", result)

else:
    print("Operation was already claimed.")


print()
print("===== SECOND ATTEMPT =====")

claimed = service.claim_operation(
    operation_id=operation_id,
    operation_type=operation_type,
)

if claimed:
    print("Operation claimed successfully.")
    print("Executing side effect...")
else:
    print("Operation already exists.")
    print("Skipping side effect.")

    existing = service.get_existing_operation(
        operation_id
    )

    print("Stored status:", existing["status"])
    print("Stored result:", existing["result"])