import sys
import time

sys.path.insert(0, "src")

from banking_investigator.repositories.idempotency_repository import (
    IdempotencyRepository,
)
from banking_investigator.services.idempotency_service import (
    IdempotencyService,
)


repository = IdempotencyRepository()
service = IdempotencyService(repository)

operation_id = "FREEZE-TXN1003"
operation_type = "FREEZE_ACCOUNT"


print("===== FIRST WORKER =====")

claimed = service.claim_operation(
    operation_id=operation_id,
    operation_type=operation_type,
    lease_seconds=5,
)

print("Claimed:", claimed)

if claimed:
    print("Operation is now PROCESSING.")
    print("Simulating worker crash...")
    print("No SUCCESS update will be performed.")


print()
print("===== IMMEDIATE SECOND WORKER =====")

claimed = service.claim_operation(
    operation_id=operation_id,
    operation_type=operation_type,
    lease_seconds=5,
)

print("Claimed:", claimed)

if not claimed:
    print("Second worker cannot claim operation.")
    print("Existing lease is still active.")


print()
print("===== WAITING FOR LEASE TO EXPIRE =====")

time.sleep(6)

print("Lease should now be expired.")


print()
print("===== RECOVERY WORKER =====")

claimed = service.claim_operation(
    operation_id=operation_id,
    operation_type=operation_type,
    lease_seconds=5,
)

print("Claimed:", claimed)

if claimed:
    print("Recovery worker successfully reclaimed operation.")
    print("Operation is PROCESSING again.")

    result = "Account ACC1003 frozen"

    service.record_success(
        operation_id=operation_id,
        result=result,
    )

    print("Operation completed.")
    print("Status: SUCCESS")
    print("Result:", result)