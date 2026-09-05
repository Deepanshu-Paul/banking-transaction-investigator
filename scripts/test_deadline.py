import sys
import time

sys.path.insert(0, "src")

from banking_investigator.agents.tool_executor import execute_tool
from banking_investigator.services.errors import RetryableError
from banking_investigator.agents import tool_executor


attempts = 0


def slow_retryable_tool(deadline=None) -> dict:
    global attempts
    attempts += 1

    print(
        f"Executing slow tool... "
        f"attempt {attempts}"
    )

    time.sleep(2)

    raise RetryableError("Simulated timeout")


tool_executor.TOOL_REGISTRY["deadline_test"] = (
    slow_retryable_tool
)

start = time.monotonic()

result = execute_tool("deadline_test", {})

elapsed = time.monotonic() - start

print("\nFINAL RESULT:")
print(result)

print(f"\nTOTAL ATTEMPTS: {attempts}")
print(f"TOTAL ELAPSED TIME: {elapsed:.2f}s")