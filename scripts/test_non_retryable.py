import sys

sys.path.insert(0, "src")

from banking_investigator.agents.tool_executor import execute_tool
from banking_investigator.agents import tool_executor


attempts = 0


def permanent_failure_tool(deadline=None) -> dict:
    global attempts
    attempts += 1

    print(
        f"Executing permanent failure tool... "
        f"attempt {attempts}"
    )

    raise ValueError("Invalid transaction ID")


tool_executor.TOOL_REGISTRY["permanent_test"] = (
    permanent_failure_tool
)

result = execute_tool("permanent_test", {})

print("\nFINAL RESULT:")
print(result)

print("\nTOTAL ATTEMPTS:")
print(attempts)