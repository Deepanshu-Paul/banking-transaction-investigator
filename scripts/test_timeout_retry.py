import sys

sys.path.insert(0, "src")

from banking_investigator.agents.tool_executor import execute_tool
from banking_investigator.services.errors import RetryableError
from banking_investigator.agents import tool_executor


def slow_tool(deadline=None) -> dict:
    print("Executing slow tool...")
    raise RetryableError("Simulated database timeout")


tool_executor.TOOL_REGISTRY["slow_test"] = slow_tool

result = execute_tool("slow_test", {})

print("\nFINAL RESULT:")
print(result)