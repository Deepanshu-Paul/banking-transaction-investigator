import sys

sys.path.insert(0, "src")

from banking_investigator.agents.tool_executor import execute_tool


attempts = 0


def permanent_failure_tool() -> dict:
    global attempts
    attempts += 1

    print(f"Executing permanent failure tool... attempt {attempts}")

    raise ValueError("Invalid transaction ID")


from banking_investigator.agents import tool_executor

tool_executor.TOOL_REGISTRY["permanent_test"] = permanent_failure_tool

result = execute_tool("permanent_test", {})

print("\nFINAL RESULT:")
print(result)

print("\nTOTAL ATTEMPTS:")
print(attempts)