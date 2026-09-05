import sys

sys.path.insert(0, "src")

from banking_investigator.agents.tool_executor import execute_tool
from banking_investigator.agents.tool_registry import TOOL_REGISTRY


attempts = {"count": 0}


def flaky_tool() -> str:
    attempts["count"] += 1

    print(f"Attempt {attempts['count']}")

    if attempts["count"] < 3:
        raise RuntimeError("Temporary failure")

    return {"message": "Success on third attempt"}


TOOL_REGISTRY["flaky_test"] = flaky_tool


result = execute_tool("flaky_test", {})

print("\nRESULT:")
print(result)

print("\nTOTAL ATTEMPTS:")
print(attempts["count"])