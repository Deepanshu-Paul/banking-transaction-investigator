from banking_investigator.agents.tool_registry import TOOL_REGISTRY
from banking_investigator.models.tool_result import ToolResult


def execute_tool(tool_name: str, arguments: dict) -> ToolResult:
    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return ToolResult(
            success=False,
            error=f"Unknown tool: {tool_name}",
        )

    try:
        result = tool(**arguments)

        return ToolResult(
            success=True,
            data=result,
        )

    except Exception as exc:
        return ToolResult(
            success=False,
            error=str(exc),
        )