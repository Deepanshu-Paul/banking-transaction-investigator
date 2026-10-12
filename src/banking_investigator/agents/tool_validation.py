from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class GetTransactionArguments(ToolArguments):
    transaction_id: str = Field(
        strict=True,
        min_length=1,
    )


class GetAccountArguments(ToolArguments):
    account_id: str = Field(
        strict=True,
        min_length=1,
    )


TOOL_ARGUMENT_SCHEMAS: dict[str, type[ToolArguments]] = {
    "get_transaction": GetTransactionArguments,
    "get_account": GetAccountArguments,
}


def validate_tool_arguments(
    tool_name: str,
    arguments: Any,
) -> dict[str, Any]:
    schema = TOOL_ARGUMENT_SCHEMAS.get(tool_name)

    if schema is None:
        raise ValueError(
            f"No argument schema configured for tool '{tool_name}'."
        )

    validated = schema.model_validate(arguments)

    return validated.model_dump()