from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None