import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def serialize_for_llm(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )