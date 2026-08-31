from typing import Literal

from pydantic import BaseModel, ConfigDict


class RouteDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    route: Literal[
        "transaction",
        "account",
        "customer",
    ]


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    next_agent: Literal[
        "transaction",
        "account",
        "finish",
    ]