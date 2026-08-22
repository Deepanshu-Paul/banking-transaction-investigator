from src.banking_investigator.tools.banking_tools import (
    get_account,
    get_transaction,
)


TOOL_REGISTRY = {
    "get_transaction": get_transaction,
    "get_account": get_account,
}