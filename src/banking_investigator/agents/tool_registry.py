from banking_investigator.tools.banking_tools import (
    get_transaction,
    get_account,
)


TOOL_REGISTRY = {
    "get_transaction": get_transaction,
    "get_account": get_account,
}