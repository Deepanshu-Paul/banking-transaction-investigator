GET_TRANSACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_transaction",
        "description": "Retrieve a banking transaction by its transaction ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "The unique transaction ID.",
                }
            },
            "required": ["transaction_id"],
        },
    },
}

GET_ACCOUNT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_account",
        "description": "Retrieve a banking account by its account ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The unique account ID.",
                }
            },
            "required": ["account_id"],
        },
    },
}