from datetime import datetime, timedelta
from decimal import Decimal
import sys

import psycopg

sys.path.insert(0, "src")

from banking_investigator.config.settings import settings


def seed_data():
    database_url = settings.database_url.replace("+psycopg", "")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:

            # Customers
            cur.executemany(
                """
                INSERT INTO customers
                    (customer_id, name, segment, created_at)
                VALUES
                    (%s, %s, %s, %s)
                ON CONFLICT (customer_id) DO NOTHING
                """,
                [
                    ("CUST1001", "Amit Sharma", "RETAIL", datetime.now()),
                    ("CUST1002", "Priya Mehta", "RETAIL", datetime.now()),
                    ("CUST1003", "Rahul Verma", "PREMIUM", datetime.now()),
                ],
            )

            # Accounts
            cur.executemany(
                """
                INSERT INTO accounts
                    (account_id, customer_id, account_type, status, created_at)
                VALUES
                    (%s, %s, %s, %s, %s)
                ON CONFLICT (account_id) DO NOTHING
                """,
                [
                    ("ACC1001", "CUST1001", "SAVINGS", "ACTIVE", datetime.now()),
                    ("ACC1002", "CUST1002", "SAVINGS", "ACTIVE", datetime.now()),
                    ("ACC1003", "CUST1003", "CURRENT", "ACTIVE", datetime.now()),
                ],
            )

            now = datetime.now()

            # Transactions
            transactions = [
                (
                    "TXN1001",
                    "ACC1001",
                    Decimal("2500.00"),
                    "INR",
                    "Amazon",
                    "CARD_PAYMENT",
                    "APPROVED",
                    now - timedelta(hours=5),
                ),
                (
                    "TXN1002",
                    "ACC1001",
                    Decimal("85000.00"),
                    "INR",
                    "ABC Electronics",
                    "CARD_PAYMENT",
                    "DECLINED",
                    now - timedelta(hours=4),
                ),
                (
                    "TXN1003",
                    "ACC1002",
                    Decimal("4500.00"),
                    "INR",
                    "Merchant A",
                    "CARD_PAYMENT",
                    "APPROVED",
                    now - timedelta(minutes=20),
                ),
                (
                    "TXN1004",
                    "ACC1002",
                    Decimal("4500.00"),
                    "INR",
                    "Merchant A",
                    "CARD_PAYMENT",
                    "APPROVED",
                    now - timedelta(minutes=15),
                ),
                (
                    "TXN1005",
                    "ACC1002",
                    Decimal("4500.00"),
                    "INR",
                    "Merchant A",
                    "CARD_PAYMENT",
                    "APPROVED",
                    now - timedelta(minutes=10),
                ),
                (
                    "TXN1006",
                    "ACC1003",
                    Decimal("1200.00"),
                    "INR",
                    "Swiggy",
                    "CARD_PAYMENT",
                    "APPROVED",
                    now - timedelta(hours=2),
                ),
            ]

            cur.executemany(
                """
                INSERT INTO transactions
                    (
                        transaction_id,
                        account_id,
                        amount,
                        currency,
                        merchant,
                        transaction_type,
                        status,
                        timestamp
                    )
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                transactions,
            )

        conn.commit()

    print("Synthetic banking data seeded successfully.")


if __name__ == "__main__":
    seed_data()