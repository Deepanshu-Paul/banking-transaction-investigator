from datetime import datetime, timezone
from pathlib import Path

import psycopg

from banking_investigator.config.settings import settings
from banking_investigator.rag.ingestion import ingest_pdf


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = (
    ROOT
    / "data"
    / "knowledge_base"
    / "disputed_transactions_sample_policy.pdf"
)

DOCUMENT_ID = "DISPUTED-TRANSACTIONS-SAMPLE"


def main() -> None:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(PDF_PATH)

    # Remove the previous copy of this sample, if present,
    # so the script can safely be rerun.
    with psycopg.connect(
        settings.postgres_conn_string
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM rag_documents
                WHERE document_id = %s
                """,
                (DOCUMENT_ID,),
            )

    result = ingest_pdf(
        pdf_path=PDF_PATH,
        document_id=DOCUMENT_ID,
        version=1,
        title="Disputed Transactions Sample Policy",
        source="data/knowledge_base/disputed_transactions_sample_policy.pdf",
        document_type="policy",
        status="active",
        effective_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        effective_to=None,
        metadata={
            "region": "UK",
            "jurisdiction": "UK",
        },
    )

    print("Ingestion successful:", result)


if __name__ == "__main__":
    main()