from pathlib import Path

from banking_investigator.rag.pdf_loader import load_pdf_text


def test_load_sample_banking_policy_pdf() -> None:
    pdf_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "knowledge_base"
        / "disputed_transactions_sample_policy.pdf"
    )

    text = load_pdf_text(pdf_path)

    assert len(text) > 500
    assert "dispute" in text.lower()