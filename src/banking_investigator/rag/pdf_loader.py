from pathlib import Path

from pypdf import PdfReader


def load_pdf_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError("Expected a PDF file")

    reader = PdfReader(str(path))

    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    text = "\n\n".join(
        page.strip()
        for page in pages
        if page.strip()
    )

    if not text:
        raise ValueError(
            "No extractable text found in the PDF"
        )

    return text