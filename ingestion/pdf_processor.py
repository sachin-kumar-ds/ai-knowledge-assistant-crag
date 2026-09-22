from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
from paddleocr import PaddleOCR
from langchain_core.documents import Document


MIN_TEXT_PER_PAGE = 50

_ocr = None


def get_ocr():
    global _ocr

    if _ocr is None:
        print("Loading PaddleOCR...")
        _ocr = PaddleOCR(lang="en")

    return _ocr


def extract_text_from_pdf(pdf_path: str | Path) -> list[Document]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf):
            text = page.get_text("text").strip()

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number + 1,
                        "ocr": False,
                    },
                )
            )

    return documents


def needs_ocr(documents: list[Document]) -> bool:
    if not documents:
        return True

    total_chars = sum(
        len(doc.page_content.strip())
        for doc in documents
    )

    average_chars_per_page = total_chars / len(documents)

    print(
        f"Average extracted characters per page: "
        f"{average_chars_per_page:.2f}"
    )

    return average_chars_per_page < MIN_TEXT_PER_PAGE


def run_ocr(pdf_path: str | Path) -> list[Document]:
    pdf_path = Path(pdf_path)

    ocr = get_ocr()
    documents = []

    print("Running PaddleOCR...")

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        with fitz.open(pdf_path) as pdf:
            total_pages = len(pdf)

            for page_number, page in enumerate(pdf):
                print(
                    f"→ OCR page {page_number + 1}/{total_pages}"
                )

                matrix = fitz.Matrix(2, 2)
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                image_path = (
                    temp_dir / f"page_{page_number + 1}.png"
                )

                pixmap.save(str(image_path))

                results = ocr.predict(str(image_path))

                page_text = []

                for result in results:
                    result_data = result.json

                    if callable(result_data):
                        result_data = result_data()

                    if not isinstance(result_data, dict):
                        continue

                    result_data = result_data.get(
                        "res",
                        result_data,
                    )

                    rec_texts = result_data.get(
                        "rec_texts",
                        [],
                    )

                    for text in rec_texts:
                        text = str(text).strip()

                        if text:
                            page_text.append(text)

                extracted_text = "\n".join(page_text).strip()

                documents.append(
                    Document(
                        page_content=extracted_text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number + 1,
                            "ocr": True,
                        },
                    )
                )

    print("PaddleOCR completed.")

    return documents


def process_pdf(pdf_path: str | Path) -> list[Document]:
    pdf_path = Path(pdf_path)

    print("\n" + "=" * 60)
    print(f"Processing: {pdf_path.name}")
    print("=" * 60)

    documents = extract_text_from_pdf(pdf_path)

    if not needs_ocr(documents):
        print("✓ Text detected.")
        print("✓ OCR not required.")
        return documents

    print("⚠ Insufficient extractable text.")
    print("→ PDF appears to be scanned.")
    print("→ Starting PaddleOCR...")

    documents = run_ocr(pdf_path)

    print(
        f"✓ Extracted text from {len(documents)} pages "
        f"using PaddleOCR."
    )

    return documents
