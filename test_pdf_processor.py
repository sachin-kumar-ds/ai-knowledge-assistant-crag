from ingestion.pdf_processor import process_pdf


pdf_path = r"C:\Users\LENOVO\Downloads\MarksheetSachin.pdf"

documents = process_pdf(pdf_path)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"Pages: {len(documents)}")

for doc in documents[:3]:

    print("\n--- PAGE ---")

    print("Source:", doc.metadata["source"])
    print("Page:", doc.metadata["page"])
    print("OCR:", doc.metadata["ocr"])

    print("\nText:")
    print(doc.page_content[:500])