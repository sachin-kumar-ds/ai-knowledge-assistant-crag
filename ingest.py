from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_processor import process_pdf


# Configuration

DOCUMENTS_DIR = Path(r"F:\ML Project CRAG\documents")
FAISS_DIR = "faiss_index"


# Find PDF files

pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in: {DOCUMENTS_DIR}"
    )

print("=" * 60)
print("PDF DOCUMENT INGESTION")
print("=" * 60)

print(f"Documents directory: {DOCUMENTS_DIR}")
print(f"PDF files found: {len(pdf_files)}")


# Process PDFs

docs = []

for pdf_path in pdf_files:

    print("\n" + "-" * 60)
    print(f"Processing: {pdf_path.name}")
    print("-" * 60)

    processed_docs = process_pdf(pdf_path)

    docs.extend(processed_docs)


print("\n" + "=" * 60)
print("DOCUMENT PROCESSING COMPLETE")
print("=" * 60)

print(f"PDF files processed : {len(pdf_files)}")
print(f"Pages extracted     : {len(docs)}")


# Split documents into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(docs)


print("\n" + "=" * 60)
print("CHUNKING COMPLETE")
print("=" * 60)

print(f"Total chunks: {len(chunks)}")


# Clean text

for document in chunks:

    document.page_content = (
        document.page_content
        .encode("utf-8", "ignore")
        .decode("utf-8", "ignore")
    )


# Create embeddings

print("\n" + "=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create FAISS vector store

print("\n" + "=" * 60)
print("CREATING FAISS INDEX")
print("=" * 60)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# Save FAISS index

vector_store.save_local(FAISS_DIR)


print("\n" + "=" * 60)
print("INGESTION COMPLETE")
print("=" * 60)

print(f"Created {len(chunks)} chunks.")
print(f"FAISS index saved to: {FAISS_DIR}/")