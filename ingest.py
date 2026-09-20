from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

docs = (
    PyPDFLoader(r"F:\ML Project CRAG\documents\AI Engineering.pdf").load()
    + PyPDFLoader(r"F:\ML Project CRAG\documents\book1.pdf").load()
    + PyPDFLoader(r"F:\ML Project CRAG\documents\Hands-On-Large-Language-Models.pdf").load()
    + PyPDFLoader(r"F:\ML Project CRAG\documents\Hands-on-Machine-Learning.pdf").load()
)

chunks = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
).split_documents(docs)

for d in chunks:
    d.page_content = d.page_content.encode(
        "utf-8", "ignore"
    ).decode("utf-8", "ignore")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

vector_store.save_local("faiss_index")

print(f"Created {len(chunks)} chunks.")
print("FAISS index saved to faiss_index/")