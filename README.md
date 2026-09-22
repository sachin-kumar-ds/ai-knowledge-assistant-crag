# 🧠 AI Knowledge Assistant — Corrective RAG

An AI-powered knowledge assistant built with **Corrective Retrieval-Augmented Generation (CRAG)** to answer questions from personal study materials, including **scanned and handwritten PDF notes**.

The system combines **OCR, semantic retrieval, document evaluation, web search, and LLM-based generation** to provide grounded answers with source and page references.

---

## 🚀 Overview

Traditional RAG systems retrieve documents and directly pass them to an LLM for answer generation.

However, retrieved documents may be:

- Irrelevant
- Incomplete
- Poorly matched to the question
- Insufficient to answer the query

This project implements a **Corrective RAG pipeline** that evaluates retrieved documents before generating the final response.

If the retrieved documents are considered insufficient, the system automatically:

1. Rewrites the user's query
2. Searches the web
3. Combines relevant local documents with web results
4. Generates the final answer

The system also supports **OCR-based ingestion of scanned and handwritten notes**, making it useful for real-world educational material where PDFs may not contain machine-readable text.

---

# ✨ Key Features

### 📚 Document Question Answering

Ask questions about uploaded study materials and receive answers grounded in the indexed documents.

### 🔎 Corrective RAG

Retrieved documents are evaluated before answer generation.

The system follows:

```text
User Question
      ↓
Document Retrieval
      ↓
Document Evaluation
      ↓
 ┌───────────────┐
 │               │
Relevant      Insufficient
 │               │
 ↓               ↓
Refine       Rewrite Query
 │               ↓
 │           Web Search
 │               ↓
 └───────→ Refine Context
                 ↓
             Generate
                 ↓
              Answer
