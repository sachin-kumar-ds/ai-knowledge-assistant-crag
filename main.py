from typing import List, TypedDict, Literal

from pydantic import BaseModel

import re

from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

from langchain_tavily import TavilySearch

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

UPPER_TH = 0.7
LOWER_TH = 0.3


class State(TypedDict):
    question: str
    docs: list[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str


def retrieve_node(state: State) -> State:
    q = state["question"]
    return {"docs": retriever.invoke(q)}


class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(
    DocEvalScore,
    method="json_schema"
)


def eval_each_doc_node(state: State) -> State:
    q = state["question"]
    scores: List[float] = []
    good: List[Document] = []

    for d in state["docs"]:
        out = doc_eval_chain.invoke({
            "question": q,
            "chunk": d.page_content
        })

        scores.append(out.score)

        if out.score > LOWER_TH:
            good.append(d)

    # Correct : at least one doc > UPPER_TH
    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}."
        }

    # Incorrect : all docs < Lower_TH
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}.",
        }

    # Ambigious : All the retrived are upper the Lower TH but not good to be used for answering
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
    }


# Sentence level decomposer -> Used in knowledge refinment process to strip the retrieved documents

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# The LLM Judge

class KeeporDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(
    KeeporDrop,
    method="json_schema"
)


# Knowledge Refinment

# If Correct : Only used the retrieved Documents

# Incorrect : Use the web search

# Ambigious : Use the retrieved Documents + Web Search

def refine(state: State) -> State:
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]

    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"]

    else:
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(
        d.page_content for d in docs_to_use
    ).strip()

    strips = decompose_to_sentences(context)
    kept: List[str] = []

    for s in strips:
        decision = filter_chain.invoke({
            "question": q,
            "sentence": s
        })

        if decision.keep:
            kept.append(s)

    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context
    }


# Query ReWrite for the Web Search

class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(
    WebQuery,
    method="json_schema"
)


def rewrite_query_node(state: State) -> State:
    out = rewrite_chain.invoke({
        "question": state["question"]
    })

    return {"web_query": out.query}


# Web Search Node : Using the web query

tavily = TavilySearch(max_results=5)


def web_search_node(state: State) -> State:
    q = state.get("web_query") or state["question"]

    response = tavily.invoke({
        "query": q
    })

    results = response.get("results", [])

    web_docs: List[Document] = []

    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")

        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

        web_docs.append(
            Document(
                page_content=text,
                metadata={
                    "url": url,
                    "title": title
                }
            )
        )

    return {"web_docs": web_docs}


# The Answer Generator Prompt

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def generate(state: State) -> State:
    out = (
        answer_prompt | llm
    ).invoke({
        "question": state["question"],
        "context": state["refined_context"]
    })

    return {"answer": out.content}


# Routing Functionn

# Correct -> Knowledge Refinment

# Incorrect / Ambigious => rewrite(query) -> web_search -> refine -> generate

def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"

    else:
        return "rewrite_query"


# Building the Graph

g = StateGraph(State)

# Adding the Nodes

g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_node", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)

# Connecting the Edges

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_node")

g.add_conditional_edges(
    "eval_each_node",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    },
)

# Not Correct Path

g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")

# Correct Path goes to refine

g.add_edge("refine", "generate")
g.add_edge("generate", END)

app = g.compile()

res = app.invoke(
    {
        "question": "Batch normalization vs layer normalization",
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }
)

print("VERDICT:", res["verdict"])
print("REASON:", res["reason"])
print("WEB_QUERY:", res["web_query"])
print("\nOUTPUT:\n", res["answer"])