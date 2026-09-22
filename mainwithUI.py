from typing import List, TypedDict, Literal
from pathlib import Path
import sqlite3
from pydantic import BaseModel
import uuid

import re

import streamlit as st

from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

from langchain_tavily import TavilySearch

from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Cached ML Resources
# --------------------------------------------------
# Streamlit reruns this script when the user switches chats.
# Cache the expensive resources so they are loaded only once
# instead of being recreated on every UI interaction.

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def load_vector_store(_embeddings, index_path: str):
    return FAISS.load_local(
        index_path,
        _embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource(show_spinner=False)
def load_retriever(_vector_store):
    return _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


llm = load_llm()
embeddings = load_embeddings()
FAISS_PATH = str(BASE_DIR / "faiss_index")
vector_store = load_vector_store(embeddings, FAISS_PATH)
retriever = load_retriever(vector_store)

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
    chat_history: List[dict]
    conversation_summary: str


@st.cache_resource(show_spinner=False)
def load_memory(db_path: str):
    conn = sqlite3.connect(
        db_path,
        check_same_thread=False
    )
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, saver


sqlite_conn, memory = load_memory(str(BASE_DIR / "chat_memory.db"))


# --------------------------------------------------
# Persistent Chat Thread Storage
# --------------------------------------------------

sqlite_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_threads (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
)
sqlite_conn.commit()


def create_thread() -> str:
    thread_id = str(uuid.uuid4())

    sqlite_conn.execute(
        "INSERT INTO chat_threads (thread_id, title) VALUES (?, ?)",
        (thread_id, "New Chat")
    )
    sqlite_conn.commit()

    return thread_id


def get_threads():
    return sqlite_conn.execute(
        """
        SELECT thread_id, title
        FROM chat_threads
        ORDER BY created_at DESC
        """
    ).fetchall()


def update_thread_title(thread_id: str, title: str):
    sqlite_conn.execute(
        """
        UPDATE chat_threads
        SET title = ?
        WHERE thread_id = ?
        """,
        (title, thread_id)
    )
    sqlite_conn.commit()


def delete_thread(thread_id: str):
    sqlite_conn.execute(
        "DELETE FROM chat_threads WHERE thread_id = ?",
        (thread_id,)
    )

    sqlite_conn.execute(
        "DELETE FROM checkpoints WHERE thread_id = ?",
        (thread_id,)
    )

    sqlite_conn.execute(
        "DELETE FROM writes WHERE thread_id = ?",
        (thread_id,)
    )

    sqlite_conn.commit()

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
        (
            "human",
            "Question: {question}\n\n"
            "Sentence:\n{sentence}"
        ),
    ]
)


filter_chain = filter_prompt | llm.with_structured_output(
    KeeporDrop,
    method="function_calling"
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

@st.cache_resource(show_spinner=False)
def load_tavily():
    return TavilySearch(max_results=5)


tavily = load_tavily()


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


# Conversation Summarization
#
# Keep only the most recent exchanges in chat_history.
# Older exchanges are compressed into conversation_summary so the
# LLM context does not keep growing indefinitely.

SUMMARY_TRIGGER = 6
RECENT_EXCHANGES = 3

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You maintain a concise summary of a conversation between a user "
            "and an AI tutor.\n"
            "Update the existing summary using the older conversation below.\n"
            "Preserve important facts, topics discussed, user preferences, "
            "decisions, unresolved questions, and useful context for future "
            "questions.\n"
            "Do not invent information. Keep the summary concise."
        ),
        (
            "human",
            "Existing summary:\n{summary}\n\n"
            "Older conversation:\n{older_history}\n\n"
            "Return the updated conversation summary."
        ),
    ]
)

summary_chain = summary_prompt | llm


# The Answer Generator Prompt

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'",
        ),
        (
            "human",
            "Previous conversation:\n{history}\n\n"
            "Current question:\n{question}\n\n"
            "Context:\n{context}"
        ),
    ]
)


def generate(state: State) -> State:

    history = state.get("chat_history", [])
    conversation_summary = state.get("conversation_summary", "")

    # Only recent exchanges are sent in full.
    recent_history = history[-RECENT_EXCHANGES:]

    history_text = "\n\n".join(
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in recent_history
    )

    # The summary represents older conversation context.
    if conversation_summary:
        history_text = (
            f"Conversation summary:\n{conversation_summary}\n\n"
            f"Recent conversation:\n{history_text}"
        )

    out = (
        answer_prompt | llm
    ).invoke({
        "question": state["question"],
        "context": state["refined_context"],
        "history": history_text
    })

    updated_history = history + [
        {
            "question": state["question"],
            "answer": out.content
        }
    ]

    updated_summary = conversation_summary

    # Once history becomes large, summarize the older exchanges and
    # retain only the most recent few exchanges verbatim.
    if len(updated_history) >= SUMMARY_TRIGGER:
        older_history = updated_history[:-RECENT_EXCHANGES]

        older_text = "\n\n".join(
            f"User: {item['question']}\nAssistant: {item['answer']}"
            for item in older_history
        )

        summary_out = summary_chain.invoke({
            "summary": conversation_summary or "No previous summary.",
            "older_history": older_text
        })

        updated_summary = summary_out.content
        updated_history = updated_history[-RECENT_EXCHANGES:]

    return {
        "answer": out.content,
        "chat_history": updated_history,
        "conversation_summary": updated_summary
    }


# Routing Functionn

# Correct -> Knowledge Refinment

# Incorrect / Ambigious => rewrite(query) -> web_search -> refine -> generate

def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"

    else:
        return "rewrite_query"


# Building the Graph

@st.cache_resource(show_spinner=False)
def build_app():
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

    return g.compile(checkpointer=memory)


app = build_app()


# --------------------------------------------------
# Streamlit Frontend
# --------------------------------------------------

if "current_thread_id" not in st.session_state:

    existing_threads = get_threads()

    if existing_threads:
        st.session_state.current_thread_id = existing_threads[0][0]
    else:
        st.session_state.current_thread_id = create_thread()


def new_chat():
    st.session_state.current_thread_id = create_thread()


st.title("📚 AI Knowledge Assistant")

# --------------------------------------------------
# Sidebar - Persistent Chat Threads
# --------------------------------------------------

with st.sidebar:

    st.header("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()

    thread_items = get_threads()

    for tid, title in thread_items:

        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(
                title,
                key=f"thread_{tid}",
                use_container_width=True
            ):
                st.session_state.current_thread_id = tid
                st.rerun()

        with col2:
            if st.button(
                "🗑️",
                key=f"delete_{tid}",
                help="Delete this chat"
            ):
                delete_thread(tid)

                if st.session_state.current_thread_id == tid:
                    remaining_threads = get_threads()

                    if remaining_threads:
                        st.session_state.current_thread_id = remaining_threads[0][0]
                    else:
                        st.session_state.current_thread_id = create_thread()

                st.rerun()


thread_id = st.session_state.current_thread_id
config = {
    "configurable": {
        "thread_id": thread_id
    }
}

# --------------------------------------------------
# Display Saved Conversation
# --------------------------------------------------

current_state = app.get_state(config)

if current_state and current_state.values:

    saved_history = current_state.values.get(
        "chat_history",
        []
    )

    if saved_history:

        for item in saved_history:
            with st.chat_message("user"):
                st.write(item["question"])

            with st.chat_message("assistant"):
                st.markdown(item["answer"])

# --------------------------------------------------
# Question Input
# --------------------------------------------------

question = st.chat_input(
    "Ask something about Machine Learning..."
)

if question and question.strip():

    question = question.strip()

    # Give a new thread a useful persistent title.
    current_thread = sqlite_conn.execute(
        "SELECT title FROM chat_threads WHERE thread_id = ?",
        (thread_id,)
    ).fetchone()

    if current_thread and current_thread[0] == "New Chat":

        title = question[:35]

        if len(question) > 35:
            title += "..."

        update_thread_title(thread_id, title)

    with st.chat_message("user"):
        st.write(question)

    status = st.empty()

    status.info("🔍 Searching your documents...")

    res = app.invoke(
        {
            "question": question,
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
        },
        config=config
    )

    status.success("✅ Done!")

    with st.chat_message("assistant"):
        st.markdown(res["answer"])

    # --------------------------------------------------
    # Debug / CRAG Information
    # --------------------------------------------------

    with st.expander("🔎 CRAG Details"):

        st.subheader("VERDICT")
        st.write(res["verdict"])

        st.subheader("REASON")
        st.write(res["reason"])

        if res.get("web_query"):
            st.subheader("WEB QUERY")
            st.write(res["web_query"])

    # --------------------------------------------------
    # Sources / Metadata
    # --------------------------------------------------

    if res.get("good_docs"):

        st.subheader("📚 Sources")

        seen_sources = set()

        for doc in res["good_docs"]:

            source = doc.metadata.get(
                "source",
                "Unknown source"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            source_key = (source, page)

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            st.markdown(
                f"- 📖 **{source}** — Page **{page}**"
            )
