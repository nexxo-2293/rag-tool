import os
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

import streamlit as st
import chromadb
from dotenv import load_dotenv
from ollama import Client  # Ollama cloud Python client

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter


# ================== ENV & CONSTANTS ==================

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
if not OLLAMA_API_KEY:
    raise ValueError("❌ Missing OLLAMA_API_KEY in .env")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud")

# Ollama Cloud client (no local server needed)
ollama_client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
)

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_PATH = BASE_DIR / "chroma_db"
DB_PATH = BASE_DIR / "rag_app.db"

DOCS_DIR.mkdir(exist_ok=True)
CHROMA_PATH.mkdir(exist_ok=True)


# ================== SQLITE SETUP ==================

def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT,
        pdf_path TEXT,
        pdf_hash TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions (id)
    )
    """)
    conn.commit()
    conn.close()


init_db()


# ================== LLAMAINDEX CONFIG (RETRIEVAL ONLY) ==================

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)


# ================== INDEX CACHE ==================

# in-memory cache: {pdf_hash: VectorStoreIndex}
if "indices" not in st.session_state:
    st.session_state["indices"] = {}


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_or_build_index(pdf_path: str, pdf_hash: str) -> VectorStoreIndex:
    """
    Use cached index if available; else build one for this PDF.
    Each PDF gets its own Chroma collection named by its hash.
    """
    if pdf_hash in st.session_state["indices"]:
        return st.session_state["indices"][pdf_hash]

    pdf_reader = PyMuPDFReader()
    documents = pdf_reader.load_data(file_path=pdf_path)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection_name = f"pdf_{pdf_hash[:16]}"

    collection = client.get_or_create_collection(collection_name)

    # If empty collection, index docs
    if collection.count() == 0:
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            show_progress=True,
        )
    else:
        # Reuse existing vectors
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store)

    st.session_state["indices"][pdf_hash] = index
    return index


# ================== OLLAMA / DEEPSEEK CALL ==================

def call_deepseek_with_context(question: str, contexts: list[str]) -> str:
    """
    Use DeepSeek Cloud via Ollama to answer based on retrieved context.
    """
    if not contexts:
        context_text = "No relevant context was retrieved."
    else:
        context_text = "\n\n---\n\n".join(contexts)

    system_prompt = (
        "You are an expert PDF question-answering assistant.\n"
        "Use ONLY the provided context snippets from the document to answer the user's question.\n"
        "If the answer is not clearly in the context, say you don't know.\n\n"
    )

    user_prompt = (
        f"{system_prompt}"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer clearly and concisely."
    )

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    resp = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=False,
    )

    # Handle both dict-style and object-style responses
    if isinstance(resp, dict):
        return resp.get("message", {}).get("content", "").strip()
    else:
        msg = getattr(resp, "message", None)
        if msg is not None and hasattr(msg, "content"):
            return msg.content.strip()
        return str(resp)


# ================== SESSION HELPERS ==================

def create_session(pdf_name: str, pdf_bytes: bytes) -> str:
    """
    Save PDF to disk, create a new chat session linked to this PDF.
    Returns session_id.
    """
    pdf_hash = compute_file_hash(pdf_bytes)
    session_id = hashlib.md5((pdf_name + pdf_hash).encode()).hexdigest()

    pdf_filename = f"{session_id}_{pdf_name}"
    pdf_path = str(DOCS_DIR / pdf_filename)

    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    now = datetime.utcnow().isoformat()

    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO sessions (id, name, pdf_path, pdf_hash, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, pdf_name, pdf_path, pdf_hash, now))
    conn.commit()
    conn.close()

    return session_id


def load_sessions():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions ORDER BY datetime(created_at) DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def load_messages(session_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content FROM messages
        WHERE session_id = ?
        ORDER BY datetime(created_at) ASC
    """, (session_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def add_message(session_id: str, role: str, content: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO messages (session_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_session(session_id: str):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    conn.close()
    return row


# ================== STREAMLIT UI ==================

st.set_page_config(page_title="PDF RAG Chat (DeepSeek Cloud)", layout="wide")
st.title("📄 PDF RAG Chat — DeepSeek Cloud + Local Retrieval")

st.caption(
    "Each chat on the left is bound to one PDF. "
    "Click a chat to switch context; upload a new PDF to start a new chat."
)

if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ----- SIDEBAR: CHAT LIST + NEW CHAT -----

with st.sidebar:
    st.header("💬 Chats (per PDF)")
    sessions = load_sessions()

    # Show existing sessions as buttons
    for s in sessions:
        label = s["name"]
        if st.button(label, key=f"session_{s['id']}"):
            st.session_state["current_session_id"] = s["id"]
            st.session_state["messages"] = load_messages(s["id"])

    st.markdown("---")
    st.subheader("➕ New Chat from PDF")

    new_pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="new_pdf")
    if st.button("Create Chat") and new_pdf:
        pdf_bytes = new_pdf.read()
        session_id = create_session(new_pdf.name, pdf_bytes)
        st.session_state["current_session_id"] = session_id
        st.session_state["messages"] = []
        st.success(f"New chat created for: {new_pdf.name}")


# ----- MAIN AREA -----

current_session_id = st.session_state["current_session_id"]
current_session = get_session(current_session_id) if current_session_id else None

if not current_session:
    st.info("Select a chat from the left sidebar, or upload a PDF to start a new chat.")
else:
    st.subheader(f"📄 Chat for: {current_session['name']}")

    # Show chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about this PDF...")

    if user_input:
        # Store and display user message
        add_message(current_session_id, "user", user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieval + DeepSeek Cloud answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and asking DeepSeek..."):
                pdf_path = current_session["pdf_path"]
                pdf_hash = current_session["pdf_hash"]

                index = get_or_build_index(pdf_path, pdf_hash)
                retriever = index.as_retriever(similarity_top_k=5)
                nodes = retriever.retrieve(user_input)

                contexts = []
                for node_ws in nodes:
                    try:
                        txt = node_ws.node.get_text()
                    except Exception:
                        txt = str(node_ws.node)
                    contexts.append(txt)

                answer = call_deepseek_with_context(user_input, contexts)
                st.markdown(answer)

                # Expandable evidence (as you chose: B)
                if contexts:
                    st.markdown("### 🔎 Retrieved Evidence")
                    for i, (node_ws, ctx) in enumerate(zip(nodes, contexts), start=1):
                        score = getattr(node_ws, "score", None)
                        label = f"Evidence #{i}"
                        if score is not None:
                            try:
                                label += f" (score: {score:.3f})"
                            except TypeError:
                                pass

                        with st.expander(label):
                            st.write(ctx[:1500])

        # Save assistant answer in DB + state
        add_message(current_session_id, "assistant", answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
