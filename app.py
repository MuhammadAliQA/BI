import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(page_title="Smart FAQ Chatbot", layout="wide")

DATA_DIR = Path("data")
VECTOR_DB_PATH = DATA_DIR / "vector_db.json"
CHAT_STATE_PATH = DATA_DIR / "chat_state.json"
RELEVANCE_THRESHOLD = 0.2
MAX_API_RETRIES = 3
REQUESTS_PER_MINUTE = 12
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


@st.cache_resource
def get_client():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None

    if not key:
        key = os.getenv("OPENAI_API_KEY")

    if not key:
        return None

    return OpenAI(api_key=key)


def call_with_retry(func, *args, **kwargs):
    last_err = None
    for attempt in range(MAX_API_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < MAX_API_RETRIES - 1:
                time.sleep(2**attempt)
    raise last_err


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_vector_db():
    ensure_data_dir()
    with VECTOR_DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(st.session_state.vector_db, f)


def load_vector_db():
    if VECTOR_DB_PATH.exists():
        with VECTOR_DB_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_chat_state():
    ensure_data_dir()
    payload = {
        "history": st.session_state.history,
        "summary_memory": st.session_state.summary_memory,
        "uploaded_files": list(st.session_state.uploaded_files),
        "feedback": st.session_state.feedback,
    }
    with CHAT_STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_chat_state():
    if CHAT_STATE_PATH.exists():
        with CHAT_STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_vector_db()
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "summary_memory" not in st.session_state:
    st.session_state.summary_memory = ""
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()
if "hydrated" not in st.session_state:
    st.session_state.hydrated = False
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []

if not st.session_state.hydrated:
    saved = load_chat_state()
    st.session_state.history = saved.get("history", st.session_state.history)
    st.session_state.summary_memory = saved.get("summary_memory", st.session_state.summary_memory)
    st.session_state.feedback = saved.get("feedback", st.session_state.feedback)
    st.session_state.uploaded_files = set(saved.get("uploaded_files", list(st.session_state.uploaded_files)))
    st.session_state.hydrated = True


def check_access():
    try:
        app_password = st.secrets.get("APP_PASSWORD")
    except Exception:
        app_password = None

    try:
        admin_password = st.secrets.get("ADMIN_PASSWORD")
    except Exception:
        admin_password = None

    if not app_password:
        st.session_state.auth_ok = True
        st.session_state.is_admin = False
        return True

    if st.session_state.auth_ok:
        return True

    st.title("Access Required")
    user_pass = st.text_input("Enter app password", type="password")
    if st.button("Login"):
        if user_pass == app_password:
            st.session_state.auth_ok = True
            st.session_state.is_admin = user_pass == admin_password if admin_password else False
            st.rerun()
        else:
            st.error("Invalid password.")
    st.stop()


check_access()


def logout():
    try:
        app_password = st.secrets.get("APP_PASSWORD")
    except Exception:
        app_password = None

    if not app_password:
        return

    st.session_state.auth_ok = False
    st.session_state.is_admin = False
    st.session_state.request_timestamps = []
    st.rerun()


def prune_old_requests():
    now = time.time()
    st.session_state.request_timestamps = [t for t in st.session_state.request_timestamps if now - t < 60]


def is_rate_limited():
    prune_old_requests()
    return len(st.session_state.request_timestamps) >= REQUESTS_PER_MINUTE


def register_request():
    prune_old_requests()
    st.session_state.request_timestamps.append(time.time())


def embed(text: str):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not set.")
    res = call_with_retry(client.embeddings.create, model="text-embedding-3-small", input=text)
    return res.data[0].embedding


def chunk(text: str, size: int = 240, overlap: int = 50):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    merged = " ".join(paragraphs)
    words = merged.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)]


def sim(a, b):
    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    if a_n == 0 or b_n == 0:
        return 0.0
    return float(np.dot(a, b) / (a_n * b_n))


def file_hash(file_bytes: bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def validate_upload(file_name: str, file_bytes: bytes):
    ext = Path(file_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {ext or 'no extension'}"

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        return False, f"File too large: {size_mb:.1f} MB (max {MAX_UPLOAD_MB} MB)"

    return True, ""


def parse_text(file_name: str, file_bytes: bytes):
    lower = file_name.lower()
    if lower.endswith(".pdf"):
        from io import BytesIO

        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    raw_text = file_bytes.decode("utf-8", errors="ignore")
    if lower.endswith(".json"):
        try:
            payload = json.loads(raw_text)
            if isinstance(payload, list):
                qa_rows = []
                for item in payload:
                    if isinstance(item, dict):
                        q = item.get("question", "").strip()
                        a = item.get("answer", "").strip()
                        if q or a:
                            qa_rows.append(f"Q: {q}\nA: {a}")
                if qa_rows:
                    return "\n\n".join(qa_rows)
        except json.JSONDecodeError:
            pass

    return raw_text


def add(text: str, source_id: str, source_name: str):
    chunks = chunk(text)
    for idx, c in enumerate(chunks):
        st.session_state.vector_db.append(
            {"text": c, "emb": embed(c), "src": source_name, "source_id": source_id, "chunk_idx": idx}
        )
    save_vector_db()


def list_sources():
    unique = {}
    for d in st.session_state.vector_db:
        sid = d.get("source_id")
        if sid and sid not in unique:
            unique[sid] = d.get("src", sid)
    return unique


def delete_sources(source_ids):
    source_ids = set(source_ids)
    st.session_state.vector_db = [d for d in st.session_state.vector_db if d.get("source_id") not in source_ids]
    st.session_state.uploaded_files = {sid for sid in st.session_state.uploaded_files if sid not in source_ids}
    save_vector_db()
    save_chat_state()


def retrieve(q: str, k: int):
    db = st.session_state.vector_db
    if not db:
        return "No documents uploaded yet.", []

    q_emb = embed(q)
    scored = [(sim(q_emb, d["emb"]), d) for d in db]
    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [(s, d) for s, d in scored if s >= RELEVANCE_THRESHOLD][:k]

    if not filtered:
        return "No relevant context found in uploaded documents.", []

    parts, srcs = [], []
    for score, d in filtered:
        parts.append(d["text"])
        src_label = f'{d["src"]}#chunk{d.get("chunk_idx", 0)} ({score:.2f})'
        srcs.append(src_label)

    return "\n\n".join(parts), srcs


def summarize_memory():
    if client is None or len(st.session_state.messages) < 6:
        return

    text = "\n".join([m["content"] for m in st.session_state.messages])
    res = call_with_retry(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize this conversation briefly:\n{text}"}],
    )
    st.session_state.summary_memory = res.choices[0].message.content
    st.session_state.messages = st.session_state.messages[-4:]
    save_chat_state()


def prompt(ctx: str, lang: str):
    return f"""
You are a smart FAQ assistant.
Use only the provided context to answer.
If answer is not in context, say clearly that information is not available in uploaded documents.
Do not invent policies, prices, contacts, or dates.
Keep answers concise and practical.

Conversation summary:
{st.session_state.summary_memory}

Context:
{ctx}

Answer in {lang}.
"""


def generate(q: str, ctx: str, lang: str):
    if client is None:
        yield "OPENAI_API_KEY topilmadi. Chat ishlashi uchun API key qo'shing."
        return

    msgs = [{"role": "system", "content": prompt(ctx, lang)}]
    msgs.extend(st.session_state.messages[-6:])
    msgs.append({"role": "user", "content": q})

    stream = call_with_retry(client.chat.completions.create, model="gpt-4o-mini", messages=msgs, stream=True)
    full = ""
    for ch in stream:
        delta = ch.choices[0].delta.content
        if delta:
            full += delta
            yield full


with st.sidebar:
    st.title("Settings")
    c1, c2 = st.columns(2)
    c1.caption("Logged in")
    try:
        has_app_password = bool(st.secrets.get("APP_PASSWORD"))
    except Exception:
        has_app_password = False

    if c2.button("Logout", disabled=not has_app_password, help="APP_PASSWORD yoqilgan bo'lsa ishlaydi"):
        logout()

    if st.session_state.is_admin:
        st.caption("Role: Admin")
    else:
        st.caption("Role: User")

    lang = st.selectbox("Language", ["Uzbek", "Russian", "English"])
    k = st.slider("Top-K", 1, 10, 4)
    try:
        has_admin_password = bool(st.secrets.get("ADMIN_PASSWORD"))
    except Exception:
        has_admin_password = False

    can_upload = (not has_admin_password) or st.session_state.is_admin

    if can_upload:
        files = st.file_uploader("Upload FAQ", accept_multiple_files=True)
        if files:
            new_files = []
            invalid_files = []
            for f in files:
                bytes_data = f.getvalue()
                valid, err = validate_upload(f.name, bytes_data)
                if not valid:
                    invalid_files.append(f"{f.name}: {err}")
                    continue
                digest = file_hash(bytes_data)
                source_id = f"{f.name}:{digest}"
                if source_id not in st.session_state.uploaded_files:
                    new_files.append((f.name, source_id, bytes_data))

            for msg in invalid_files:
                st.warning(msg)

            if new_files:
                progress = st.progress(0)
                for idx, (name, source_id, bytes_data) in enumerate(new_files, start=1):
                    text = parse_text(name, bytes_data)
                    add(text, source_id, name)
                    st.session_state.uploaded_files.add(source_id)
                    progress.progress(int((idx / len(new_files)) * 100))
                save_chat_state()
                st.success(f"Uploaded {len(new_files)} file(s).")
            else:
                st.info("These files are already uploaded.")

        st.markdown("---")
        st.subheader("Source Manager")
        sources = list_sources()
        if sources:
            selected = st.multiselect(
                "Delete uploaded sources",
                options=list(sources.keys()),
                format_func=lambda sid: sources[sid],
                key="source_delete_select",
            )
            if st.button("Delete Selected Sources"):
                if selected:
                    delete_sources(selected)
                    st.success(f"Deleted {len(selected)} source(s).")
                    st.rerun()
                else:
                    st.info("No sources selected.")
        else:
            st.caption("No sources uploaded yet.")
    else:
        st.info("Upload faqat admin uchun ochiq.")

    if st.button("Clear DB + Chats"):
        st.session_state.vector_db = []
        st.session_state.uploaded_files = set()
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.summary_memory = ""
        save_vector_db()
        save_chat_state()
        st.success("Vector DB and chats cleared.")

    st.markdown("---")
    st.subheader("History")

    if st.button("Save Chat"):
        st.session_state.history.append(st.session_state.messages.copy())
        save_chat_state()

    for i, h in enumerate(st.session_state.history):
        if st.button(f"Chat {i + 1}", key=f"h{i}"):
            st.session_state.messages = h
            st.rerun()


st.title("Smart FAQ Chatbot (RAG)")

if not client:
    st.error("OPENAI_API_KEY topilmadi.")

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            c1, c2 = st.columns(2)
            if c1.button("Like", key=f"up{i}"):
                st.session_state.feedback[i] = "good"
                save_chat_state()
            if c2.button("Dislike", key=f"down{i}"):
                st.session_state.feedback[i] = "bad"
                save_chat_state()

q = st.chat_input("Savol yozing...")
if q:
    if is_rate_limited():
        st.warning(f"Rate limit: max {REQUESTS_PER_MINUTE} requests/minute. Please wait a bit.")
        st.stop()

    register_request()
    st.session_state.messages.append({"role": "user", "content": q})
    summarize_memory()

    with st.chat_message("assistant"):
        box = st.empty()
        try:
            ctx, src = retrieve(q, k)
        except Exception as e:
            ctx, src = f"Retrieval error: {e}", []

        out = ""
        for token in generate(q, ctx, lang):
            out = token
            box.write(out)

        if src:
            st.caption(f"Sources: {', '.join(src)}")

    st.session_state.messages.append({"role": "assistant", "content": out})
    save_chat_state()
