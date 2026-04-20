import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests
import fitz
import base64
import io
import os
from datetime import datetime
from unstructured.partition.auto import partition
from unstructured.documents.elements import Title

# ── Config ────────────────────────────────────────────────────────────────────

TEMP_DIR   = "temp_files_multimodal"
CHROMA_DIR = "chroma_data_multimodal"
os.makedirs(TEMP_DIR, exist_ok=True)

OLLAMA_URL   = "http://localhost:11434/api/generate"
TEXT_MODEL   = "tinyllama"
VISION_MODEL = "moondream"

VISION_TIMEOUT = 600   # seconds per image — moondream on MacBook Air ~5–15s each
TEXT_TIMEOUT   = 120

MIN_IMG_WIDTH  = 100   # skip decorative/icon images
MIN_IMG_HEIGHT = 100

# ── ChromaDB ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_collection():
    client   = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="course_docs", embedding_function=embed_fn
    )

collection = get_collection()

# ── Extraction ────────────────────────────────────────────────────────────────

def extract_text_chunks(file_path: str, file_name: str) -> list[dict]:
    """Layout-aware text extraction via unstructured."""
    chunks, current, current_type = [], [], None
    try:
        elements = partition(filename=file_path)
        for el in elements:
            text = str(el).strip()
            if not text:
                continue
            el_type = type(el).__name__
            if isinstance(el, Title) and current:
                chunks.append({"text": "\n".join(t for t, _ in current),
                                "type": current_type or "NarrativeText"})
                current = []
            current.append((text, el_type))
            current_type = el_type
        if current:
            chunks.append({"text": "\n".join(t for t, _ in current),
                            "type": current_type or "NarrativeText"})
    except Exception as e:
        st.warning(f"Text parsing warning for {file_name}: {e}")
    return chunks


def caption_image(img_bytes: bytes) -> str | None:
    """Send image to vision model; return caption or None on failure."""
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload  = {
        "model":  VISION_MODEL,
        "prompt": (
            "Describe this diagram or image in detail. "
            "Focus on any data, text labels, and relationships shown. "
            "Be concise but complete."
        ),
        "images": [img_b64],
        "stream": False,
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=VISION_TIMEOUT)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        st.warning("⏱️ Vision model timed out on an image — skipping.")
    except Exception as e:
        st.warning(f"Vision error: {e}")
    return None


def extract_image_chunks(file_path: str, file_name: str,
                         progress_placeholder) -> list[dict]:
    """Extract and caption all non-trivial images from a PDF."""
    chunks = []
    try:
        doc        = fitz.open(file_path)
        all_images = [
            (page_num, img)
            for page_num in range(len(doc))
            for img in doc.load_page(page_num).get_images(full=True)
        ]
        total = len(all_images)
        for i, (page_num, img) in enumerate(all_images):
            xref       = img[0]
            base_image = doc.extract_image(xref)

            # Skip tiny images (logos, icons, decorations)
            if (base_image.get("width",  0) < MIN_IMG_WIDTH or
                    base_image.get("height", 0) < MIN_IMG_HEIGHT):
                continue

            progress_placeholder.text(
                f"  🖼️  Captioning image {i + 1}/{total} on page {page_num + 1}…"
            )
            caption = caption_image(base_image["image"])
            if caption:
                chunks.append({
                    "text": f"[IMAGE DESCRIPTION — Page {page_num + 1}]: {caption}",
                    "type": "ImageCaption",
                })
    except Exception as e:
        st.warning(f"Image extraction failed for {file_name}: {e}")
    return chunks


def extract_all_chunks(file_path: str, file_name: str,
                       progress_placeholder) -> list[dict]:
    chunks = extract_text_chunks(file_path, file_name)
    if file_name.lower().endswith(".pdf"):
        chunks += extract_image_chunks(file_path, file_name, progress_placeholder)
    return chunks or [{"text": "No content extracted.", "type": "Unknown"}]

# ── RAG helpers ───────────────────────────────────────────────────────────────

SEARCH_TYPES = ["NarrativeText", "Title", "ListItem", "Table", "ImageCaption"]

def retrieve_chunks(question: str, n: int = 5) -> tuple[list, list]:
    """Retrieve top-n chunks; fall back to unfiltered if no typed results."""
    try:
        results = collection.query(
            query_texts=[question],
            n_results=n,
            where={"type": {"$in": SEARCH_TYPES}},
        )
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if docs:
            return docs, metas
    except Exception:
        pass
    results = collection.query(query_texts=[question], n_results=n)
    return results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]


def ask_llm(context: str, question: str) -> str:
    prompt = (
        "You are a helpful academic tutor for WU Vienna students. "
        "Use ONLY the course material excerpts below to answer. "
        "If the answer is not in the excerpts, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    payload = {"model": TEXT_MODEL, "prompt": prompt, "stream": False}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=TEXT_TIMEOUT)
        return (res.json().get("response", "").strip()
                if res.status_code == 200 else f"LLM error {res.status_code}")
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WU Vienna — Multimodal RAG",
    page_icon="🎓",
    layout="wide",
)
st.title("🎓 WU Vienna Course Tutor — Multimodal RAG")
st.caption(
    f"Text model: **{TEXT_MODEL}** · Vision model: **{VISION_MODEL}** · "
    f"DB: `{CHROMA_DIR}`"
)

# ── Sidebar: ingest ───────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Knowledge Base")

    # Show what's already indexed
    try:
        count = collection.count()
        st.metric("Chunks indexed", count)
    except Exception:
        st.metric("Chunks indexed", "—")

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "txt", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if st.button("⚡ Ingest", disabled=not uploaded_files, use_container_width=True):
        for file in uploaded_files:
            file_path = os.path.join(TEMP_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            status_text = st.empty()
            status_text.text(f"Processing {file.name}…")

            chunks = extract_all_chunks(file_path, file.name, status_text)

            # Upsert so re-ingesting the same file doesn't duplicate
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{file.name}_chunk{idx + 1}"
                collection.upsert(
                    documents=[chunk["text"]],
                    metadatas=[{"source": file.name,
                                "chunk": idx + 1,
                                "type":  chunk["type"]}],
                    ids=[chunk_id],
                )

            status_text.empty()
            st.success(f"✅ {file.name} — {len(chunks)} chunks")

    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        collection.delete(where={"type": {"$in": SEARCH_TYPES + ["Unknown"]}})
        st.rerun()

# ── Main: chat ────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role": "user"|"assistant", "content": str}

# Render conversation history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something about your course materials…")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            chunks, metas = retrieve_chunks(query)

        if not chunks:
            answer = "I couldn't find any relevant content in the knowledge base."
        else:
            context = "\n---\n".join(chunks)
            with st.spinner("Generating answer…"):
                answer = ask_llm(context, query)

        st.write(answer)

        # Show sources in an expander
        if chunks:
            with st.expander("📎 Retrieved context", expanded=False):
                for i, (chunk, meta) in enumerate(zip(chunks, metas), 1):
                    source   = meta.get("source", "unknown")
                    ctype    = meta.get("type",   "unknown")
                    is_image = ctype == "ImageCaption"
                    icon     = "🖼️" if is_image else "📄"
                    st.markdown(
                        f"**{icon} Chunk {i}** · `{source}` · *{ctype}*"
                    )
                    st.caption(chunk[:800] + ("…" if len(chunk) > 800 else ""))
                    st.divider()

    st.session_state.history.append({"role": "assistant", "content": answer})