import os
import re
import io
import json
import time
from typing import List, Dict, Any

import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Optional LLM (Groq). Falls back to extractive summaries if GROQ_API_KEY not set
try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

# ──────────────────────────────────────────────────────────────────────────────
# Utility: Simple metadata extractors for resume fields
# ──────────────────────────────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?|\d{4}[\s-]?)\d{3}[\s-]?\d{4}")
NAME_HINT_RE = re.compile(r"^(?:name\s*[:\-]\s*)?([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)*)$", re.IGNORECASE)


def extract_contact(text: str) -> Dict[str, str]:
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    # Heuristic for name: first non-empty line that looks like a name
    name = None
    for line in text.splitlines()[:20]:
        line = line.strip()
        if not line:
            continue
        m = NAME_HINT_RE.match(line)
        if m:
            name = m.group(1).strip()
            break
    return {
        "name": name or "",
        "email": emails[0] if emails else "",
        "phone": phones[0] if phones else "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build or update the vector index from uploaded PDFs
# ──────────────────────────────────────────────────────────────────────────────

def load_pdfs_to_docs(files: List[io.BytesIO]) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        # Save to a temp buffer so PyPDFLoader can read
        data = f.read()
        bio = io.BytesIO(data)
        # PyPDFLoader works with file paths; write to a temp file-like via NamedTemporaryFile isn't strictly needed.
        # Instead, use PyPDFLoader with bytes via workaround: write to /tmp.
        tmp_path = os.path.join("/tmp", f"upload_{time.time_ns()}.pdf")
        with open(tmp_path, "wb") as tmp:
            tmp.write(data)
        loader = PyPDFLoader(tmp_path)
        file_docs = loader.load()
        # Attach file name in metadata
        for d in file_docs:
            d.metadata = d.metadata or {}
            d.metadata.update({"source_file": getattr(f, 'name', 'uploaded.pdf'), "tmp_path": tmp_path})
        docs.extend(file_docs)
    return docs


def chunk_docs(docs: List[Document], chunk_size=800, chunk_overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", "!", "?", ",", " "]
    )
    return splitter.split_documents(docs)


# ──────────────────────────────────────────────────────────────────────────────
# Candidate Registry: ties chunks/files to a candidate identity for filtering
# ──────────────────────────────────────────────────────────────────────────────

def build_candidate_registry(raw_docs: List[Document]) -> Dict[str, Dict[str, Any]]:
    """Infer candidate name/email/phone per file and return a registry.
    Keyed by a candidate_key (prefer email>phone>filename).
    """
    registry: Dict[str, Dict[str, Any]] = {}
    by_file_text: Dict[str, str] = {}

    # Aggregate text per file to extract metadata once per resume
    for d in raw_docs:
        file = d.metadata.get("source_file", "unknown.pdf")
        by_file_text.setdefault(file, "")
        by_file_text[file] += "\n" + d.page_content

    for file, text in by_file_text.items():
        contact = extract_contact(text)
        key = contact["email"] or contact["phone"] or file
        registry[key] = {
            "candidate_key": key,
            "name": contact["name"] or "Unknown",
            "email": contact["email"],
            "phone": contact["phone"],
            "source_file": file,
        }
    return registry


# ──────────────────────────────────────────────────────────────────────────────
# LLM helper (optional via GROQ). Fallback to extractive answer if not available
# ──────────────────────────────────────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if HAS_GROQ and api_key:
        return ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=768,
        )
    return None


def llm_answer(llm, question: str, context: str) -> str:
    if llm is None:
        # Simple extractive fallback: return the most relevant lines
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
        top = "\n".join(lines[:50])
        return (
            "(No LLM configured; showing an extractive answer)\n\n" + top
        )
    prompt = (
        "You are a helpful recruiter assistant. Answer the question using the given resume excerpts.\n"
        "If the answer depends on a specific candidate, only use those filtered excerpts.\n"
        "If not found, say 'Not found in resumes.'\n\n"
        f"Question: {question}\n\nResume Excerpts:\n{context}\n"
    )
    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 Resume RAG Chatbot (LangChain + Streamlit)")

with st.sidebar:
    st.header("Settings")
    st.write("Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local)")
    persist = st.checkbox("Persist FAISS index to ./index", value=False)
    top_k = st.slider("Retriever top_k", 2, 10, 4)
    st.divider()
    st.caption("Optional: Set GROQ_API_KEY in your environment for LLM answers.")

# Session state setup
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "registry" not in st.session_state:
    st.session_state.registry = {}
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Upload block
st.subheader("1) Upload bulk resume PDFs")
files = st.file_uploader(
    "Upload one or more PDF resumes", type=["pdf"], accept_multiple_files=True
)

col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("Build/Update Index", type="primary", disabled=not files):
        with st.spinner("Reading PDFs, splitting, embedding, and indexing..."):
            raw_docs = load_pdfs_to_docs(files)
            st.session_state.raw_docs.extend(raw_docs)

            # Build candidate registry from all raw docs so far
            st.session_state.registry = build_candidate_registry(st.session_state.raw_docs)

            # Chunk everything (idempotent-ish: rebuild from all docs to keep it simple)
            chunks = chunk_docs(st.session_state.raw_docs)

            # Add candidate_key onto each chunk using source_file lookup
            file_to_key = {v["source_file"]: k for k, v in st.session_state.registry.items()}
            enriched: List[Document] = []
            for d in chunks:
                file = d.metadata.get("source_file", "unknown.pdf")
                candidate_key = file_to_key.get(file, file)
                d.metadata = d.metadata or {}
                d.metadata.update({"candidate_key": candidate_key})
                enriched.append(d)
            st.session_state.chunks = enriched

            # Build vector store
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            if persist:
                vs = FAISS.from_documents(enriched, embed)
                vs.save_local("index")
                st.session_state.vectorstore = vs
            else:
                st.session_state.vectorstore = FAISS.from_documents(enriched, embed)
        st.success(f"Indexed {len(st.session_state.chunks)} chunks from {len(st.session_state.registry)} candidate(s).")

with col_b:
    if st.button("Load persisted index from ./index", disabled=st.session_state.vectorstore is not None):
        try:
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.load_local("index", embed, allow_dangerous_deserialization=True)
            st.success("Index loaded from ./index")
        except Exception as e:
            st.error(f"Failed to load index: {e}")

st.divider()

# Candidate Browser
st.subheader("2) Find a particular candidate")
st.caption("Type a name, email, or phone. We'll fuzzy-match against the registry and embeddings.")
query_cand = st.text_input("Search candidate")

sel_key = None
if query_cand and st.session_state.registry:
    # First pass: direct key/name/email/phone match
    qnorm = query_cand.strip().lower()
    exact_hits = [k for k,v in st.session_state.registry.items() if qnorm in (v["name"].lower()+" "+v["email"].lower()+" "+v["phone"].lower())]

    # Second pass: vector search constrained to contact-like terms
    if not exact_hits and st.session_state.vectorstore is not None:
        retr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
        cand_docs = retr.get_relevant_documents(f"Candidate identity for: {query_cand}")
        for d in cand_docs:
            key = d.metadata.get("candidate_key")
            if key and key not in exact_hits:
                exact_hits.append(key)

    if exact_hits:
        sel_key = st.selectbox("Matching candidates", options=exact_hits, format_func=lambda k: f"{st.session_state.registry[k]['name']}  |  {st.session_state.registry[k]['email']}  |  {st.session_state.registry[k]['phone']}")
    else:
        st.info("No matching candidate detected yet. Try another query or upload more resumes.")

if sel_key:
    info = st.session_state.registry[sel_key]
    st.success(f"Selected: {info['name']} | {info['email']} | {info['phone']} | File: {info['source_file']}")

st.divider()

# Chat Area
st.subheader("3) Ask questions")
st.caption("Ask general questions or about the selected candidate. If a candidate is selected, retrieval is filtered to that resume.")
user_q = st.text_input("Your question", placeholder="e.g., Show projects of Priya, or What is John Doe's experience with React?")

if "chat" not in st.session_state:
    st.session_state.chat = []

if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})

llm = get_llm()

if st.button("Ask") and user_q:
    if st.session_state.vectorstore is None:
        st.warning("Please upload PDFs and build the index first.")
    else:
        if sel_key:
            # Filter chunks by candidate_key, then run vector search on that subset via a metadata filter workaround.
            # FAISS in LangChain doesn't support metadata filters natively; approximate by retrieving more and filtering.
            retr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 12})
            docs = retr.get_relevant_documents(user_q)
            cand_docs = [d for d in docs if d.metadata.get("candidate_key") == sel_key]
            if not cand_docs:
                # Fallback: manually scan chunks for candidate_key and keyword
                cand_docs = [d for d in st.session_state.chunks if d.metadata.get("candidate_key") == sel_key][:12]
        else:
            retr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
            cand_docs = retr.get_relevant_documents(user_q)

        context = "\n\n".join(
            [f"[p{d.metadata.get('page', '?')}] {d.page_content[:1200]}" for d in cand_docs]
        )
        answer = llm_answer(llm, user_q, context)
        st.session_state.chat.append({"role": "assistant", "content": answer, "sources": cand_docs})

# Display chat
for turn in st.session_state.chat[-8:]:
    if turn["role"] == "user":
        st.chat_message("user").write(turn["content"])
    else:
        with st.chat_message("assistant"):
            st.write(turn["content"])
            if turn.get("sources"):
                with st.expander("Show sources"):
                    for i, d in enumerate(turn["sources"], start=1):
                        meta = json.dumps({k:v for k,v in d.metadata.items() if k in ("source_file","page","candidate_key")})
                        st.markdown(f"**Source {i}** — {meta}")
                        st.text(d.page_content[:1500])

st.divider()

st.caption("Tips: Use the candidate search box to select a person first, then ask targeted questions. Set GROQ_API_KEY to enable LLM answers; otherwise the app returns extractive snippets.")


# ──────────────────────────────────────────────────────────────────────────────
# Notes
# ──────────────────────────────────────────────────────────────────────────────
# 1) Create a virtual env and install requirements.txt
#    python3 -m venv .venv && source .venv/bin/activate
#    pip install -r requirements.txt
# 2) (Optional) export GROQ_API_KEY=your_key
# 3) Run: streamlit run app.py
# 4) Upload bulk PDF resumes. Use the sidebar to persist/load the FAISS index.
# 5) Search/select a candidate, then ask questions.





# import streamlit as st
# from pypdf import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq

# # -------------------------------
# # Function to load and tag PDFs
# # -------------------------------
# def load_pdfs(uploaded_files):
#     docs = []
#     for file in uploaded_files:
#         pdf_reader = PdfReader(file)
#         candidate_name = file.name.replace(".pdf", "")  # ✅ use filename as candidate ID
#         for page_num, page in enumerate(pdf_reader.pages, start=1):
#             text = page.extract_text()
#             if text:
#                 docs.append({
#                     "text": text,
#                     "metadata": {"candidate": candidate_name, "page": page_num}
#                 })
#     return docs

# # -------------------------------
# # Streamlit App
# # -------------------------------
# st.title("📄 Resume RAG Chatbot with Candidate Search")

# uploaded_files = st.file_uploader("Upload bulk resumes (PDFs)", type="pdf", accept_multiple_files=True)

# if uploaded_files:
#     with st.spinner("Processing resumes..."):
#         docs = load_pdfs(uploaded_files)

#         # Split into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         all_chunks = []
#         for d in docs:
#             chunks = text_splitter.split_text(d["text"])
#             for chunk in chunks:
#                 all_chunks.append((chunk, d["metadata"]))

#         # Embeddings
#         embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#         # VectorDB with metadata
#         texts, metadatas = zip(*all_chunks)
#         vectordb = FAISS.from_texts(texts, embed, metadatas=metadatas)

#         # Retriever
#         retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#         # LLM
#         llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

#         # RetrievalQA
#         qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

#     st.success("Resumes processed successfully! ✅")

#     query = st.text_input("Ask about a candidate (e.g., 'Show Priya's experience')")

#     if query:
#         with st.spinner("Fetching candidate details..."):
#             # Force retrieval to focus on candidate name
#             candidate_docs = retriever.invoke(query)
#             if candidate_docs:
#                 response = qa.run(query)
#                 st.write(response)

#                 # Show source info
#                 st.markdown("### 📑 Sources:")
#                 for d in candidate_docs:
#                     st.write(f"Candidate: {d.metadata['candidate']}, Page: {d.metadata['page']}")
#             else:
#                 st.warning("❌ No matching candidate found. Try another name.")
