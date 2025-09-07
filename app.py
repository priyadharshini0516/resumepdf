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
# Custom CSS Styling
# ──────────────────────────────────────────────────────────────────────────────
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #667eea;
        --primary-dark: #5a67d8;
        --secondary-color: #f093fb;
        --accent-color: #4facfe;
        --success-color: #48bb78;
        --warning-color: #ed8936;
        --error-color: #f56565;
        --background-light: #f8fafc;
        --background-card: #ffffff;
        --text-primary: #2d3748;
        --text-secondary: #718096;
        --border-light: #e2e8f0;
        --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.06);
        --shadow-medium: 0 8px 25px rgba(0, 0, 0, 0.1);
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-accent: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Main app styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Header styling */
    h1 {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--background-card);
        border-radius: 15px;
        box-shadow: var(--shadow-soft);
        padding: 1.5rem;
        margin: 1rem;
    }
    
    /* Card-like containers */
    .stContainer > div {
        background: var(--background-card);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-light);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stContainer > div:hover {
        box-shadow: var(--shadow-medium);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: var(--gradient-secondary);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        background: var(--gradient-accent);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    .stButton > button[kind="secondary"]:hover {
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 2px solid var(--border-light);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: var(--background-card);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid var(--border-light);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed var(--primary-color);
        border-radius: 15px;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-dark);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        transform: translateY(-2px);
    }
    
    /* Success/Info/Warning/Error messages */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(72, 187, 120, 0.05) 100%);
        border-left: 4px solid var(--success-color);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(79, 172, 254, 0.05) 100%);
        border-left: 4px solid var(--accent-color);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(237, 137, 54, 0.1) 0%, rgba(237, 137, 54, 0.05) 100%);
        border-left: 4px solid var(--warning-color);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.1) 0%, rgba(245, 101, 101, 0.05) 100%);
        border-left: 4px solid var(--error-color);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: var(--background-card);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-light);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stChatMessage:hover {
        box-shadow: var(--shadow-medium);
    }
    
    /* Chat message user */
    .stChatMessage[data-testid*="user"] {
        background: var(--gradient-primary);
        color: white;
        margin-left: 20%;
    }
    
    /* Chat message assistant */
    .stChatMessage[data-testid*="assistant"] {
        background: var(--background-card);
        margin-right: 20%;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--gradient-accent);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-weight: 500;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .streamlit-expanderHeader:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    .streamlit-expanderContent {
        border: 1px solid var(--border-light);
        border-radius: 0 0 12px 12px;
        padding: 1rem;
        background: var(--background-card);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: var(--gradient-primary);
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Divider styling */
    .stDivider {
        margin: 2rem 0;
    }
    
    .stDivider > div {
        background: var(--gradient-primary);
        height: 2px;
        border-radius: 1px;
    }
    
    /* Caption styling */
    .stCaption {
        color: var(--text-secondary);
        font-style: italic;
        margin: 0.5rem 0;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Custom section styling */
    .section-header {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header::before {
        content: '';
        width: 4px;
        height: 1.5rem;
        background: var(--gradient-primary);
        border-radius: 2px;
    }
    
    /* Metrics styling */
    .stMetric {
        background: var(--background-card);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-light);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Sidebar header */
    .css-1d391kg h2 {
        color: var(--primary-color);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Loading animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Smooth transitions for all interactive elements */
    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }
        
        .stContainer > div {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .stChatMessage[data-testid*="user"] {
            margin-left: 10%;
        }
        
        .stChatMessage[data-testid*="assistant"] {
            margin-right: 10%;
        }
    }
    
    /* Focus indicators for accessibility */
    *:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

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
            "🤖 **AI Response** (Extractive mode - set GROQ_API_KEY for enhanced answers)\n\n" + top
        )
    prompt = (
        "You are a helpful recruiter assistant. Answer the question using the given resume excerpts.\n"
        "If the answer depends on a specific candidate, only use those filtered excerpts.\n"
        "If not found, say 'Not found in resumes.'\n\n"
        f"Question: {question}\n\nResume Excerpts:\n{context}\n"
    )
    resp = llm.invoke(prompt)
    return "🤖 **AI Response**\n\n" + getattr(resp, "content", str(resp))


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume RAG Chatbot", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
load_custom_css()

# Header with emoji and styling
st.markdown('<h1>📄 Resume RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #718096; font-size: 1.1rem; margin-bottom: 2rem;">Powered by LangChain • Streamlit • AI</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)
    st.write("🔍 **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (local)")
    persist = st.checkbox("💾 Persist FAISS index to ./index", value=False)
    top_k = st.slider("📊 Retriever top_k", 2, 10, 4)
    st.divider()
    st.markdown("💡 **Tip:** Set `GROQ_API_KEY` environment variable for enhanced LLM responses!", help="Without API key, the app uses extractive summaries")

# Session state setup
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "registry" not in st.session_state:
    st.session_state.registry = {}
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Upload section with enhanced styling
st.markdown('<div class="section-header">1️⃣ Upload Resume PDFs</div>', unsafe_allow_html=True)
files = st.file_uploader(
    "📂 Drag and drop or browse PDF resumes", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="Upload multiple PDF resumes to build your candidate database"
)

col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("🚀 Build/Update Index", type="primary", disabled=not files):
        with st.spinner("🔄 Processing PDFs, extracting text, and building vector index..."):
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
        st.success(f"✅ Successfully indexed {len(st.session_state.chunks)} chunks from {len(st.session_state.registry)} candidate(s)!")

with col_b:
    if st.button("📥 Load Saved Index", disabled=st.session_state.vectorstore is not None):
        try:
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.load_local("index", embed, allow_dangerous_deserialization=True)
            st.success("✅ Index loaded successfully from ./index")
        except Exception as e:
            st.error(f"❌ Failed to load index: {e}")

st.divider()

# Candidate search section
st.markdown('<div class="section-header">2️⃣ Find Candidate</div>', unsafe_allow_html=True)
st.caption("🔍 Search by name, email, or phone number. We'll match against registry and embeddings.")
query_cand = st.text_input("🔎 Search candidate", placeholder="e.g., John Doe, john@email.com, +1234567890")

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
        sel_key = st.selectbox(
            "👤 Matching candidates", 
            options=exact_hits, 
            format_func=lambda k: f"👨‍💼 {st.session_state.registry[k]['name']} | 📧 {st.session_state.registry[k]['email']} | 📱 {st.session_state.registry[k]['phone']}"
        )
    else:
        st.info("🔍 No matching candidates found. Try a different search term or upload more resumes.")

if sel_key:
    info = st.session_state.registry[sel_key]
    st.success(f"✅ **Selected:** 👤 {info['name']} | 📧 {info['email']} | 📱 {info['phone']} | 📄 {info['source_file']}")

st.divider()

# Chat section
st.markdown('<div class="section-header">3️⃣ Ask Questions</div>', unsafe_allow_html=True)
st.caption("💬 Ask general questions or about the selected candidate. Candidate-specific queries are automatically filtered.")
user_q = st.text_input("💭 Your question", placeholder="e.g., What is John's experience with React? Show me Priya's projects.")

if "chat" not in st.session_state:
    st.session_state.chat = []

if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})

llm = get_llm()

if st.button("🎯 Ask Question", type="primary") and user_q:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload PDFs and build the index first.")
    else:
        with st.spinner("🧠 AI is thinking..."):
            if sel_key:
                # Filter chunks by candidate_key, then run vector search on that subset
                retr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 12})
                docs = retr.get_relevant_documents(user_q)
                cand_docs = [d for d in docs if d.metadata.get("candidate_key") == sel_key]
                if not cand_docs:
                    # Fallback: manually scan chunks for candidate_key
                    cand_docs = [d for d in st.session_state.chunks if d.metadata.get("candidate_key") == sel_key][:12]
            else:
                retr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                cand_docs = retr.get_relevant_documents(user_q)

            context = "\n\n".join(
                [f"[Page {d.metadata.get('page', '?')}] {d.page_content[:1200]}" for d in cand_docs]
            )
            answer = llm_answer(llm, user_q, context)
            st.session_state.chat.append({"role": "assistant", "content": answer, "sources": cand_docs})

# Display chat with enhanced styling
st.markdown('<div class="section-header">💬 Conversation</div>', unsafe_allow_html=True)

for i, turn in enumerate(st.session_state.chat[-8:]):
    if turn["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**You:** {turn['content']}")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("📚 View Sources & References", expanded=False):
                    for i, d in enumerate(turn["sources"], start=1):
                        meta = {k:v for k,v in d.metadata.items() if k in ("source_file","page","candidate_key")}
                        st.markdown(f"**📄 Source {i}** — `{json.dumps(meta)}`")
                        st.markdown(f"```\n{d.page_content[:1500]}{'...' if len(d.page_content) > 1500 else ''}\n```")

st.divider()

# Enhanced tips section
st.markdown('<div class="section-header">💡 Usage Tips</div>', unsafe_allow_html=True)

tip_cols = st.columns(3)
with tip_cols[0]:
    st.markdown("""
    **🎯 Candidate Search**
    - Use the search box to find specific candidates
    - Search by name, email, or phone number
    - Select a candidate for targeted questions
    """)

with tip_cols[1]:
    st.markdown("""
    **💬 Smart Questions**
    - Ask about skills, experience, projects
    - Compare candidates: "Who has React experience?"
    - Get summaries: "Tell me about John's background"
    """)

with tip_cols[2]:
    st.markdown("""
    **⚡ Performance**
    - Set `GROQ_API_KEY` for AI-powered answers
    - Use "Persist Index" to save processing time
    - Adjust top_k in sidebar for retrieval depth
    """)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #718096; padding: 1rem;">
    <p>Built with ❤️ using <strong>Streamlit</strong> • <strong>LangChain</strong> • <strong>FAISS</strong> • <strong>HuggingFace</strong></p>
    <p style="font-size: 0.9rem;">🚀 Upload resumes → 🔍 Search candidates → 💬 Ask questions → 📊 Get insights</p>
</div>
""", unsafe_allow_html=True)

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