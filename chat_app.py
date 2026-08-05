import sys
from pathlib import Path

import streamlit as st

# Allow imports from src/
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_chain import RAGChain
from src.document_manager import DocumentManager
from src.auth import register_user, login_user, decode_token

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)

# --------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "token" not in st.session_state:
    st.session_state.token = ""

# --------------------------------------------------------
# JWT Token Validation on Every Load
# --------------------------------------------------------
# If a token exists in session, validate it is still live.
# An expired or tampered token forces the user back to login.

if st.session_state.token:
    payload = decode_token(st.session_state.token)
    if payload is None:
        # Token expired or invalid — force re-login
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.token = ""
        st.warning("⏰ Your session has expired. Please log in again.")

# --------------------------------------------------------
# Authentication Gate
# --------------------------------------------------------

if not st.session_state.logged_in:

    st.title("🔐 Enterprise Knowledge Assistant")
    st.subheader("Login / Register")

    auth_mode = st.radio(
        "Authentication",
        ["Login", "Register"],
        horizontal=True,
    )

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password",
    )

    if st.button(auth_mode):

        if auth_mode == "Register":

            success, msg = register_user(
                username,
                password,
            )

            if success:
                st.success(msg)
            else:
                st.error(msg)

        else:

            success, result = login_user(
                username,
                password,
            )

            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                # Store JWT token — validated on each page load
                st.session_state.token = result
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(result)

    st.stop()

# --------------------------------------------------------
# Main App — authenticated users only
# --------------------------------------------------------

st.title("🤖 Enterprise Knowledge Assistant")
st.caption(f"Welcome **{st.session_state.username}**")

doc_manager = DocumentManager(
    st.session_state.username
)

# --------------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Pipeline Configuration")

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["hybrid", "dense", "sparse"],
        index=0,
    )

    use_reranker = st.toggle("Enable Reranker (Cross-Encoder)", value=False)
    use_query_expansion = st.toggle("Enable Multi-Query Expansion", value=False)
    top_k = st.slider("Top-K Sources", min_value=1, max_value=10, value=5)

    st.divider()
    st.header("📂 Document Manager")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md", "docx"],
    )

    if uploaded_file:
        saved_path = doc_manager.save_uploaded_file(uploaded_file)
        st.success(f"Saved {saved_path.name}")

    docs = doc_manager.list_documents()

    if docs:
        st.subheader("Indexed Documents")
        for doc in docs:
            st.write(f"📄 {doc.name}")
    else:
        st.info("No documents uploaded.")

    if st.button("🚀 Index Documents"):
        with st.spinner("Creating embeddings..."):
            success, output = doc_manager.ingest_documents()
        if success:
          st.success(output)
          st.cache_resource.clear()
          st.rerun()
        else:
            st.error(output)

    st.divider()
    st.write(f"👤 {st.session_state.username}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.token = ""
        st.rerun()

# --------------------------------------------------------
# Knowledge Base Check
# --------------------------------------------------------

bm25_file = Path("storage") / f"bm25_{st.session_state.username}.pkl"

rag = None

if bm25_file.exists():
    @st.cache_resource
    def load_rag(mode, rerank, expansion, k):
        return RAGChain(
            config_name=st.session_state.username,
            retrieval_mode=mode,
            use_reranker=rerank,
            use_query_expansion=expansion,
            top_k=k,
        )

    rag = load_rag(
        retrieval_mode,
        use_reranker,
        use_query_expansion,
        top_k,
    )
else:
    # ── Better first-time user experience ──────────────────────────────────
    st.divider()
    st.info(
        "### 📭 No knowledge base found\n\n"
        "It looks like this is your first time here, or you haven't indexed "
        "any documents yet.\n\n"
        "**To get started:**\n"
        "1. Upload one or more documents in the **📂 Document Manager** panel on the left\n"
        "2. Click **🚀 Index Documents** to build your personal knowledge base\n"
        "3. Come back here and start chatting!\n\n"
        "_Supported formats: PDF, TXT, Markdown, DOCX_"
    )
    st.stop()

# --------------------------------------------------------
# Chat History
# --------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------------
# Chat Interaction
# --------------------------------------------------------

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating grounded response..."):
            response = rag.query(question)

        # Groundedness Guardrail Alert
        if response.is_grounded:
            st.success(f"🟢 Grounded Answer (Confidence: {int(response.groundedness_score * 100)}%)")
        else:
            st.warning(f"⚠️ Potential Low Groundedness / Hallucination Risk (Confidence: {int(response.groundedness_score * 100)}%)")

        st.markdown(response.answer)

        st.divider()
        st.subheader("📚 Sources")

        for idx, doc in enumerate(response.retrieved_docs, start=1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "-")
            rerank = doc.get("rerank_score", 0)

            with st.expander(f"📄 Source {idx}: {source}"):
                col1, col2 = st.columns(2)
                col1.metric("Page", page)
                col2.metric("Rerank Score", f"{rerank:.2f}")

                st.markdown("### Retrieved Text")
                st.write(doc["text"])

        st.caption(
            f"Retrieval: {response.retrieval_latency_ms:.0f} ms | "
            f"Generation: {response.generation_latency_ms:.0f} ms | "
            f"Total: {response.total_latency_ms:.0f} ms"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
        }
    )