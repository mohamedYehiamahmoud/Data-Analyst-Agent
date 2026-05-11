"""
streamlit_app.py — AutoAnalyst Frontend
========================================
A Streamlit UI that talks to the AutoAnalyst FastAPI backend.

Make sure the FastAPI server is running first:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

Then run this UI:
    streamlit run streamlit_app.py
"""

import io
import uuid
import requests
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AutoAnalyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 800; color: #1a1a2e; margin-bottom: 0.1rem; }
    .main-subtitle { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }

    .badge-success {
        background: #d1fae5; color: #065f46;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
    }
    .badge-error {
        background: #fee2e2; color: #991b1b;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600;
    }

    .chat-user {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem;
    }
    .chat-assistant {
        background: #f9fafb; border-left: 4px solid #10b981;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;
    }

    .col-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.5rem 0.8rem;
        margin: 4px 0; font-size: 0.85rem;
    }

    .info-box {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 8px; padding: 0.8rem;
        font-size: 0.83rem; color: #1e40af; margin-bottom: 0.5rem;
    }

    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State — persists across Streamlit reruns
# ─────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "columns" not in st.session_state:
    st.session_state.columns = None
if "df_head" not in st.session_state:
    st.session_state.df_head = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""


# ─────────────────────────────────────────────
# API Helper Functions
# ─────────────────────────────────────────────

def check_api_health() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def upload_csv(file_bytes: bytes, filename: str, session_id: str) -> dict:
    """Upload a CSV file to the backend. Returns file_id and column preview."""
    r = requests.post(
        f"{API_BASE}/upload",
        files={"file": (filename, io.BytesIO(file_bytes), "text/csv")},
        data={"session_id": session_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def analyze_query(query: str, file_id: str, session_id: str, max_retries: int) -> dict:
    """Send a question to the agent and return the report."""
    r = requests.post(
        f"{API_BASE}/analyze",
        data={
            "query": query,
            "file_id": file_id,
            "session_id": session_id,
            "max_retries": max_retries,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def describe_file(file_id: str) -> dict:
    """Get column descriptions for an uploaded file."""
    r = requests.get(f"{API_BASE}/describe/{file_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def send_email(session_id: str, email: str) -> dict:
    """Send the last report via email."""
    r = requests.post(
        f"{API_BASE}/email-report",
        json={"session_id": session_id, "email": email},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def clear_session(session_id: str):
    """Tell the backend to clear conversation history for this session."""
    try:
        requests.delete(f"{API_BASE}/session/{session_id}", timeout=5)
    except Exception:
        pass


def fetch_image(filename: str) -> bytes | None:
    """Fetch a generated chart image from the backend."""
    try:
        r = requests.get(f"{API_BASE}/images/{filename}", timeout=10)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 AutoAnalyst")
    st.markdown("*AI-powered CSV analysis*")
    st.divider()

    # ── API Health Status ──
    api_ok = check_api_health()
    if api_ok:
        st.markdown('<span class="badge-success">● API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-error">● API Offline</span>', unsafe_allow_html=True)
        st.warning("Start the backend:\n```\nuvicorn src.main:app --reload\n```")

    st.divider()

    # ── File Upload ──
    st.markdown("### 1. Upload your CSV")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Max 50 MB. Only CSV files are supported.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("📤 Upload & Analyze", use_container_width=True, type="primary"):
            if not api_ok:
                st.error("API is not running. Please start the backend first.")
            else:
                with st.spinner("Uploading..."):
                    try:
                        file_bytes = uploaded_file.read()

                        result = upload_csv(
                            file_bytes,
                            uploaded_file.name,
                            st.session_state.session_id,
                        )
                        st.session_state.file_id = result["file_id"]
                        st.session_state.filename = result["filename"]
                        st.session_state.chat_history = []
                        # FIX: clear any stale prefill on new file upload
                        st.session_state.prefill_query = ""

                        desc = describe_file(st.session_state.file_id)
                        st.session_state.columns = desc

                        try:
                            df_preview = pd.read_csv(io.BytesIO(file_bytes), nrows=10)
                            st.session_state.df_head = df_preview
                        except Exception:
                            st.session_state.df_head = None

                        st.success(f"✅ Uploaded: **{result['filename']}**")

                    except requests.HTTPError as e:
                        st.error(f"Upload failed: {e.response.json().get('detail', str(e))}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()

    # ── Dataset Info ──
    if st.session_state.file_id and st.session_state.columns:
        st.markdown("### 📋 Dataset Info")
        cols_data = st.session_state.columns
        st.markdown(f"**File:** {st.session_state.filename}")
        st.markdown(f"**Rows:** {cols_data['row_count']:,}")
        st.markdown(f"**Columns:** {len(cols_data['columns'])}")

        with st.expander("View Columns", expanded=False):
            for col_name, col_desc in cols_data["columns"].items():
                if "numeric" in col_desc:
                    icon = "🔢"
                elif "categorical" in col_desc:
                    icon = "🏷️"
                else:
                    icon = "📝"
                st.markdown(
                    f'<div class="col-card">{icon} <strong>{col_name}</strong><br>'
                    f'<span style="color:#666">{col_desc}</span></div>',
                    unsafe_allow_html=True,
                )

        st.divider()

    # ── Settings ──
    st.markdown("### ⚙️ Settings")
    max_retries = st.slider(
        "Max code retries",
        min_value=1,
        max_value=10,
        value=5,
        help="How many times the agent retries fixing broken code before giving up.",
    )

    # ── Session Controls ──
    st.divider()
    st.markdown("### 🔄 Session")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            clear_session(st.session_state.session_id)
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("New Session", use_container_width=True):
            clear_session(st.session_state.session_id)
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.file_id = None
            st.session_state.filename = None
            st.session_state.columns = None
            st.session_state.df_head = None
            st.session_state.chat_history = []
            st.session_state.prefill_query = ""
            st.rerun()

    st.divider()
    st.markdown(
        '<div class="info-box">💡 <strong>How it works</strong><br>'
        '1. Upload a CSV<br>'
        '2. Ask a question in plain English<br>'
        '3. The AI generates & runs Python code<br>'
        '4. You get a markdown report + charts</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔑 Session ID", expanded=False):
        st.code(st.session_state.session_id, language=None)


# ─────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────

st.markdown('<div class="main-title">📊 AutoAnalyst</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Upload a CSV and ask questions in plain English. '
    'The AI agent generates Python code, runs it, and gives you a report.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.file_id:
    st.markdown("---")
    st.markdown("#### 🔧 How the Agent Works")

    cols = st.columns(7)
    steps = [
        ("🛡️", "Guardrail",  "Validates your query for safety"),
        ("🎯", "Relevancy",  "Checks if question matches your data"),
        ("✏️", "Rewrite",    "Makes query more analytical"),
        ("🐍", "Code Gen",   "Writes pandas Python code"),
        ("🔒", "Security",   "Two-layer code safety check"),
        ("▶️", "Execute",    "Runs code in a sandbox"),
        ("📄", "Report",     "Formats results as markdown"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div style='text-align:center; background:#f8fafc; border:1px solid #e2e8f0; "
                f"border-radius:10px; padding:10px 6px;'>"
                f"<div style='font-size:1.5rem;'>{icon}</div>"
                f"<div style='font-weight:700; font-size:0.78rem; margin-top:4px;'>{title}</div>"
                f"<div style='font-size:0.7rem; color:#666; margin-top:3px;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.info("👈 **Upload a CSV file from the sidebar to get started.**")

else:
    # ── Data Preview ──
    if st.session_state.df_head is not None:
        with st.expander("🗂️ Data Preview — First 10 Rows", expanded=False):
            st.dataframe(st.session_state.df_head, use_container_width=True)

    # ── Conversation History ──
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑 <strong>You:</strong> {turn["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="chat-assistant">🤖 <strong>AutoAnalyst:</strong></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(turn["content"])

                if turn.get("images"):
                    img_cols = st.columns(min(len(turn["images"]), 3))
                    for i, img_filename in enumerate(turn["images"]):
                        img_bytes = fetch_image(img_filename)
                        if img_bytes:
                            with img_cols[i % 3]:
                                st.image(img_bytes, caption=img_filename, use_container_width=True)

                st.markdown("---")

    # ── Query Input ──
    st.markdown("### 💬 Ask a Question")

    # FIX: Example buttons write into `prefill_query` (a plain session state
    # key), NOT into `query_input` (the widget key). Writing directly into a
    # widget key before the widget is rendered raises a Streamlit ValueError.
    # Instead we consume `prefill_query` once — right before rendering the
    # text_area — and pass it as the `value` argument.
    st.markdown("**Quick examples:**")
    example_cols = st.columns(3)
    examples = [
        "What is the average value by category?",
        "Show me the top 5 rows with highest values",
        "Are there any missing values in the dataset?",
    ]
    for i, (col, example) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Stage the prefill; the text_area below will pick it up.
                st.session_state.prefill_query = example


    prefill_value = st.session_state.prefill_query
    # if prefill_value:
    #     st.session_state.prefill_query = ""  # consumed — reset immediately

    query = st.text_area(
        label="Your question:",
        value=prefill_value,
        placeholder="e.g. What is the average salary by department?",
        height=90,
        label_visibility="collapsed",
    )

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=False)

    if analyze_btn:
        # FIX: Guard against empty / whitespace-only queries.
        clean_query = query.strip() if query else ""

        if not clean_query:
            st.warning("⚠️ Please enter a question before clicking Analyze.")
        elif len(clean_query) < 3:
            st.warning("⚠️ Query is too short — please write at least 3 characters.")
        elif not api_ok:
            st.error("❌ API is offline. Please start the FastAPI backend.")
        else:
            # Append user turn to history immediately so it renders on rerun.
            st.session_state.chat_history.append({
                "role": "user",
                "content": clean_query,
            })

            # FIX: The st.status + API call block was orphaned inside a
            # commented-out else clause in the original file, so it never
            # ran. It is now correctly placed inside this else branch.
            with st.status("🤖 Agent is working...", expanded=True) as status:
                st.write("🛡️ Validating query...")
                st.write("🎯 Checking data relevancy...")
                st.write("🐍 Generating Python code...")
                st.write("🔒 Security scanning code...")
                st.write("▶️ Executing analysis...")
                st.write("📄 Writing report...")

                try:
                    result = analyze_query(
                        query=clean_query,
                        file_id=st.session_state.file_id,
                        session_id=st.session_state.session_id,
                        max_retries=max_retries,
                    )
                    status.update(label="✅ Analysis complete!", state="complete", expanded=False)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("report", "No report generated."),
                        "images": result.get("images", []),
                    })

                except requests.HTTPError as e:
                    try:
                        error_detail = e.response.json().get("detail", str(e))
                    except Exception:
                        error_detail = str(e)
                    status.update(label="❌ Error", state="error", expanded=False)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"⚠️ **Error:** {error_detail}",
                        "images": [],
                    })

                except Exception as e:
                    status.update(label="❌ Error", state="error", expanded=False)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"⚠️ **Unexpected error:** {str(e)}\n\nMake sure the backend is running.",
                        "images": [],
                    })

            st.rerun()

    # ── Email Report ──
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### 📧 Email this Report")
        col1, col2 = st.columns([3, 1])
        with col1:
            email_addr = st.text_input("Enter email address:", placeholder="your@email.com", label_visibility="collapsed")
        with col2:
            if st.button("✉️ Send Report", use_container_width=True):
                if not email_addr:
                    st.warning("Please enter an email.")
                elif "@" not in email_addr:
                    st.error("Invalid email address.")
                else:
                    with st.spinner("Sending..."):
                        try:
                            send_email(st.session_state.session_id, email_addr)
                            st.success("✅ Report sent!")
                        except Exception as e:
                            st.error(f"Failed to send: {e}")
