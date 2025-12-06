# backend/app.py
import sys, os
# === FIX PATH để import backend ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import uuid
from datetime import datetime
from pathlib import Path

from backend.core.rag_metrics import RAGEvaluator
from backend.core.session_manager import (
    ensure_session_tree,
    list_sessions,
    get_session_paths,
    load_history,
    save_history,
)
from backend.core.rag_docling import RAGDocling, DEFAULT_K_CANDIDATES, DEFAULT_K_FINAL

# NEW: import LangGraph workflow
from backend.core.rag_graph import build_rag_graph

st.set_page_config(page_title="Session-based RAG (LangGraph)", layout="wide")
st.title("Chatbot đọc hiểu tài liệu • LangChain + LangGraph")

# ===== Sidebar: Session =====
st.sidebar.header("Hội thoại (Session)")

all_sessions = list_sessions()
if "session_id" not in st.session_state:
    st.session_state.session_id = all_sessions[0] if all_sessions else None

if all_sessions:
    try:
        idx = all_sessions.index(st.session_state.session_id) if st.session_state.session_id in all_sessions else 0
    except Exception:
        idx = 0
    current = st.sidebar.selectbox("Chọn session", all_sessions, index=idx)
    st.session_state.session_id = current
else:
    st.sidebar.info("Chưa có session. Hãy tạo mới.")

col1, col2 = st.sidebar.columns([1, 1])
with col1:
    if st.button("+ Tạo session"):
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
        ensure_session_tree(new_id)
        st.session_state.session_id = new_id
        st.rerun()
with col2:
    with st.expander("Đổi tên", expanded=False):
        new_name = st.text_input("Tên mới", key="rename_input")
        if st.button("Lưu tên mới"):
            old_path = Path("backend/data") / st.session_state.session_id
            new_path = Path("backend/data") / new_name
            try:
                import shutil
                shutil.copytree(old_path, new_path, dirs_exist_ok=True)
                shutil.rmtree(old_path)
                st.session_state.session_id = new_name
                st.success(f"Đã đổi tên thành {new_name}")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi đổi tên: {e}")

if not st.session_state.session_id:
    st.stop()

# ===== Sidebar: Chunking Settings =====
st.sidebar.markdown("### Cấu hình Chunking")
chunk_size = st.sidebar.slider(
    "Chunk size (kích thước mỗi đoạn)",
    min_value=300, max_value=1500, value=800, step=10,
)
chunk_overlap = st.sidebar.slider(
    "Chunk overlap (chồng lặp giữa các đoạn)",
    min_value=0, max_value=400, value=300, step=10,
)
chunk_method = st.sidebar.selectbox(
    "Phương pháp Chunking",
    options=["recursive", "fixed", "heading"],
    index=0,
    help="recursive: ngữ cảnh; fixed: đều ký tự; heading: theo tiêu đề."
)

# === NEW: Slider chỉnh K và page tolerance ===
st.sidebar.markdown("---")
k_candidates = st.sidebar.slider(
    "Top-K candidates (FAISS)",
    50, 1000, 300,step=10
)
k_final = st.sidebar.slider(
    "Top-K final (reranker)",
    1, 100, 5,step=1
)
page_tolerance = st.sidebar.slider(
    "±Page tolerance (đánh giá)",
    0, 3, 1,
)

# ===== Sidebar: Upload =====
paths = get_session_paths(st.session_state.session_id)
st.sidebar.header("Tài liệu của session")
uploaded_files = st.sidebar.file_uploader(
    "Tải nhiều tài liệu (PDF, DOCX, PPTX, HTML, TXT)",
    type=["pdf", "docx", "pptx", "html", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uf in uploaded_files:
        save_path = os.path.join(paths["docs_dir"], uf.name)
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
    st.sidebar.success(f"Đã lưu {len(uploaded_files)} file vào: {paths['docs_dir']}")
    # Build nhanh bằng RAGDocling (tận dụng sẵn logic build)
    rag_tmp = RAGDocling(
        session_id=st.session_state.session_id,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunk_method=chunk_method
    )
    rag_tmp.build_or_refresh_index(force=True)

# Manual rebuild
if st.sidebar.button("Build/Rebuild index"):
    rag_tmp = RAGDocling(
        session_id=st.session_state.session_id,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunk_method=chunk_method
    )
    rag_tmp.build_or_refresh_index(force=True)
    st.sidebar.success("Đã build/rebuild index cho session hiện tại.")

# Docs list
with st.sidebar.expander("Danh sách tài liệu trong session", expanded=True):
    files = os.listdir(paths["docs_dir"]) if os.path.exists(paths["docs_dir"]) else []
    if files:
        st.write("**docs/**:")
        for name in sorted(files):
            st.write("- ", name)
    else:
        st.info("Chưa có tài liệu trong session này.")

# ===== Chat history =====
if "messages" not in st.session_state:
    st.session_state.messages = load_history(st.session_state.session_id)
else:
    if st.session_state.get("__last_session__") != st.session_state.session_id:
        st.session_state.messages = load_history(st.session_state.session_id)
st.session_state.__last_session__ = st.session_state.session_id

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ===== Chat input =====
user_msg = st.chat_input("Nhập câu hỏi về tài liệu trong session này...")

if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # === LangGraph: invoke workflow
    rag_graph = build_rag_graph()
    state_in = {
        "session_id": st.session_state.session_id,
        "query": user_msg,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_method": chunk_method,
        "k_candidates": k_candidates,
        "k_final": k_final,
        "page_tolerance": page_tolerance,
    }

    with st.chat_message("assistant"):
        with st.spinner("Đang chạy workflow (LangGraph)..."):
            state_out = rag_graph.invoke(state_in)
        answer = state_out.get("answer", "Không có câu trả lời.")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_history(st.session_state.session_id, st.session_state.messages)

    # --- Hiển thị đánh giá nếu có ---
    eval_block = state_out.get("evaluation", {})
    faith = eval_block.get("faithfulness")
    rel = eval_block.get("relevance")
    feedback = eval_block.get("feedback")
    if isinstance(faith, (int, float)) and isinstance(rel, (int, float)):
        st.caption(f"**Faithfulness:** {faith}/5 | **Relevance:** {rel}/5")
        if feedback:
            st.caption(f"[Feedback] *{feedback}*")

# --- ĐÁNH GIÁ RETRIEVAL (Page-level) ---
with st.expander("Đánh giá Retrieval Metrics (Page-level)", expanded=False):
    try:
        ev = RAGEvaluator()
        # Đọc FAISS/chunks từ RAGDocling để tính
        rag_eval = RAGDocling(
            session_id=st.session_state.session_id,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunk_method=chunk_method
        )
        rag_eval._ensure_loaded()
        if "messages" in st.session_state and st.session_state.messages:
            last_user = None
            for m in reversed(st.session_state.messages):
                if m["role"] == "user":
                    last_user = m["content"]
                    break
        else:
            last_user = ""
        if rag_eval._index and rag_eval._index.index and last_user:
            top_ids = rag_eval._index.search(last_user, k=k_final)
            retrieved_ids = [i for i in top_ids if i is not None]
            all_chunks = rag_eval._index.chunks or []
            if all_chunks and retrieved_ids:
                score = ev.evaluate_retrieval_page_level(
                    all_chunks=all_chunks,
                    retrieved_ids=retrieved_ids,
                    query=last_user,
                    k=k_final,
                    tolerance=page_tolerance,
                )
                st.markdown(
                    f"**Precision@{k_final}:** `{score['Precision@5']:.2f}`  \n"
                    f"**Recall@{k_final}:** `{score['Recall@5']:.2f}`  \n"
                    f"**MRR:** `{score['MRR']:.2f}`"
                )
            else:
                st.info("Chưa đủ dữ liệu để đánh giá. Hãy đặt câu hỏi trước.")
        else:
            st.info("Chưa có index/chunks.")
    except Exception as e:
        st.warning(f"[WARN] Lỗi khi đánh giá retrieval (page-level): {e}")


import streamlit as st
import time

# st.set_page_config(page_title="RAGDocling", layout="wide")
# st.title("RAGDocling đã khởi động thành công")

# st.write("Ứng dụng đang chạy nền và sẵn sàng nhận truy vấn.")
# st.write("Nếu bạn thấy trang này, backend đã được load thành công.")

# Giữ tiến trình Streamlit luôn hoạt động
while True:
    time.sleep(1)
