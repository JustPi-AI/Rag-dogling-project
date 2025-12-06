# backend/core/rag_graph.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import re, json

from langgraph.graph import StateGraph, END

# Reuse existing components
from backend.core.session_manager import get_session_paths
from backend.core.rag_docling import RAGDocling, DEFAULT_K_CANDIDATES, DEFAULT_K_FINAL
from backend.core.rag_metrics import RAGEvaluator

# Simple dict-based state
# Keys we use:
# - session_id: str
# - query: str
# - chunk_size, chunk_overlap: int
# - chunk_method: "recursive"|"fixed"|"heading"
# - k_candidates, k_final: int
# - page_tolerance: int
# - docs_dir: Path
# - rag: RAGDocling (runtime)
# - retrieved_ids: List[int]
# - reranked: List[Dict]
# - context: str
# - answer: str
# - evaluation: dict

def _page_aware_query(q: str) -> str:
    m = re.search(r"trang\s+(\d+)", (q or "").lower())
    if m:
        num = m.group(1)
        return re.sub(r"trang\s+\d+", f"page {num}", q.lower())
    return q

def node_ensure_index(state: Dict[str, Any]) -> Dict[str, Any]:
    session_id = state["session_id"]
    paths = get_session_paths(session_id)
    docs_dir = Path(paths["docs_dir"])
    state["docs_dir"] = docs_dir

    rag = RAGDocling(
        session_id=session_id,
        chunk_size=state.get("chunk_size", 550),
        chunk_overlap=state.get("chunk_overlap", 250),
        chunk_method=state.get("chunk_method", "recursive"),
    )
    # Build only if needed (same logic as ._ensure_loaded() + build_or_refresh_index)
    rag._ensure_loaded()
    if rag._index is None or rag._index.index is None or rag._index.index.ntotal == 0:
        rag.build_or_refresh_index(force=True)
        rag._ensure_loaded()

    state["rag"] = rag
    return state

def node_retrieve_rerank(state: Dict[str, Any]) -> Dict[str, Any]:
    rag: RAGDocling = state["rag"]
    query = state["query"]
    k_candidates = int(state.get("k_candidates", DEFAULT_K_CANDIDATES))
    k_final = int(state.get("k_final", DEFAULT_K_FINAL))

    if rag._index is None or rag._index.index is None or rag._index.index.ntotal == 0:
        state["answer"] = "Không tìm thấy index hoặc tài liệu trong session này."
        state["retrieved_ids"] = []
        state["reranked"] = []
        state["context"] = ""
        return state

    q_aug = _page_aware_query(query)

    ids = rag._index.search(q_aug, k=k_candidates)
    prelim = []
    for i in ids:
        if i is None or i >= len(rag._index.chunks):
            continue
        ch = rag._index.chunks[i]
        prelim.append({
            "idx": i,
            "text": ch.get("text", ""),
            "title": ch.get("title", ""),
            "source": ch.get("source", ""),
        })

    if not prelim:
        state["answer"] = "Không tìm thấy đoạn liên quan trong tài liệu."
        state["retrieved_ids"] = []
        state["reranked"] = []
        state["context"] = ""
        return state

    reranked = rag.reranker.rerank(query, prelim, top_k=k_final)
    context = "\n\n---\n\n".join([
        f"[{i+1}] {p.get('title','')}\nSource: {p.get('source','')}\n{p.get('text','')[:1500]}"
        for i, p in enumerate(reranked)
    ])

    state["retrieved_ids"] = [int(p["idx"]) for p in reranked if "idx" in p and isinstance(p["idx"], int)]
    state["reranked"] = reranked
    state["context"] = context
    return state

def node_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    rag: RAGDocling = state["rag"]
    query = state["query"]
    context = state.get("context", "")

    # List files for system prompt (same flavor as rag.ask)
    files_list = ", ".join(sorted([
        p.name for p in state["docs_dir"].glob("*")
        if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".html", ".txt"}
    ])) or "Không có tài liệu nào trong session."

    # Build messages like in rag.ask
    msgs = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý AI RAG tiếng Việt. "
                "Chỉ trả lời dựa trên các đoạn văn bản được truy xuất từ thư mục docs/ của người dùng.\n\n"
                f"Các tài liệu hiện có: {files_list}"
            ),
        },
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user", "content": f"Câu hỏi hiện tại: {query}"},
    ]

    # Reuse OpenAI client the same way as in rag.ask (api key via api/gpt_api.txt)
    try:
        # Import lazily to avoid top-level dependency
        from openai import OpenAI
        from pathlib import Path as _Path
        API_DIR = _Path(__file__).resolve().parents[2] / "api"
        with open(API_DIR / "gpt_api.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
        state["answer"] = res.choices[0].message.content.strip()
    except Exception as e:
        state["answer"] = f"(OpenAI error) {e}\n\nContext:\n{context}"

    return state

def node_evaluate(state: Dict[str, Any]) -> Dict[str, Any]:
    # Optional scoring (faithfulness/relevance); safe if API key is present
    try:
        evaluator = RAGEvaluator()
        data = [{
            "question": state.get("query", ""),
            "answer": state.get("answer", ""),
            "context": state.get("context", ""),
        }]
        scores = evaluator.evaluate_generation(data)
        state["evaluation"] = scores[0] if scores else {}
    except Exception as e:
        state["evaluation"] = {"error": str(e)}
    return state

# Build graph
def build_rag_graph():
    g = StateGraph(dict)

    g.add_node("ensure_index", node_ensure_index)
    g.add_node("retrieve_rerank", node_retrieve_rerank)
    g.add_node("generate", node_generate)
    g.add_node("evaluate", node_evaluate)

    g.set_entry_point("ensure_index")
    g.add_edge("ensure_index", "retrieve_rerank")
    g.add_edge("retrieve_rerank", "generate")
    g.add_edge("generate", "evaluate")
    g.add_edge("evaluate", END)

    return g.compile()
