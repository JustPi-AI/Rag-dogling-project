import os
import json
from typing import List

from pathlib import Path

# === FIXED: Đường dẫn data gốc ===
BASE_DIR = Path(__file__).resolve().parents[1]  # -> backend/
ROOT_DATA = str(BASE_DIR / "data")

# ROOT_DATA = "data"

def ensure_session_tree(session_id: str) -> None:
    sess_dir = os.path.join(ROOT_DATA, session_id)
    docs_dir = os.path.join(sess_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    hist_path = os.path.join(sess_dir, "chat_history.json")
    if not os.path.exists(hist_path):
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def list_sessions() -> List[str]:
    if not os.path.exists(ROOT_DATA):
        os.makedirs(ROOT_DATA, exist_ok=True)
        return []
    sessions = []
    for name in os.listdir(ROOT_DATA):
        sess_dir = os.path.join(ROOT_DATA, name)
        if os.path.isdir(sess_dir) and os.path.exists(os.path.join(sess_dir, "docs")):
            sessions.append(name)
    sessions.sort()
    return sessions

def get_session_paths(session_id: str) -> dict:
    sess_dir = os.path.join(ROOT_DATA, session_id)
    docs_dir = os.path.join(sess_dir, "docs")
    index_path = os.path.join(sess_dir, "embeddings.faiss")
    chunks_path = os.path.join(sess_dir, "embeddings.chunks.jsonl")
    hist_path = os.path.join(sess_dir, "chat_history.json")
    ensure_session_tree(session_id)
    return {
        "sess_dir": sess_dir,
        "docs_dir": docs_dir,
        "index_path": index_path,
        "chunks_path": chunks_path,
        "history_path": hist_path,
    }

def load_history(session_id: str):
    paths = get_session_paths(session_id)
    if os.path.exists(paths["history_path"]):
        with open(paths["history_path"], "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(session_id: str, history):
    paths = get_session_paths(session_id)
    with open(paths["history_path"], "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
