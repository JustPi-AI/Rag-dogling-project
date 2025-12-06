import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_DISABLE_META_LOADING"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Docling ===
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# === LangChain splitters ===
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# === PyMuPDF fallback ===
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

import torch
from sentence_transformers import SentenceTransformer

torch.nn.Module.to_empty = lambda self, *args, **kwargs: self


# === ĐỊNH NGHĨA ĐƯỜNG DẪN API KEY ===
BASE_DIR = Path(__file__).resolve().parents[2]   # thư mục ragdocling_project/
API_DIR = BASE_DIR / "api"

# ================== DEFAULT CONFIG ==================
EMBED_MODEL = "intfloat/multilingual-e5-base" # base qua small
# EMBED_MODEL = "intfloat/e5-mistral-instruct"
RERANK_MODEL = "BAAI/bge-reranker-base" 
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1" # 

DEFAULT_K_CANDIDATES = 100
DEFAULT_K_FINAL = 40
CHUNK_METHODS = ("fixed", "recursive", "heading")
DEFAULT_CHUNK_METHOD = "recursive"


# ================== RERANKER ==================
# class _Reranker:
#     """Nhẹ RAM: fallback sang CrossEncoder nếu model lớn lỗi paging file."""

#     def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         try:
#             self.model = CrossEncoder(model_name, device=self.device)
#             print(f"[Reranker] Loaded CrossEncoder on {self.device}")
#         except Exception as e:
#             print(f"[Reranker] Failed to load model: {e}")
#             self.model = None

#     def rerank(self, query: str, passages: List[Dict[str, Any]], top_k=DEFAULT_K_FINAL):
#         if not passages or self.model is None:
#             return passages[:top_k]
#         pairs = [[query, p["text"]] for p in passages]
#         scores = self.model.predict(pairs)
#         for s, p in zip(scores, passages):
#             p["score_rerank"] = float(s)
#         passages.sort(key=lambda x: x.get("score_rerank", 0.0), reverse=True)
#         return passages[:top_k]
# ================== RERANKER ==================
class _Reranker:
    """Nhẹ RAM/GPU: fallback sang CPU khi CUDA lỗi hoặc hết VRAM."""

    _cached_model = None
    _cached_device = None
    _cached_name = None

    def __init__(self, model_name: str = RERANK_MODEL):
        # Ưu tiên GPU nếu có và không ép CPU
        device = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"

        # Dùng cache để không load nhiều lần
        if (_Reranker._cached_model is not None
                and _Reranker._cached_name == model_name):
            self.model = _Reranker._cached_model
            self.device = _Reranker._cached_device
            return

        try:
            self.model = CrossEncoder(model_name, device=device)
            self.device = device
        except RuntimeError as e:
            # Hết VRAM hoặc lỗi CUDA -> chuyển sang CPU
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.model = CrossEncoder(model_name, device="cpu")
            self.device = "cpu"

        _Reranker._cached_model = self.model
        _Reranker._cached_device = self.device
        _Reranker._cached_name = model_name

    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k=DEFAULT_K_FINAL):
        if not passages or self.model is None:
            return passages[:top_k]
        pairs = [[query, p["text"]] for p in passages]
        # Giới hạn batch để giảm đỉnh bộ nhớ
        scores = self.model.predict(pairs, batch_size=16 if self.device == "cuda" else 32)
        for s, p in zip(scores, passages):
            p["score_rerank"] = float(s)
        passages.sort(key=lambda x: x.get("score_rerank", 0.0), reverse=True)
        return passages[:top_k]


# ================== EMBEDDING INDEX ==================
# class _EmbedIndex:
#     def __init__(self, embed_model: str = EMBED_MODEL):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.embedder = SentenceTransformer(embed_model, device=self.device)
#         self.index = None
#         self.chunks: List[Dict[str, Any]] = []

#     def build(self, texts: List[str]) -> None:
#         if not texts:
#             return
#         embs = self.embedder.encode(
#             texts, batch_size=64, convert_to_numpy=True,
#             normalize_embeddings=True, show_progress_bar=True
#         )
#         dim = embs.shape[1]
#         self.index = faiss.IndexFlatIP(dim)
#         self.index.add(embs)
#         print(f"[EmbedIndex] Built with {len(texts)} chunks, dim={dim}")

#     def search(self, query: str, k=DEFAULT_K_CANDIDATES) -> List[int]:
#         if self.index is None or self.index.ntotal == 0:
#             return []
#         q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
#         scores, ids = self.index.search(q, k)
#         return ids[0].tolist() if len(ids) else []
# ================== EMBEDDING INDEX ==================
class _EmbedIndex:
    # Cache global để tránh load model nhiều lần
    _cached_embedder = None
    _cached_model_name = None
    _cached_device = None

    def __init__(self, embed_model: str = EMBED_MODEL):
        # Quyết định device ưu tiên
        prefer_cuda = torch.cuda.is_available() and not FORCE_CPU
        device = "cuda" if prefer_cuda else "cpu"

        # Dùng lại cache nếu cùng model
        if (_EmbedIndex._cached_embedder is not None
                and _EmbedIndex._cached_model_name == embed_model):
            self.embedder = _EmbedIndex._cached_embedder
            self.device = _EmbedIndex._cached_device
        else:
            # Thử load trên GPU trước, nếu có
            if device == "cuda":
                try:
                    self.embedder = SentenceTransformer(embed_model, device="cuda")
                    self.device = "cuda"
                except RuntimeError as e:
                    # Hết VRAM hoặc lỗi CUDA -> fallback CPU
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    self.embedder = SentenceTransformer(embed_model, device="cpu")
                    self.device = "cpu"
            else:
                self.embedder = SentenceTransformer(embed_model, device="cpu")
                self.device = "cpu"

            # Cập nhật cache
            _EmbedIndex._cached_embedder = self.embedder
            _EmbedIndex._cached_model_name = embed_model
            _EmbedIndex._cached_device = self.device

        self.index = None
        self.chunks: List[Dict[str, Any]] = []

    def _encode_batch_size(self) -> int:
        # Batch nhỏ hơn trên GPU 6GB để tránh đỉnh VRAM; CPU có thể để lớn hơn chút
        return 16 if self.device == "cuda" else 32

    def build(self, texts: List[str]) -> None:
        if not texts:
            return
        embs = self.embedder.encode(
            texts,
            batch_size=self._encode_batch_size(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        print(f"[EmbedIndex] Built with {len(texts)} chunks, dim={dim}, device={self.device}")

    def search(self, query: str, k=DEFAULT_K_CANDIDATES) -> List[int]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self.embedder.encode(
            [query],
            batch_size=self._encode_batch_size(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores, ids = self.index.search(q, k)
        return ids[0].tolist() if len(ids) else []


# ================== RAG DOCLING ==================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # -> backend/
DATA_DIR = BASE_DIR / "data"

class RAGDocling:
    def __init__(
        self,
        session_id: str,
        chunk_size: int = 550,
        chunk_overlap: int = 250,
        chunk_method: str = DEFAULT_CHUNK_METHOD
    ):
        self.session_id = session_id
        # self.sess_dir = Path("data") / session_id
        self.sess_dir = DATA_DIR / session_id     
        self.docs_dir = self.sess_dir / "docs"
        self.index_path = self.sess_dir / "embeddings.faiss"
        self.chunks_path = self.sess_dir / "embeddings.chunks.jsonl"

        # Cấu hình chunk
        chunk_method = (chunk_method or DEFAULT_CHUNK_METHOD).lower().strip()
        self.chunk_method = chunk_method if chunk_method in CHUNK_METHODS else DEFAULT_CHUNK_METHOD
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Docling converter (OCR off)
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        do_ocr=False,
                        ocr_engine_autodetect=False
                    )
                )
            }
        )

        self.reranker = _Reranker()
        self._index = None
        self.last_retrieved_ids: List[int] = []
        self.last_retrieved: List[str] = []


    # ================== CHUNKING METHODS ==================
    def _splitter(self):
        print(f"[Chunking] method={self.chunk_method}, size={self.chunk_size}, overlap={self.chunk_overlap}")
        if self.chunk_method == "heading":
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "header1"), ("##", "header2"), ("###", "header3")
                ]
            )
        elif self.chunk_method == "fixed":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[""]
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )


    # ================== LOAD + SPLIT DOCUMENTS ==================
    def _collect_chunks_from_docs(self) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        splitter = self._splitter()

        for f in sorted(self.docs_dir.glob("*")):
            if not f.is_file():
                continue
            try:
                suffix = f.suffix.lower()
                if suffix == ".pdf" and fitz is not None:
                    with fitz.open(f) as pdf:
                        for i, page in enumerate(pdf):
                            page_num = i + 1
                            text = (page.get_text("text") or "").strip()
                            if not text:
                                continue
                            docs = splitter.create_documents(
                                [text],
                                metadatas=[{"page": page_num, "title": f.name, "source": str(f)}]
                            )
                            for d in docs:
                                chunks.append({
                                    "text": f"[Page {page_num}] {d.page_content}",
                                    "title": f.name,
                                    "source": str(f),
                                    "page": page_num
                                })

                elif suffix == ".txt":
                    text = f.read_text(encoding="utf-8", errors="ignore")
                    docs = splitter.create_documents([text], metadatas=[{"page": 1, "title": f.name, "source": str(f)}])
                    for d in docs:
                        chunks.append({
                            "text": f"[Page 1] {d.page_content}",
                            "title": f.name,
                            "source": str(f),
                            "page": 1
                        })

                else:
                    doc = self.converter.convert(str(f))
                    d = getattr(doc, "document", doc)
                    if hasattr(d, "pages"):
                        for i, p in enumerate(d.pages):
                            page_num = i + 1
                            text = p.export_to_markdown() or ""
                            if not text.strip():
                                continue
                            docs = splitter.create_documents(
                                [text],
                                metadatas=[{"page": page_num, "title": f.name, "source": str(f)}]
                            )
                            for ddoc in docs:
                                chunks.append({
                                    "text": f"[Page {page_num}] {ddoc.page_content}",
                                    "title": f.name,
                                    "source": str(f),
                                    "page": page_num
                                })
                    elif hasattr(d, "export_to_markdown"):
                        text = d.export_to_markdown() or ""
                        docs = splitter.create_documents([text], metadatas=[{"page": 1, "title": f.name, "source": str(f)}])
                        for ddoc in docs:
                            chunks.append({
                                "text": f"[Page 1] {ddoc.page_content}",
                                "title": f.name,
                                "source": str(f),
                                "page": 1
                            })

            except Exception as e:
                print(f"[Docling Error] {f.name}: {e}")

        print(f"[RAGDocling] ✅ Collected {len(chunks)} chunks.")
        return chunks


    # ===== SAVE / LOAD CHUNKS =====
    def _save_chunks(self, chunks: List[Dict[str, Any]]):
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ===== BUILD / LOAD INDEX =====
    def build_or_refresh_index(self, force: bool = False):
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        files_now = sorted([
            p.name for p in self.docs_dir.glob("*")
            if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".html", ".txt"}
        ])
        prev_chunks = self._load_chunks()
        have_index = self.index_path.exists()
        need_rebuild = force or (not have_index) or (not prev_chunks)

        if not need_rebuild:
            prev_sources = {Path(ch["source"]).name for ch in prev_chunks}
            if set(files_now) != prev_sources:
                need_rebuild = True

        if not need_rebuild:
            print(f"[RAG] Using existing index ({len(prev_chunks)} chunks).")
            return

        print("[RAG] Building or refreshing FAISS index (page-aware)...")
        chunks = self._collect_chunks_from_docs()
        if not chunks:
            print("[RAG] Không tạo được chunk nào.")
            return

        texts = [c["text"] for c in chunks]
        embed_index = _EmbedIndex()
        embed_index.build(texts)
        embed_index.chunks = chunks

        faiss.write_index(embed_index.index, str(self.index_path))
        self._save_chunks(chunks)
        self._index = embed_index
        print(f"[RAG] Built index: {len(chunks)} chunks (page-aware, saved for reuse).")

        # thêm
        # Giảm giữ bộ nhớ GPU sau build
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


    def _ensure_loaded(self):
        if self._index is not None:
            return
        if not self.index_path.exists():
            self.build_or_refresh_index()
        if self.index_path.exists():
            idx = faiss.read_index(str(self.index_path))
            chunks = self._load_chunks()
            emb = _EmbedIndex()
            emb.index = idx
            emb.chunks = chunks
            self._index = emb
            print(f"[RAG] Loaded existing index ({len(chunks)} chunks).")

    # ===== ASK =====
    def ask(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        k_candidates: Optional[int] = None,
        k_final: Optional[int] = None,
    ) -> str:
        """
        Cho phép override K từ UI.
        """
        from openai import OpenAI

        try:
            with open(API_DIR / "gpt_api.txt", "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except Exception:
            return "Thiếu file gpt_api.txt."


        client = OpenAI(api_key=api_key)
        k_candidates = k_candidates or DEFAULT_K_CANDIDATES
        k_final = k_final or DEFAULT_K_FINAL

        # Ngữ cảnh hội thoại
        context_text = ""
        if history:
            for m in history[-3:]:
                role = "Người dùng" if m["role"] == "user" else "Trợ lý"
                context_text += f"{role}: {m['content']}\n"

        self._ensure_loaded()
        if self._index is None or self._index.index is None or self._index.index.ntotal == 0:
            return "Không tìm thấy index hoặc tài liệu trong session này."

        # === PAGE-AWARE QUERY NORMALIZATION (thêm 'page N' để giúp FAISS khớp)
        match = re.search(r"trang\s+(\d+)", query.lower())
        if match:
            page_num = match.group(1)
            query_aug = re.sub(r"trang\s+\d+", f"page {page_num}", query.lower())
        else:
            query_aug = query

        ids = self._index.search(query_aug, k=k_candidates)

        prelim = []
        for i in ids:
            if i is None or i >= len(self._index.chunks):
                continue
            ch = self._index.chunks[i]
            prelim.append({
                "idx": i,
                "text": ch["text"],
                "title": ch["title"],
                "source": ch["source"],
            })

        if not prelim:
            return "Không tìm thấy đoạn liên quan trong tài liệu."

        reranked = self.reranker.rerank(query, prelim, top_k=k_final)
        context = "\n\n---\n\n".join([
            f"[{i+1}] {p['title']}\nSource: {p['source']}\n{p['text'][:1500]}"
            for i, p in enumerate(reranked)
        ])

        self.last_retrieved_ids = [int(p["idx"]) for p in reranked if "idx" in p]
        self.last_retrieved = [Path(p["source"]).name for p in reranked]
        print(f"[RAG Metrics] last_retrieved_ids = {self.last_retrieved_ids}")
        print(f"[RAG Metrics] last_retrieved = {self.last_retrieved}")

        files_list = ", ".join(sorted([
            p.name for p in self.docs_dir.glob("*")
            if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".html", ".txt"}
        ])) or "Không có tài liệu nào trong session."

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
            {"role": "user", "content": f"{context_text}\nCâu hỏi hiện tại: {query}"},
        ]

        try:
            res = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
            return res.choices[0].message.content.strip()
        except Exception as e:
            return f"(OpenAI error) {e}\n\nContext:\n{context}"


__all__ = ["DEFAULT_K_CANDIDATES", "DEFAULT_K_FINAL", "RAGDocling"]
