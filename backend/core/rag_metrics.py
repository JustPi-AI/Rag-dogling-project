import json
import re
import numpy as np
from collections import Counter
from openai import OpenAI
from pathlib import Path

from gpu_memory_fix import setup_torch_memory, safe_inference

setup_torch_memory()

# === FIXED: Đường dẫn thư mục API ===
BASE_DIR = Path(__file__).resolve().parents[2]   # thư mục ragdocling_project/
API_DIR = BASE_DIR / "api"


class RAGEvaluator:
    """
    - Retrieval metrics (page-level): Precision@k, Recall@k, MRR
    - Generation metrics: Faithfulness, Relevance (GPT)
    """

    def __init__(self, api_key_path: Path = API_DIR / "gpt_api.txt"):
        self.client = None
        try:
            with open(api_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            self.client = OpenAI(api_key=api_key)
            print("[RAG Metrics] Đã khởi tạo OpenAI client.")
        except Exception as e:
            print(f"[RAG Metrics] Không thể khởi tạo OpenAI client ({e})")

    # -------- Retrieval base --------
    def precision_at_k(self, retrieved, relevant, k=5) -> float:
        if not retrieved or k <= 0:
            return 0.0
        return len(set(retrieved[:k]) & set(relevant)) / k

    def recall_at_k(self, retrieved, relevant, k=5) -> float:
        if not relevant:
            return 0.0
        return len(set(retrieved[:k]) & set(relevant)) / len(relevant)

    def mean_reciprocal_rank(self, retrieved, relevant) -> float:
        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1 / i
        return 0.0

    # -------- Page-level evaluation --------
    def evaluate_retrieval_page_level(self, all_chunks, retrieved_ids, query, k=5, tolerance=1):
        """
        Xác định page đích:
          1) Nếu query có 'trang N' → target = {N}
          2) Nếu không, chọn page phổ biến nhất trong retrieved_ids
        Relevance = các chunk có page ∈ [N - tol, N + tol]
        """
        # 1) Tìm trong query
        match = re.search(r"trang\s+(\d+)", (query or "").lower())
        if match:
            target_pages = [int(match.group(1))]
        else:
            # 2) Suy luận từ retrieved
            retrieved_pages = [
                int(all_chunks[i].get("page", -1))
                for i in retrieved_ids if 0 <= i < len(all_chunks)
            ]
            retrieved_pages = [p for p in retrieved_pages if p > 0]
            if not retrieved_pages:
                print("[RAG Metrics] Không có page trong retrieved để suy luận.")
                return {"Precision@5": 0.0, "Recall@5": 0.0, "MRR": 0.0}
            target_pages = [Counter(retrieved_pages).most_common(1)[0][0]]

        # 3) Build relevant_ids theo ± tolerance
        window = set()
        for tp in target_pages:
            for d in range(-tolerance, tolerance + 1):
                window.add(tp + d)

        relevant_ids = [
            idx for idx, ch in enumerate(all_chunks)
            if isinstance(ch.get("page"), (int, float)) and int(ch["page"]) in window
        ]

        if not relevant_ids:
            print(f"[RAG Metrics] Không có chunk nào cho page {target_pages} (±{tolerance}).")
            return {"Precision@5": 0.0, "Recall@5": 0.0, "MRR": 0.0}

        prec = self.precision_at_k(retrieved_ids, relevant_ids, k)
        rec = self.recall_at_k(retrieved_ids, relevant_ids, k)
        mrr = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)

        print(f"[RAG Metrics] target_pages = {target_pages}, tolerance=±{tolerance}")
        print(f"[RAG Metrics] relevant_ids = {len(relevant_ids)}")
        print(f"[Retrieval Metrics] P@{k}={prec:.2f}, R@{k}={rec:.2f}, MRR={mrr:.2f}")

        return {"Precision@5": round(prec, 2), "Recall@5": round(rec, 2), "MRR": round(mrr, 2)}

    # -------- Generation evaluation --------
    def gpt_score(self, question: str, answer: str, context: str) -> dict:
        if not self.client:
            return {"faithfulness": 0, "relevance": 0, "feedback": "Không có GPT client."}

        prompt = f"""
        Bạn là chuyên gia đánh giá hệ thống RAG.
        Hãy đánh giá câu trả lời dưới đây theo hai tiêu chí:

        1. Faithfulness (Trung thực với context, không bịa)
        2. Relevance (Liên quan và trả đúng trọng tâm câu hỏi)

        Trả về JSON hợp lệ:
        {{
          "faithfulness": <1-5>,
          "relevance": <1-5>,
          "feedback": "<nhận xét ngắn>"
        }}

        --- DỮ LIỆU ---
        Câu hỏi: {question}
        Câu trả lời: {answer}
        Context: {context}
        """

        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            raw = res.choices[0].message.content.strip()
            raw = raw.strip("`").replace("json", "").replace("```", "").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # fallback parse
                f = re.search(r"Faithfulness[^\d]*(\d)", raw)
                r = re.search(r"Relevance[^\d]*(\d)", raw)
                return {
                    "faithfulness": int(f.group(1)) if f else 0,
                    "relevance": int(r.group(1)) if r else 0,
                    "feedback": raw
                }
        except Exception as e:
            return {"faithfulness": 0, "relevance": 0, "feedback": f"Lỗi GPT: {e}"}

    def evaluate_generation(self, dataset: list) -> list:
        if not self.client:
            return [{"faithfulness": 0, "relevance": 0, "feedback": "Thiếu API key"}]

        results = []
        for i, d in enumerate(dataset, start=1):
            q, a, c = d.get("question", ""), d.get("answer", ""), d.get("context", "")
            score = self.gpt_score(q, a, c)
            print(f"[{i}] Faithfulness={score.get('faithfulness')} | Relevance={score.get('relevance')} | {score.get('feedback')}")
            results.append(score)

        f_avg = np.mean([r["faithfulness"] for r in results])
        r_avg = np.mean([r["relevance"] for r in results])
        print(f"\nTrung bình Faithfulness: {f_avg:.2f}")
        print(f"Trung bình Relevance: {r_avg:.2f}")
        return results
