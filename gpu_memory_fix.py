"""
gpu_memory_fix.py
Tự động tối ưu quản lý bộ nhớ GPU cho PyTorch (máy VRAM thấp, ví dụ 6GB)
Phù hợp cho RAG, LangChain, LangGraph, hoặc khi chạy embedding/rerank.
"""

import os
import torch

# 🔧 Cấu hình PyTorch tránh phân mảnh bộ nhớ
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_torch_memory():
    """
    Thiết lập an toàn cho PyTorch:
    - Dọn bộ nhớ GPU chưa dùng
    - Bật chế độ inference (tắt gradient)
    - In dung lượng GPU hiện tại
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free = total - reserved

        print(f"[GPU INFO] Thiết bị: {device_name}")
        print(f"[GPU INFO] Tổng VRAM: {total:.2f} GB | Đang dùng: {allocated:.2f} GB | Còn trống: {free:.2f} GB")

        # Dọn cache nếu còn bộ nhớ rác
        torch.cuda.empty_cache()
        print("[GPU INFO] Đã dọn cache GPU.\n")

    else:
        print("[INFO] Không phát hiện GPU, chạy ở chế độ CPU.\n")


def safe_inference(model, inputs):
    """
    Chạy model ở chế độ an toàn (tiết kiệm VRAM).
    Dùng khi gọi reranker hoặc embedding để tránh lỗi OOM.
    """
    with torch.no_grad():
        if torch.cuda.is_available():
            model.to("cuda")
        else:
            model.to("cpu")
        # outputs = model(**inputs)
        outputs = safe_inference(model, inputs)

    return outputs
