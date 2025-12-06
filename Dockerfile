# =========================
# Stage 1: Build environment
# =========================
# FROM python:3.11-slim AS base
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# Giữ log realtime
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cài đặt dependency hệ thống tối thiểu
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    poppler-utils \
    tesseract-ocr \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement và cài
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Mặc định chạy Streamlit app
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0"]
