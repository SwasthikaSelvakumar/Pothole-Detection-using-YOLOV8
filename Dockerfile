# ════════════════════════════════════════════════
#  RoadSense AI v2.0 — CUDA-Enhanced Dockerfile
#  YOLOv8m + FastAPI + Multi-Agent Groq LLM
#  SDG 3 & SDG 11 | SwasthikaSelvakumar
# ════════════════════════════════════════════════

# Use CUDA base image for GPU support
# For CPU-only deployment, replace with: python:3.10-slim
FROM python:3.10-slim

LABEL maintainer="SwasthikaSelvakumar"
LABEL description="RoadSense AI v2.0 — CUDA YOLOv8m + FastAPI + Multi-Agent LLM"
LABEL version="2.0.0"
LABEL cuda.enabled="true"
LABEL org.opencontainers.image.source="https://github.com/SwasthikaSelvakumar/Pothole-Detection-using-YOLOV8"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements_cuda.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app_cuda.py app.py
COPY templates/ templates/

# Copy trained model weights
# COPY best.pt .

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

# Health check hits /health which returns CUDA stats
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Launch with uvicorn (async ASGI — handles concurrent requests)
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--workers", "1", \
     "--timeout-keep-alive", "120"]
