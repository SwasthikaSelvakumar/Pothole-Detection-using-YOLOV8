# ════════════════════════════════════════════════
#  RoadSense AI — Lightweight Dockerfile
#  Optimized for CPU deployment (no heavy CUDA)
#  SwasthikaSelvakumar
# ════════════════════════════════════════════════

FROM python:3.10-slim

LABEL maintainer="SwasthikaSelvakumar"
LABEL description="RoadSense AI — YOLOv8 + FastAPI + Multi-Agent Groq LLM"
LABEL version="2.0.0"

# ── System deps (fixed package names) ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Lightweight requirements ──────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    python-multipart==0.0.9 \
    Pillow==10.3.0 \
    numpy==1.26.4 \
    requests==2.32.3 \
    pydantic==2.7.1 \
    jinja2==3.1.4 \
    aiofiles==23.2.1

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir ultralytics==8.2.0
# ── Copy app files ────────────────────────────────
COPY app.py .
COPY templates/ templates/
# COPY best.pt .

# ── Non-root user ─────────────────────────────────
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]