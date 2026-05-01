# ════════════════════════════════════════════
#  Pothole Detection — Production Dockerfile
#  YOLOv8 + Flask + Groq LLM
#  SDG 3 & SDG 11
# ════════════════════════════════════════════

FROM python:3.10-slim

LABEL maintainer="venkatavamsi01"
LABEL description="Pothole Detection API — YOLOv8 + Groq LLM Agentic Alerts"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/venkatavamsi01/Pothole-Detection-using-YOLOV8"

# ── System dependencies ───────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application files ───────────────────
COPY app.py .

# ── Copy YOLOv8 weights ──────────────────────
# Place your trained best.pt in the same directory before building
# COPY best.pt .

# ── Non-root user for security ───────────────
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# ── Expose port ───────────────────────────────
EXPOSE 5000

# ── Health check ──────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# ── Launch with Gunicorn ──────────────────────
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:app"]
