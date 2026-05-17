# ──────────────────────────────────────────────────────────────
# RAG Technical Documentation Assistant — Dockerfile
# Multi-stage build for smaller production image
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Build dependencies ───────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production image ─────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash raguser && \
    mkdir -p /app/data/chroma_db /app/data/sample_docs && \
    chown -R raguser:raguser /app

# Copy application source
COPY --chown=raguser:raguser app/ ./app/
COPY --chown=raguser:raguser data/ ./data/

# Switch to non-root user
USER raguser

# Environment defaults (can be overridden by docker run -e or docker-compose)
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    CHROMA_PERSIST_DIR=/app/data/chroma_db \
    CHROMA_COLLECTION=technical_docs \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    EMBEDDING_DEVICE=cpu \
    GROQ_MODEL=llama3-70b-8192 \
    LLM_TEMPERATURE=0.1 \
    LLM_MAX_TOKENS=2048 \
    RETRIEVAL_TOP_K=5 \
    LOG_LEVEL=INFO

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Entry point
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
