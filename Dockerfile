# syntax=docker/dockerfile:1
# Production container for the Local AI Document RAG Stack.
#
# Build with the default (text-only) extras:
#   docker build -t local-rag-stack:latest .
#
# Build with ColQwen / late-interaction visual retrieval:
#   docker build -t local-rag-stack:visual --build-arg LRS_EXTRAS=ollama,visual,qdrant .
#
# Run against a remote/co-located Ollama:
#   docker run -p 8000:8000 \
#     -e LRS_OLLAMA_HOST=http://host.docker.internal:11434 \
#     -v lrs-data:/app/data \
#     local-rag-stack:latest

FROM python:3.12-slim

ARG LRS_EXTRAS="ollama,qdrant"
ARG PIP_INDEX_URL="https://pypi.org/simple"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /app

# System dependencies:
# - build-essential / git: for building any source-distribution wheels
# - libgl1 / libglib2.0-0: OpenCV / Pillow runtime requirements used by extraction/visual backends
# - gosu: drop from root to a non-privileged user at container start so volumes can be chowned
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gosu \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and source first to maximize layer caching.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package and requested extras.
RUN pip install --upgrade pip wheel \
    && pip install ".[${LRS_EXTRAS}]"

# Create a non-root runtime user and the persistent data directory.
RUN useradd --create-home --uid 1000 lrs \
    && mkdir -p /app/data \
    && chown -R lrs:lrs /app /home/lrs

# Container-friendly defaults. The compose file and .env.example mirror these.
ENV LRS_DATA_DIR=/app/data \
    LRS_SQLITE_VEC_PATH=/app/data/lrs.vec.sqlite \
    LRS_GRAPH_STORE_PATH=/app/data/lrs.graph.sqlite \
    LRS_API_HOST=0.0.0.0 \
    LRS_API_PORT=8000 \
    LRS_EXTRACTION_ENGINE=plaintext \
    LRS_VECTOR_STORE=sqlite \
    LRS_OLLAMA_HOST=http://ollama:11434 \
    HF_HOME=/app/data/hf_cache

EXPOSE 8000

# The entrypoint fixes volume ownership before dropping privileges.
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["lrs", "serve"]
