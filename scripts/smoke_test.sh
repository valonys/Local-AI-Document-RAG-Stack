#!/usr/bin/env bash
set -euo pipefail

# Smoke test for the Local RAG Stack API.
# Requires the package to be installed and Ollama running with default models.

HOST="${LRS_HOST:-http://localhost:8000}"

echo "==> Smoke testing Local RAG Stack at $HOST"

# Health
echo "-> /health"
curl -s "$HOST/health" | head -c 500

echo ""
echo "-> /ingest with README"
curl -s -X POST "$HOST/ingest" \
    -F "file=@README.md" \
    -F 'metadata={"test": true}' | head -c 500

echo ""
echo "-> /documents"
curl -s "$HOST/documents" | head -c 500

echo ""
echo "-> /query"
curl -s -X POST "$HOST/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the Local AI Document RAG Stack?", "top_k": 3}' | head -c 1000

echo ""
echo "==> Smoke test complete"
