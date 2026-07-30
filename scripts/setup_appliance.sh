#!/usr/bin/env bash
set -euo pipefail

# Setup script for a Local RAG Stack appliance.
# Tested on macOS Apple Silicon. Adjust for Linux as needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

echo "==> Local RAG Stack appliance setup"
echo "    Project: $PROJECT_DIR"
echo "    Python:  $PYTHON_VERSION"

# 1. Ensure Python
if ! command -v python${PYTHON_VERSION} &>/dev/null; then
    echo "ERROR: python${PYTHON_VERSION} not found. Install Python ${PYTHON_VERSION} first."
    exit 1
fi

# 2. Create venv
cd "$PROJECT_DIR"
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment..."
    python${PYTHON_VERSION} -m venv .venv
fi
source .venv/bin/activate

# 3. Upgrade tooling
echo "==> Upgrading pip..."
pip install --upgrade pip wheel

# 4. Install core package + Ollama + Qdrant (no heavy ML wheels)
echo "==> Installing local-rag-stack core + Ollama + Qdrant..."
pip install -e ".[ollama,qdrant]"

# 5. Install extraction backend (Docling by default; Marker optional, GPL-3)
echo "==> Installing Docling extraction backend..."
pip install -e ".[docling]" || echo "WARNING: Docling install failed; plaintext fallback still works."

# 6. Install visual / multi-vector backend without forcing a torch downgrade.
#    colpali-engine pins torch<2.12, but newer torch versions usually work.
#    We install the runtime deps first, then colpali-engine with --no-deps.
echo "==> Installing visual (ColQwen) multi-vector backend..."
pip install torch torchvision transformers peft accelerate || true
pip install --no-deps colpali-engine || pip install -e ".[visual]"

# 7. Install Apple Silicon extras if on macOS ARM
if [[ "$OSTYPE" == "darwin"* && "$(uname -m)" == "arm64" ]]; then
    echo "==> Installing Apple Silicon MLX extras..."
    pip install -e ".[mlx]" || echo "WARNING: MLX extras install failed."
fi

# 8. Ensure Ollama models are pulled
if command -v ollama &>/dev/null; then
    echo "==> Pulling default Ollama models..."
    ollama pull qwen3:8b || true
    ollama pull qwen3-vl || true
    ollama pull qwen3-embedding:4b || true
else
    echo "WARNING: ollama not found. Install Ollama and pull models manually."
fi

# 9. Create data directory
mkdir -p "$DATA_DIR"

echo "==> Setup complete. Activate with:"
echo "    source .venv/bin/activate"
echo "    lrs serve"
