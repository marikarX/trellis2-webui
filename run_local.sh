#!/usr/bin/env bash
# =============================================================================
# Local Run Script
# Runs the WebUI using the upstream TRELLIS.2 environment
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRELLIS_DIR="${TRELLIS_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/TRELLIS.2}"
VENV_DIR="${TRELLIS_DIR}/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ TRELLIS.2 venv not found at $VENV_DIR"
    echo "Run ./setup_local.sh first."
    exit 1
fi

# Activate the upstream venv
source "${VENV_DIR}/bin/activate"

# Environment
export TRELLIS_MODEL_PATH="${TRELLIS_MODEL_PATH:-$HOME/AI/Models/TRELLIS.2-4B}"
export PYTHONPATH="${TRELLIS_DIR}:${PYTHONPATH:-}"
export TRELLIS_EXECUTION_PRESET="${TRELLIS_EXECUTION_PRESET:-auto}"
export TRELLIS_CPU_OFFLOAD="${TRELLIS_CPU_OFFLOAD:-auto}"
export TRELLIS_ATTENTION_BACKEND="${TRELLIS_ATTENTION_BACKEND:-auto}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"

PORT="${TRELLIS_PORT:-8000}"

echo "🚀 Starting TRELLIS.2 WebUI..."
echo "   Model: $TRELLIS_MODEL_PATH"
echo "   Port:  $PORT"
echo ""

cd "$SCRIPT_DIR"
exec python -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
