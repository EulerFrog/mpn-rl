#!/bin/bash
# Sets up a virtual environment and installs dependencies.
# Uses UV_CACHE_DIR on the same filesystem as the repo to enable hardlinking,
# which avoids slow cross-filesystem copies on NFS clusters.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_CACHE_DIR="${SCRIPT_DIR}/.uv-cache"

echo "Creating virtual environment..."
uv venv "${SCRIPT_DIR}/.venv"

echo "Installing dependencies (cache: ${UV_CACHE_DIR})..."
UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip install -e "${SCRIPT_DIR}"

echo ""
echo "Done. Activate with:"
echo "  source .venv/bin/activate"
