#!/usr/bin/env bash
# install_token_export.sh
# Sets up a dedicated Python environment for Nymeria RGB-aligned SMPL token export.
#
# Usage: bash install_scripts/install_token_export.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv_token_export"
TOKEN_EXPORT_TORCH_SPEC="${TOKEN_EXPORT_TORCH_SPEC:-torch}"
TOKEN_EXPORT_TORCH_INDEX_URL="${TOKEN_EXPORT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

export UV_NO_CACHE=1

uv_without_package_indexes() (
    # Keep server-wide package source overrides from replacing the PyTorch CPU index.
    unset PIP_INDEX_URL
    unset PIP_EXTRA_INDEX_URL
    unset PIP_FIND_LINKS
    unset UV_INDEX
    unset UV_INDEX_URL
    unset UV_EXTRA_INDEX_URL
    unset UV_DEFAULT_INDEX
    unset UV_FIND_LINKS
    export UV_NO_CONFIG=1
    uv "$@"
)

ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found; installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary was not found on PATH."
        echo "        Add ~/.local/bin or ~/.cargo/bin to PATH and re-run this script."
        exit 1
    fi
fi
echo "[OK] $(uv --version)"

echo "[INFO] Installing uv-managed Python 3.10..."
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_token_export if present..."
rm -rf "$VENV_DIR"

echo "[INFO] Creating .venv_token_export..."
uv venv "$VENV_DIR" --python "$MANAGED_PY" --prompt gear_sonic_token_export
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[INFO] Installing CPU-only PyTorch..."
if [ -n "${TOKEN_EXPORT_TORCH_WHEEL:-}" ]; then
    if [ ! -f "$TOKEN_EXPORT_TORCH_WHEEL" ]; then
        echo "[ERROR] TOKEN_EXPORT_TORCH_WHEEL does not exist: $TOKEN_EXPORT_TORCH_WHEEL"
        exit 1
    fi
    echo "[INFO] Installing PyTorch from local wheel: $TOKEN_EXPORT_TORCH_WHEEL"
    uv pip install --no-cache "$TOKEN_EXPORT_TORCH_WHEEL"
else
    echo "[INFO] Installing $TOKEN_EXPORT_TORCH_SPEC from $TOKEN_EXPORT_TORCH_INDEX_URL"
    if ! uv_without_package_indexes pip install --no-cache "$TOKEN_EXPORT_TORCH_SPEC" --index-url "$TOKEN_EXPORT_TORCH_INDEX_URL"; then
        cat <<EOF
[ERROR] CPU-only PyTorch install failed.

If this server cannot reach the official PyTorch CPU wheel host, use one of:

  TOKEN_EXPORT_TORCH_WHEEL=/path/to/torch-...+cpu-...whl bash install_scripts/install_token_export.sh

  TOKEN_EXPORT_TORCH_INDEX_URL=https://your-internal-pytorch-cpu-index/simple \\
  TOKEN_EXPORT_TORCH_SPEC='torch==<version>+cpu' \\
  bash install_scripts/install_token_export.sh
EOF
        exit 1
    fi
fi

echo "[INFO] Installing token-export Python dependencies..."
uv pip install --no-cache \
    "numpy==1.26.4" \
    "scipy==1.15.3" \
    "joblib" \
    "tqdm" \
    "easydict" \
    "loguru" \
    "onnxruntime"

echo "[INFO] Installing local gear_sonic package without dependency re-resolution..."
uv pip install --no-cache -e "gear_sonic" --no-deps

echo "[INFO] Verifying token-export imports..."
python - <<'PY'
import importlib

modules = ["numpy", "scipy", "torch", "onnxruntime", "tqdm", "gear_sonic"]
for name in modules:
    importlib.import_module(name)

import torch

if torch.version.cuda is not None:
    raise RuntimeError(f"Expected CPU-only PyTorch, got CUDA build: {torch.version.cuda}")
print("[OK] Import check passed:", ", ".join(modules))
print("[OK] PyTorch build: CPU-only")
PY

cat <<EOF

Token export environment setup complete.

Activate it with:
  source gear_sonic_deploy/scripts/setup_token_export_env.sh

Then run:
  python gear_sonic_deploy/reference/export_nymeria_rgb_smpl_tokens.py /path/to/ny_batch --overwrite
EOF
