#!/usr/bin/env bash
# install_token_export.sh
# Sets up a dedicated Python environment for Nymeria RGB-aligned SMPL token export.
#
# Usage: bash install_scripts/install_token_export.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv_token_export"

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
uv pip install "torch" --index-url "https://download.pytorch.org/whl/cpu"

echo "[INFO] Installing token-export Python dependencies..."
uv pip install \
    "numpy==1.26.4" \
    "scipy==1.15.3" \
    "joblib" \
    "tqdm" \
    "easydict" \
    "loguru" \
    "onnxruntime"

echo "[INFO] Installing local gear_sonic package without dependency re-resolution..."
uv pip install -e "gear_sonic" --no-deps

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
