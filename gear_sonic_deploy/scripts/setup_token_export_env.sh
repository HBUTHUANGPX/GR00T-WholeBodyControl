#!/usr/bin/env bash
# Source this file before running Nymeria RGB-aligned SMPL token export:
#   source gear_sonic_deploy/scripts/setup_token_export_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced, not executed:"
    echo "  source gear_sonic_deploy/scripts/setup_token_export_env.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv_token_export"
VENV_ACTIVATE="$REPO_ROOT/.venv_token_export/bin/activate"

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Token export venv not found: $VENV_DIR"
    echo "Create it first:"
    echo "  bash install_scripts/install_token_export.sh"
    return 1
fi

# shellcheck disable=SC1091
source "$VENV_ACTIVATE"

export TOKEN_EXPORT_REPO_ROOT="$REPO_ROOT"
export TOKEN_EXPORT_ENCODER_MODEL="$REPO_ROOT/gear_sonic_deploy/policy/release/model_encoder.onnx"

case ":${PYTHONPATH:-}:" in
    *":$REPO_ROOT:"*) ;;
    *) export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

echo "Token export environment ready: $VIRTUAL_ENV"
echo "Repository root: $TOKEN_EXPORT_REPO_ROOT"
