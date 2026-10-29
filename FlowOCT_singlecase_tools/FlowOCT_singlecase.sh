#!/usr/bin/env bash
set -Eeuo pipefail
PYTHON_BIN="${FLOWOCT_PYTHON:-/workspace/.conda/envs/lyh_flowoct/bin/python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'FlowOCT Python not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/FlowOCT_singlecase.py" "$@"
