#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_python="${FLOWOCT_BASE_PYTHON:-/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python}"
env_dir="${FLOWOCT_ENV:-/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct}"
if [[ ! -x "$env_dir/bin/python" ]]; then
  "$base_python" -m venv --system-site-packages "$env_dir"
fi
"$env_dir/bin/python" -m pip install -r "$root/requirements.txt"
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" -m pip freeze > "$root/verification/server_environment_freeze.txt"
printf 'FlowOCT Python: %s\n' "$env_dir/bin/python"
