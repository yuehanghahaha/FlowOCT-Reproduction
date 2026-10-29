#!/usr/bin/env bash
# Uses the configured author execution; activate your environment first.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
fov="${1:?Usage: bash scripts/run_experiment.sh 3m|6m [run_name]}"
case "$fov" in 3m|6m) ;; *) printf 'Choose 3m or 6m\n' >&2; exit 2 ;; esac
run_name="${2:-flowoct_${fov}_seed2026}"
python_bin="${FLOWOCT_PYTHON:-python}"
"$python_bin" -u run.py train --config "configs/$fov.json" --output "runs/$run_name"
# Inference reuses the configuration saved by this training run.
"$python_bin" -u run.py predict --config "runs/$run_name/config.json" --checkpoint "runs/$run_name/best.pt" --output "results/$run_name"
"$python_bin" -u run.py evaluate --predictions "results/$run_name" --output "evaluations/$run_name"
