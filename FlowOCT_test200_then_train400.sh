#!/usr/bin/env bash
# Run inside flowoct_reproduction_a100; use the known training interpreter.
set -Eeuo pipefail
trap 'printf "\nStopped at line %s. Keep the error output and existing results.\n" "$LINENO" >&2' ERR

PYTHON_BIN=/workspace/.conda/envs/lyh_flowoct/bin/python
if [[ -n "${FLOWOCT_PYTHON:-}" ]]; then
    PYTHON_BIN="$FLOWOCT_PYTHON"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf "Training Python not found or not executable: %s\n" "$PYTHON_BIN" >&2
    exit 1
fi

PROJECT_DIR="$(pwd -P)"
WORK_DIR="$(dirname "$PROJECT_DIR")"
BASE="runs/flowoct_3m_200e"
EXTENDED="runs/flowoct_3m_400e_from200"
PREDICTIONS="results/flowoct_3m_best_after200"
EVALUATION="evaluations/flowoct_3m_best_after200"
TEST_LOG="$WORK_DIR/logs/flowoct_3m_best_after200_test.log"
TRAIN_LOG="$WORK_DIR/logs/flowoct_3m_201_to_400_train.log"
REVIEW="$WORK_DIR/uploads/flowoct_3m_best_after200_review.tar"

if [[ ! -f run.py || ! -f "$BASE/latest.pt" || ! -f "$BASE/best.pt" ]]; then
    printf "Enter /workspace/flowoct_work/flowoct_reproduction_a100 and check the 200-epoch checkpoints.\n" >&2
    exit 1
fi
if [[ ! -f "$WORK_DIR/logs/flowoct_3m_train.log" ]]; then
    printf "Original training log is missing: %s/logs/flowoct_3m_train.log\n" "$WORK_DIR" >&2
    exit 1
fi
for target in "$EXTENDED" "$PREDICTIONS" "$EVALUATION" "$TRAIN_LOG" "$REVIEW" "$REVIEW.partial" "$REVIEW.sha256"; do
    if [[ -e "$target" ]]; then
        printf "Output already exists; keeping it unchanged: %s\n" "$target" >&2
        exit 1
    fi
done

printf "\n[1/5] Check the training environment and 200-epoch checkpoints\n"
"$PYTHON_BIN" - <<'PY'
import json
import sys
from pathlib import Path
import torch
import torchio
import monai
import numpy
import skimage

print("Python:", sys.executable, flush=True)
print("torch:", torch.__version__, "torchio:", torchio.__version__, flush=True)

run = Path("runs/flowoct_3m_200e")
latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
assert latest["format"] == best["format"] == "flowoct-reproduction-v1"
assert latest["epoch"] == 200 and latest["global_step"] == 28000, "Expected a completed 200-epoch run"
assert cfg["epochs"] == 200 and cfg["fov"] == "3M", "Expected the original 200-epoch 3M run"
assert cfg["network_chunk"] == 0 and not cfg.get("diagnostic_limits"), "Expected the formal unchunked run"
assert 1 <= best["epoch"] <= 200
assert best["best_psnr"] == latest["best_psnr"], "Best PSNR differs between checkpoints"
history = [json.loads(line) for line in (run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
winner = max(history, key=lambda row: row["validation"]["metrics"]["volume_psnr"]["mean"])
assert history[-1]["epoch"] == 200
assert best["epoch"] == winner["epoch"], "Best checkpoint epoch differs from the validation log"
print("Test checkpoint epoch:", best["epoch"], "validation PSNR:", best["best_psnr"], flush=True)
print("Resume checkpoint epoch: 200; target epoch: 400", flush=True)
PY

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/uploads"
if [[ -e "$TEST_LOG" ]]; then
    PREVIOUS_LOG="$TEST_LOG.previous_$(date -u +%Y%m%d_%H%M%S)_$$"
    mv "$TEST_LOG" "$PREVIOUS_LOG"
    printf "Saved the previous test log: %s\n" "$PREVIOUS_LOG"
fi
printf "\n[2/5] Test the best checkpoint on all 50 test cases\n"
"$PYTHON_BIN" -u run.py predict --config "$BASE/config.json" --device cuda:0 \
    --checkpoint "$BASE/best.pt" --split test --output "$PREDICTIONS" \
    2>&1 | tee "$TEST_LOG"
"$PYTHON_BIN" -u run.py evaluate --predictions "$PREDICTIONS" --output "$EVALUATION" \
    2>&1 | tee -a "$TEST_LOG"

"$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
from pathlib import Path

pred = Path("results/flowoct_3m_best_after200")
info = json.loads((pred / "prediction.json").read_text(encoding="utf-8"))
summary = json.loads(Path("evaluations/flowoct_3m_best_after200/summary.json").read_text(encoding="utf-8"))
expected = {row["case_id"] for row in info["records"]}
actual = {p.stem for p in pred.glob("*.npz")}
with Path("evaluations/flowoct_3m_best_after200/per_case.csv").open(encoding="utf-8-sig", newline="") as stream:
    evaluated = list(csv.DictReader(stream))
assert info["split"] == "test" and not info["limited_diagnostic"]
assert len(info["records"]) == len(expected) == len(actual) == summary["cases"] == len(evaluated) == 50
assert actual == expected == {row["case_id"] for row in evaluated}
assert summary["protocol"]["checkpoint_sha256"] == info["checkpoint_sha256"]
assert hashlib.sha256(Path("runs/flowoct_3m_200e/best.pt").read_bytes()).hexdigest() == info["checkpoint_sha256"]
print("Verified 50 test cases; checkpoint epoch:", info["checkpoint_epoch"], flush=True)
PY

printf "\n[3/5] Package the 200-epoch test results for download\n"
tar -cf "$REVIEW.partial" \
    "$PREDICTIONS" "$EVALUATION" \
    "$BASE/best.pt" "$BASE/config.json" "$BASE/epochs.jsonl" \
    "$BASE/manifest.csv" "$BASE/environment.json" "$BASE/training_scope.json" \
    -C "$WORK_DIR" logs/flowoct_3m_train.log logs/flowoct_3m_best_after200_test.log
mv "$REVIEW.partial" "$REVIEW"
sha256sum "$REVIEW" > "$REVIEW.sha256"
printf "Ready to download: %s\nChecksum file: %s.sha256\n" "$REVIEW" "$REVIEW"
du -h "$REVIEW"

printf "\n[4/5] Copy training state to a separate 400-epoch run\n"
cp -a "$BASE" "$EXTENDED"

printf "\n[5/5] Resume training from epoch 201 through epoch 400\n"
"$PYTHON_BIN" -u run.py train --config "$EXTENDED/config.json" --device cuda:0 \
    --output "$EXTENDED" --resume "$EXTENDED/latest.pt" --epochs 400 \
    2>&1 | tee "$TRAIN_LOG"

printf "\nTraining finished. Extended run: %s/%s\n" "$PROJECT_DIR" "$EXTENDED"

