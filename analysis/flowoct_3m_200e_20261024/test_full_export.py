"""Verify the CPU projection exporter against one existing local epoch-98 case."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = ROOT / "FlowOCT_export_FULL_review.py"
PRED = Path(r"D:\data\FSdata\results\flowoct_3m_best_preview")
EVAL = Path(r"D:\data\FSdata\evaluations\flowoct_3m_best_preview")
REFERENCE = ROOT / "analysis/flowoct_3m_best_test_20261022/octa500_full/full_metrics.json"
OUTPUT = HERE / "full_export_check_epoch98"

def main():
    original = json.loads((PRED / "prediction.json").read_text(encoding="utf-8"))
    case_record = original["records"][0]
    case = case_record["case_id"]
    source_npz = PRED / f"{case}.npz"
    source_hash = hashlib.sha256(source_npz.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="full_export_fixture_", dir=HERE) as folder:
        temp = Path(folder).resolve()
        assert temp.is_relative_to(HERE.resolve())
        pred = temp / "predictions"
        evaluation = temp / "evaluation"
        pred.mkdir()
        evaluation.mkdir()
        os.link(source_npz, pred / source_npz.name)
        info = copy.deepcopy(original)
        info["verification_fixture"] = "One existing case selected only for local exporter verification; not a new full test run."
        info["records"] = [case_record]
        info["summary"]["cases"] = 1
        (pred / "prediction.json").write_text(json.dumps(info), encoding="utf-8")
        summary = json.loads((EVAL / "summary.json").read_text(encoding="utf-8"))
        summary["cases"] = 1
        (evaluation / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with (EVAL / "per_case.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            selected = [row for row in reader if row["case_id"] == case]
        assert len(selected) == 1
        with (evaluation / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(selected)
        command = [sys.executable, "-B", "-X", "utf8", str(SCRIPT),
                   "--predictions", str(pred), "--evaluation", str(evaluation), "--output", str(OUTPUT)]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((OUTPUT / "full_metrics.json").read_text(encoding="utf-8"))
        with np.load(source_npz, allow_pickle=False) as arrays:
            gt = arrays["octa_gt"]
            prediction = arrays["octa"]
            expected_gt = gt.mean(axis=0, dtype=np.float64).T.astype(np.float32)
            expected_prediction = prediction.mean(axis=0, dtype=np.float64).T.astype(np.float32)
            with np.load(OUTPUT / "projections" / f"{case}.npz", allow_pickle=False) as maps:
                assert maps["axis_order"].item() == "XY"
                assert np.array_equal(maps["gt_full"], expected_gt)
                assert np.array_equal(maps["pred_full"], expected_prediction)
                assert not np.allclose(maps["gt_full"], gt.max(axis=0).T)
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        expected = next(row for row in reference["records"] if row["case_id"] == case)
        for name in ("mae", "psnr", "ssim"):
            assert abs(report["records"][0][name] - expected[name]) < 1e-10, (name, report["records"][0], expected)
        archive = Path(str(OUTPUT) + ".zip")
        archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        assert Path(str(archive) + ".sha256").read_text().split()[0] == archive_hash
        with zipfile.ZipFile(archive) as package:
            assert package.testzip() is None
            assert f"figures/{case}.png" in package.namelist()
            assert f"projections/{case}.npz" in package.namelist()
            assert all(not name.endswith(".pt") for name in package.namelist())
        first_hash = archive_hash
        duplicate = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        assert duplicate.returncode != 0
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == first_hash
        assert hashlib.sha256(source_npz.read_bytes()).hexdigest() == source_hash
        verification = {
            "scope": "CPU exporter verification on one selected existing epoch-98 test case; not the current server epoch-193 result.",
            "case_id": case, "checkpoint_epoch": original["checkpoint_epoch"],
            "full_mean_arrays_match": True, "full_is_not_maximum_projection": True,
            "metrics_match_previous_verified_FULL_export": True,
            "original_volume_unchanged": True, "duplicate_output_preserved": True,
            "zip_integrity_and_checksum_passed": True,
            "source_npz_mib": source_npz.stat().st_size / 2**20,
            "review_zip_mib": archive.stat().st_size / 2**20,
            "metrics": report["metrics"],
        }
        (HERE / "full_export_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
        print(json.dumps(verification, indent=2))

if __name__ == "__main__":
    main()

