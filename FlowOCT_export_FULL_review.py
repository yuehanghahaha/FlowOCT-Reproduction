"""Export a CPU-only FULL mean-projection review package from saved predictions."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


def scores(prediction, target):
    difference = prediction.astype(np.float64) - target
    mse = float(np.mean(difference ** 2))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "psnr": float(-10 * np.log10(max(mse, np.finfo(np.float64).eps))),
        "ssim": float(structural_similarity(target, prediction, data_range=1.0, win_size=7)),
    }


def aggregate(rows):
    return {
        "cases": len(rows),
        "metrics": {
            name: {"mean": float(np.mean([row[name] for row in rows])),
                   "std": float(np.std([row[name] for row in rows], ddof=0))}
            for name in ("mae", "psnr", "ssim")
        },
    }


def export(predictions, evaluation, output):
    predictions, evaluation, output = [Path(p).resolve() for p in (predictions, evaluation, output)]
    archive = Path(str(output) + ".zip")
    partial = Path(str(archive) + ".partial")
    for destination in (output, archive, partial):
        if destination.exists():
            raise FileExistsError(f"Choose a new output name; existing file or directory: {destination}")
    info_path = predictions / "prediction.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["split"] != "test" or info.get("limited_diagnostic"):
        raise ValueError("Use a complete test-set prediction, not a limited diagnostic")
    records = info["records"]
    expected = {str(record["case_id"]) for record in records}
    actual = {path.stem for path in predictions.glob("*.npz")}
    if not records or len(expected) != len(records) or expected != actual:
        raise ValueError("Prediction case files do not match prediction.json")
    summary_path = evaluation / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["cases"] != len(records):
        raise ValueError("Evaluation case count differs from the prediction case count")
    if summary["protocol"]["checkpoint_sha256"] != info["checkpoint_sha256"]:
        raise ValueError("Evaluation was generated from a different checkpoint")
    required_metadata = (info_path, summary_path, evaluation / "per_case.csv")
    for path in required_metadata:
        if not path.is_file():
            raise FileNotFoundError(path)

    (output / "figures").mkdir(parents=True)
    (output / "projections").mkdir()
    results = []
    for index, record in enumerate(records, 1):
        case = str(record["case_id"])
        geometry = record["geometry"]
        if geometry.get("model_axis_order") != "YXZ":
            raise ValueError(f"Unrecognized model axis order: {case}")
        if geometry["flipped_z"] or geometry["crop_start_yxz"][2] != 0:
            raise ValueError(f"FULL requires an unflipped complete prepared axial depth: {case}")
        if geometry["resized_shape"][2] != geometry["crop_size"]:
            raise ValueError(f"Prepared axial depth was cropped: {case}")
        with np.load(predictions / f"{case}.npz", allow_pickle=False) as data:
            if data["axis_order"].item() != "ZYX":
                raise ValueError(f"Expected saved ZYX axes: {case}")
            target, prediction = data["octa_gt"], data["octa"]
        if target.shape != prediction.shape or target.ndim != 3 or min(target.shape) < 7:
            raise ValueError(f"Invalid volume shapes for {case}")
        expected_size = geometry["crop_size"]
        if target.shape != (expected_size,) * 3:
            raise ValueError(f"Volume shape differs from recorded crop geometry: {case}")
        for array in (target, prediction):
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError(f"Expected finite normalized [0,1] arrays: {case}")
        # Saved ZYX axis 0 is physical depth. Transpose YX maps to official XY display.
        gt_full = target.mean(axis=0, dtype=np.float64).T.astype(np.float32)
        pred_full = prediction.mean(axis=0, dtype=np.float64).T.astype(np.float32)
        # One central B-scan is a visual example, not a complete 3D audit.
        gt_slice = target[:, target.shape[1] // 2, :].copy()
        pred_slice = prediction[:, prediction.shape[1] // 2, :].copy()
        del target, prediction
        values = scores(pred_full, gt_full)
        lo = float(gt_full.min())
        hi = float(np.percentile(gt_full, 99))
        if hi <= lo:
            hi = lo + 1e-6
        np.savez_compressed(output / "projections" / f"{case}.npz",
                            gt_full=gt_full, pred_full=pred_full,
                            axis_order=np.asarray("XY"), case_id=np.asarray(case))
        row = {
            "case_id": case, "patient_id": record["patient_id"],
            "disease": record["disease"], **values, "geometry": geometry,
            "common_full_display_window": [lo, hi],
        }
        results.append(row)
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
        panels = (
            (gt_full, "Real OCTA: FULL mean", "gray", lo, hi),
            (pred_full, "Predicted OCTA: FULL mean", "gray", lo, hi),
            (np.abs(pred_full - gt_full), "FULL absolute error", "magma", 0, hi - lo),
            (gt_slice, "Real OCTA: central B-scan (ZX)", "gray", 0, 1),
            (pred_slice, "Predicted OCTA: central B-scan (ZX)", "gray", 0, 1),
            (np.abs(pred_slice - gt_slice), "B-scan absolute error", "magma", 0, 1),
        )
        for ax, (array, title, cmap, low, high) in zip(axes.flat, panels):
            artist = ax.imshow(array, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            if cmap == "magma":
                fig.colorbar(artist, ax=ax, fraction=.045, pad=.02)
        fig.suptitle(
            f"{case} | {record['disease']} | checkpoint epoch {info['checkpoint_epoch']}\n"
            f"Prepared center crop; FULL over all saved Z samples. "
            f"PSNR={values['psnr']:.3f} dB, SSIM={values['ssim']:.4f}\n"
            "Real/pred FULL share the GT min-to-P99 display window; metrics use raw [0,1] values.",
            fontsize=10,
        )
        fig.savefig(output / "figures" / f"{case}.png", dpi=120, facecolor="white")
        plt.close(fig)
        print(f"FULL {index}/{len(records)}: {case} PSNR={values['psnr']:.4f} SSIM={values['ssim']:.4f}", flush=True)

    report = {
        "checkpoint_epoch": info["checkpoint_epoch"],
        "checkpoint_sha256": info["checkpoint_sha256"],
        "split": info["split"], **aggregate(results), "records": results,
        "by_disease": {
            disease: aggregate([row for row in results if row["disease"] == disease])
            for disease in sorted({row["disease"] for row in results})
        },
        "protocol": {
            "projection": "FULL arithmetic mean over saved physical Z, ZYX axis=0",
            "orientation": "XY, transposed from YX to match the supplied official BMP orientation",
            "target": "Mean of the paired preprocessed GT volume, not the official display BMP",
            "coverage": info["coverage"],
            "data_range": 1.0, "ssim_win_size": 7,
            "display": "GT and prediction share GT FULL min-to-P99 window; display scaling is excluded from metrics",
            "array_storage": "projections/*.npz contain raw float32 gt_full and pred_full before display scaling",
            "b_scan": "one central Y plane for a visual example; other 3D voxels are not included",
            "full_volume_metrics": "Copied from existing evaluation; not recomputed by this projection exporter",
            "model_changed": False, "used_to_select_checkpoint": False,
        },
    }
    (output / "full_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for path in required_metadata:
        shutil.copy2(path, output / path.name)
    (output / "README.txt").write_text(
        "FULL projection review package\n"
        "figures/: paired FULL mean projections, absolute errors, and one central B-scan per case.\n"
        "projections/: raw float32 2D FULL arrays for independent metric checks.\n"
        "full_metrics.json: per-case, aggregate and disease-group FULL metrics.\n"
        "prediction.json, summary.json, per_case.csv: unchanged original test records.\n"
        "The original 3D NPZ files and checkpoints remain on the server.\n"
        "This package supports projection review and selected slice visualization, not a full 3D re-audit.\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output))
    partial.replace(archive)
    # Stream the archive hash so the exporter's memory use stays bounded.
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    checksum = Path(str(archive) + ".sha256")
    checksum.write_text(f"{digest.hexdigest()}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"cases": len(results), "checkpoint_epoch": info["checkpoint_epoch"],
                      "archive": str(archive), "archive_mib": round(archive.stat().st_size / 2**20, 2),
                      "full_metrics": report["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.predictions, args.evaluation, args.output)


if __name__ == "__main__":
    main()

