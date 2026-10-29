"""Read the downloaded test exports; write analysis beside this script only."""

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-only", action="store_true", help="Keep existing overview/review images")
args = parser.parse_args()
SOURCE = Path(r"D:\data\FSdata")
OUTPUT = Path(__file__).resolve().parent
EVALUATION = SOURCE / "evaluations/flowoct_3m_best_preview"
FIGURES = SOURCE / "figures/flowoct_3m_best_preview"
SUMMARY = json.loads((EVALUATION / "summary.json").read_text(encoding="utf-8"))
with (EVALUATION / "per_case.csv").open(encoding="utf-8-sig", newline="") as stream:
    ROWS = list(csv.DictReader(stream))
KEYS = list(SUMMARY["metrics"])
for row in ROWS:
    for key in KEYS:
        row[key] = float(row[key])
BY_ID = {row["case_id"]: row for row in ROWS}

font = Path(r"C:\Windows\Fonts\msyh.ttc")
if font.is_file():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
COLORS = {"NORMAL": "#326b9b", "DR": "#c47b32", "AMD": "#bf4b57", "CNV": "#8160aa"}
DISEASES = list(COLORS)


def image_panel(case_id, column):
    # Exact data-panel pixels in the existing 1800 x 1350 exports. No intensity
    # transformation or contrast stretching; this is a visual review of PNGs.
    with Image.open(FIGURES / f"{case_id}.png") as image:
        if image.size != (1800, 1350):
            raise ValueError(f"Unexpected PNG layout for {case_id}: {image.size}")
        x = {"target": 488, "prediction": 938, "error": 1388}[column]
        return np.asarray(image.convert("RGB"))[970:1344, x:x + 374].copy()


audit = {
    "cases": len(ROWS), "unique_cases": len(BY_ID),
    "unique_patient_ids": len({row["patient_id"] for row in ROWS}),
    "image_count": len(list(FIGURES.glob("*.png"))),
    "missing_images": sorted(set(BY_ID) - {path.stem for path in FIGURES.glob("*.png")}),
    "all_finite": bool(np.isfinite([[row[k] for k in KEYS] for row in ROWS]).all()),
    "summary_max_abs_mean_difference": max(abs(np.mean([r[k] for r in ROWS]) -
                                                 SUMMARY["metrics"][k]["mean"]) for k in KEYS),
    "summary_max_abs_std_difference": max(abs(np.std([r[k] for r in ROWS], ddof=0) -
                                                SUMMARY["metrics"][k]["std"]) for k in KEYS),
    "disease_counts": dict(Counter(row["disease"] for row in ROWS)),
    "protocol": SUMMARY["protocol"],
}
assert audit["cases"] == audit["unique_cases"] == audit["image_count"] == 50
assert not audit["missing_images"] and audit["all_finite"]
assert audit["summary_max_abs_mean_difference"] < 1e-12
assert audit["summary_max_abs_std_difference"] < 1e-12
for disease, group in SUMMARY["by_disease"].items():
    rows = [r for r in ROWS if r["disease"] == disease]
    assert len(rows) == group["cases"]
    for key in KEYS:
        assert abs(np.mean([r[key] for r in rows]) - group["metrics"][key]["mean"]) < 1e-12
        assert abs(np.std([r[key] for r in rows]) - group["metrics"][key]["std"]) < 1e-12
audit["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (EVALUATION / "summary.json", EVALUATION / "per_case.csv")}
stats = {}
for key in KEYS:
    values = np.asarray([row[key] for row in ROWS])
    ordered = sorted(ROWS, key=lambda r: r[key], reverse=not key.endswith("mae"))
    stats[key] = {"mean": float(values.mean()), "std_population": float(values.std()),
                  "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
                  "q75": float(np.quantile(values, .75)), "min": float(values.min()),
                  "max": float(values.max()), "best_five": [r["case_id"] for r in ordered[:5]],
                  "worst_five": [r["case_id"] for r in reversed(ordered[-5:])]}
predictions = {}
for label, folder in (("epoch98", "flowoct_3m_best_preview"),
                      ("epoch100", "flowoct_3m_epoch0100_test")):
    path = SOURCE / "results" / folder / "prediction.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    records = {row["case_id"]: row for row in info["records"]}
    assert set(records) == set(BY_ID)
    assert info["split"] == "test" and not info["limited_diagnostic"]
    paths = list(path.parent.glob("*.npz"))
    assert {p.stem for p in paths} == set(BY_ID)
    assert all(zipfile.is_zipfile(p) for p in paths)
    for key in KEYS:
        assert abs(np.mean([r[key] for r in records.values()]) - info["summary"]["metrics"][key]["mean"]) < 1e-12
        assert abs(np.std([r[key] for r in records.values()]) - info["summary"]["metrics"][key]["std"]) < 1e-12
    predictions[label] = {"path": path, "info": info, "records": records}
best, hundred = predictions["epoch98"], predictions["epoch100"]
assert best["info"]["checkpoint_epoch"] == 98 and hundred["info"]["checkpoint_epoch"] == 100
assert best["info"]["checkpoint_sha256"] == SUMMARY["protocol"]["checkpoint_sha256"]
assert best["info"]["manifest_sha256"] == hundred["info"]["manifest_sha256"]
assert best["info"]["config"] == hundred["info"]["config"]
assert all(best["records"][cid]["geometry"] == hundred["records"][cid]["geometry"] for cid in BY_ID)
assert all(best["records"][cid][k] == BY_ID[cid][k] for cid in BY_ID for k in KEYS)
audit["confirmed_checkpoint_epoch"] = 98
audit["paired_epoch100_same_manifest_config_and_geometry"] = True
audit["valid_npz_zip_containers"] = {label: 50 for label in predictions}
audit["prediction_metadata"] = {label: {k: p["info"][k] for k in
    ("checkpoint", "checkpoint_epoch", "checkpoint_sha256", "manifest_sha256")}
    for label, p in predictions.items()}
audit["source_sha256"].update({str(p["path"]): hashlib.sha256(p["path"].read_bytes()).hexdigest()
                               for p in predictions.values()})
comparison = {}
for key in KEYS:
    differences = np.asarray([best["records"][cid][key] - hundred["records"][cid][key] for cid in BY_ID])
    wins = differences < 0 if key.endswith("mae") else differences > 0
    comparison[key] = {"epoch98": SUMMARY["metrics"][key]["mean"],
                        "epoch100": hundred["info"]["summary"]["metrics"][key]["mean"],
                        "difference98_minus100": float(differences.mean()),
                        "cases_better_epoch98": int(wins.sum())}

# Recalculate MAE/PSNR for four representative cases, using both original NPZ
# sets. Exact-zero GT is a numerical partition, not an anatomical ROI metric.
raw_checks = []
for case_id in ("10305", "10454", "10325", "10484"):
    checks = {"case_id": case_id}
    gt_hashes = []
    for label, p in predictions.items():
        with np.load(p["path"].parent / f"{case_id}.npz", allow_pickle=False) as arrays:
            gt, prediction = arrays["octa_gt"], arrays["octa"]
            assert gt.shape == prediction.shape == (256, 256, 256)
            assert str(arrays["axis_order"]) == "ZYX"
            assert np.isfinite(gt).all() and np.isfinite(prediction).all()
            assert min(gt.min(), prediction.min()) >= 0 and max(gt.max(), prediction.max()) <= 1
            gt_hashes.append(hashlib.sha256(gt.tobytes()).hexdigest())
            difference = prediction.astype(np.float64) - gt
            projection_difference = prediction.max(axis=0).astype(np.float64) - gt.max(axis=0)
            recalculated = {"volume_mae": float(np.abs(difference).mean()),
                "volume_psnr": float(-10 * np.log10(np.mean(difference ** 2))),
                "axial_mip_mae": float(np.abs(projection_difference).mean()),
                "axial_mip_psnr": float(-10 * np.log10(np.mean(projection_difference ** 2)))}
            assert all(abs(value - p["records"][case_id][key]) < 1e-10 for key, value in recalculated.items())
            zero = gt == 0
            checks[label] = {"recalculated": recalculated,
                "gt_exact_zero_fraction": float(zero.mean()),
                "prediction_mean_at_exact_zero_gt": float(prediction[zero].mean(dtype=np.float64)),
                "mae_at_nonzero_gt": float(np.abs(difference[~zero]).mean())}
            del gt, prediction, difference, projection_difference, zero
    checks["identical_gt_between_models"] = gt_hashes[0] == gt_hashes[1]
    assert checks["identical_gt_between_models"]
    raw_checks.append(checks)
(OUTPUT / "analysis.json").write_text(json.dumps({"audit": audit, "statistics": stats,
    "epoch98_vs_epoch100": comparison, "raw_volume_spot_checks": raw_checks},
    ensure_ascii=False, indent=2), encoding="utf-8")

fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for ax, key, title, ylabel in [
    (axes[0, 0], "volume_psnr", "体积 PSNR", "PSNR (dB)"),
    (axes[0, 1], "volume_ssim", "体积 SSIM", "SSIM"),
    (axes[1, 0], "axial_mip_psnr", "最大投影 PSNR", "PSNR (dB)"),
    (axes[1, 1], "axial_mip_ssim", "最大投影 SSIM", "SSIM"),
]:
    for case_id in BY_ID:
        ax.plot([0, 1], [best["records"][case_id][key], hundred["records"][case_id][key]],
                color="#a7b5c4", alpha=.35, linewidth=.8)
    means = [comparison[key]["epoch98"], comparison[key]["epoch100"]]
    ax.plot([0, 1], means, color="#244c77", marker="o", linewidth=2.5, markersize=7)
    for i, value in enumerate(means):
        ax.annotate(f"均值 {value:.4f}", (i, value), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color="#244c77")
    ax.set_xticks([0, 1], ["第98轮", "第100轮"])
    ax.set(title=title, ylabel=ylabel, xlim=(-.35, 1.35))
    ax.grid(axis="y", alpha=.18)
fig.suptitle("同50例测试病例：第98轮与第100轮\n浅线为单病例，深线为均值；第98轮事先由验证集选出", fontsize=13)
fig.savefig(OUTPUT / "epoch98_vs_epoch100.png", dpi=160)
plt.close(fig)
if args.audit_only:
    print(json.dumps({"audit": audit, "comparison": comparison, "raw_spot_checks": raw_checks}, ensure_ascii=False))
    raise SystemExit(0)

# Each point is one scan. Error bars in the group panels are population SD,
# describing case variation; they are not confidence intervals or model seeds.
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for ax, xkey, ykey, title, xlabel, ylabel in [
    (axes[0, 0], "volume_psnr", "axial_mip_psnr", "体积与投影 PSNR", "体积 PSNR (dB)", "投影 PSNR (dB)"),
    (axes[0, 1], "volume_ssim", "axial_mip_ssim", "体积与投影 SSIM", "体积 SSIM", "投影 SSIM"),
]:
    for disease in DISEASES:
        rows = [r for r in ROWS if r["disease"] == disease]
        ax.scatter([r[xkey] for r in rows], [r[ykey] for r in rows],
                   color=COLORS[disease], label=f"{disease} (n={len(rows)})", s=35, alpha=.85)
    for case_id in ("10325", "10454", "10448", "10484", "10404"):
        row = BY_ID[case_id]
        ax.annotate(case_id, (row[xkey], row[ykey]), xytext=(4, 5),
                    textcoords="offset points", fontsize=8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.18)
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
for ax, key, title, ylabel in [
    (axes[1, 0], "axial_mip_psnr", "各组投影 PSNR：均值与病例标准差", "投影 PSNR (dB)"),
    (axes[1, 1], "axial_mip_ssim", "各组投影 SSIM：均值与病例标准差", "投影 SSIM"),
]:
    for i, disease in enumerate(DISEASES):
        values = np.asarray([r[key] for r in ROWS if r["disease"] == disease])
        offsets = np.linspace(-.12, .12, len(values)) if len(values) > 1 else np.array([0])
        ax.scatter(i + offsets, values, color=COLORS[disease], alpha=.35, s=18)
        ax.errorbar(i, values.mean(), yerr=values.std() if len(values) > 1 else None,
                    fmt="o", color=COLORS[disease], capsize=5, markersize=7, linewidth=1.7)
    ax.set_xticks(range(len(DISEASES)), [f"{d}\nn={audit['disease_counts'][d]}" for d in DISEASES])
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=.18)
fig.suptitle("FlowOCT 3M：50例测试结果\n病例等权；AMD、CNV各1例，不能据此推断疾病总体表现", fontsize=14)
fig.savefig(OUTPUT / "metrics_overview.png", dpi=160)
plt.close(fig)

# Systematic review: inspect every downloaded PNG, in case-id order.
review = OUTPUT / "review"
review.mkdir(exist_ok=True)
ordered = sorted(ROWS, key=lambda r: r["case_id"])
for page, start in enumerate(range(0, len(ordered), 10), 1):
    fig, axes = plt.subplots(5, 4, figsize=(12, 15), constrained_layout=True)
    for i, row in enumerate(ordered[start:start + 10]):
        for j, column in enumerate(("target", "prediction")):
            ax = axes[i // 2, 2 * (i % 2) + j]
            ax.imshow(image_panel(row["case_id"], column))
            label = "真实 OCTA" if j == 0 else "生成 OCTA"
            ax.set_title(f"{row['case_id']} {row['disease']} | {label}\n投影 SSIM={row['axial_mip_ssim']:.3f}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"最大投影逐病例复核 {page}/5（原PNG显示范围，无对比度调整）", fontsize=13)
    fig.savefig(review / f"contact_{page:02d}.png", dpi=130)
    plt.close(fig)

cases = [("10484", "投影 SSIM 最高"), ("10305", "正常组示例"),
         ("10454", "投影 SSIM 最低"), ("10325", "体积 PSNR 最低")]
fig, axes = plt.subplots(len(cases), 3, figsize=(10, 13), constrained_layout=True)
for i, (case_id, reason) in enumerate(cases):
    row = BY_ID[case_id]
    for j, column in enumerate(("target", "prediction", "error")):
        ax = axes[i, j]
        ax.imshow(image_panel(case_id, column))
        ax.axis("off")
        name = {"target": "真实 OCTA", "prediction": "生成 OCTA", "error": "绝对误差"}[column]
        ax.set_title(f"{case_id} {row['disease']} | {name}\n{reason}；投影 SSIM={row['axial_mip_ssim']:.3f}", fontsize=9)
fig.suptitle("代表病例的最大投影对照\n取自原PNG，未调整亮度或对比度；误差图为原图配色", fontsize=13)
fig.savefig(OUTPUT / "visual_examples.png", dpi=150)
plt.close(fig)
print(json.dumps({"output": str(OUTPUT), "audit": audit, "statistics": stats}, ensure_ascii=False))
