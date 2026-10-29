"""Diagnose detail fidelity without changing predictions or training settings."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIRS = ROOT / "analysis/flowoct_3m_best193_test_20261024/comparison_arrays"
INFO = Path(r"D:\data\FSdata\results\flowoct_3m_best_after200\prediction.json")


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    info = json.loads(INFO.read_text(encoding="utf-8"))
    sources = {}
    for name, expected in info["environment"]["source_sha256"].items():
        path = ROOT / "flowoct_reproduction" / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, name
        sources[name] = actual
    rows = []
    for path in sorted(PAIRS.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            gt = data["gt_full"].astype(np.float64)
            gt_range = float(np.ptp(gt))
            gt_high = gt - gaussian_filter(gt, 1)
            gt_low = gaussian_filter(gt, 3)
            row = {"case_id": path.stem, "gt_min_max": [float(gt.min()), float(gt.max())],
                   "gt_range": gt_range, "gt_p1_p99": np.percentile(gt, [1, 99]).tolist()}
            for epoch in (98, 193):
                pred = data[f"pred_full_{epoch}"].astype(np.float64)
                high = pred - gaussian_filter(pred, 1)
                mse = float(np.mean((pred - gt) ** 2))
                row[str(epoch)] = {
                    "full_ssim_range1": float(structural_similarity(gt, pred, data_range=1., win_size=7)),
                    "full_ssim_gt_observed_range": float(structural_similarity(gt, pred, data_range=gt_range, win_size=7)),
                    "full_psnr_range1": float(-10*np.log10(mse)),
                    "full_psnr_gt_observed_range": float(10*np.log10(gt_range**2/mse)),
                    "highpass_correlation": corr(gt_high, high),
                    "highpass_std_ratio_pred_over_gt": float(high.std()/gt_high.std()),
                    "lowpass_correlation": corr(gt_low, gaussian_filter(pred, 3)),
                    "gt_mean": float(gt.mean()), "prediction_mean": float(pred.mean()),
                }
        rows.append(row)
    assert len(rows) == 50
    summary = {
        str(epoch): {key: float(np.mean([row[str(epoch)][key] for row in rows]))
                     for key in rows[0][str(epoch)]}
        for epoch in (98, 193)
    }
    result = {
        "cases": len(rows), "checkpoint_epoch": info["checkpoint_epoch"],
        "checkpoint_sha256": info["checkpoint_sha256"], "sources_match_prediction": sources,
        "gt_range_mean": float(np.mean([r["gt_range"] for r in rows])),
        "gt_range_min_max": [min(r["gt_range"] for r in rows), max(r["gt_range"] for r in rows)],
        "summary": summary, "records": rows,
        "protocol": {
            "input": "Unchanged paired FULL arithmetic means, saved as raw float32 arrays",
            "diagnostic_dtype": "float64",
            "range_sensitivity": "Same maps and win_size=7; only SSIM data_range changes from 1 to each GT map max-min. No clipping or rescaling of predictions.",
            "range_sensitivity_limit": "Exploratory diagnostic, not a replacement for the prespecified scores or a paper-comparable metric",
            "highpass": "image minus Gaussian sigma=1 pixel; Pearson correlation with paired GT",
            "lowpass": "Gaussian sigma=3 pixels; Pearson correlation with paired GT",
            "frequency_limit": "Diagnostic texture agreement, not vessel segmentation, accuracy, or a causal test; Gaussian filters only used to compute diagnostics, never to alter reported predictions",
            "model_changed": False, "training_changed": False, "prediction_changed": False,
        },
    }
    (HERE / "projection_detail.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    case = "10305"  # Fixed reference used in the previous audit, not a new best/worst selection.
    with np.load(PAIRS / f"{case}.npz", allow_pickle=False) as data:
        maps = [data[key] for key in ("gt_full", "pred_full_98", "pred_full_193")]
    gt = maps[0]
    lo, hi = float(gt.min()), float(np.percentile(gt, 99))
    # Fixed 64x64 upper-right patch in displayed XY coordinates, all columns identical.
    x, y, side = 144, 40, 64
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.6), constrained_layout=True)
    for col, (array, title) in enumerate(zip(maps, ("真实 OCTA", "第98轮预测", "第193轮预测"))):
        axes[0, col].imshow(array, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axes[0, col].add_patch(Rectangle((x-.5, y-.5), side, side, fill=False, edgecolor="#ff9800", linewidth=1.5))
        axes[0, col].set_title(title, fontsize=14)
        axes[1, col].imshow(array[y:y+side, x:x+side], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axes[1, col].set_title("相同64×64像素局部，直接放大", fontsize=11)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle("10305 固定复核病例：大血管相似，细纹理仍不匹配\n"
                 "三列共用真实图灰度窗；最近邻显示；未锐化、降噪或分别增强对比度", fontsize=14)
    fig.savefig(HERE / "10305_原始像素局部对照.png", dpi=160, facecolor="white")
    plt.close(fig)
    print(json.dumps({"cases": len(rows), "gt_range_mean": result["gt_range_mean"],
                      "summary": summary, "figure": str(HERE / "10305_原始像素局部对照.png")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
