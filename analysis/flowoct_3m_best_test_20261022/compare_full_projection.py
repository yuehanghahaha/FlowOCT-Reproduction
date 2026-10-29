"""OCTA-500 FULL mean projection of existing predictions; no model changes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torchio as tio


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True, help="Existing OCTA500_common_v1 root")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("Choose a new output directory; existing results are preserved")
info = json.loads((args.predictions / "prediction.json").read_text(encoding="utf-8"))
assert info["split"] == "test" and not info["limited_diagnostic"]
assert info["config"]["fov"] == "3M"
args.output.mkdir(parents=True)
torch.set_num_threads(info["config"]["threads"])
font = Path(r"C:\Windows\Fonts\msyh.ttc")
if font.is_file():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})


def aligned_official(image_xy, geometry):
    """Match the actual TorchIO resize and recorded YXZ center crop."""
    sy, sx, sz = geometry["crop_start_yxz"]
    ry, rx, rz = geometry["resized_shape"]
    size = geometry["crop_size"]
    if sz != 0 or rz != size or geometry["flipped_z"]:
        raise ValueError("FULL mean requires all axial samples and an unflipped test volume")
    resized_yx = np.asarray(tio.Resize((ry, rx, 1))(
        image_xy.T.copy()[None, ..., None]))[0, ..., 0]
    return resized_yx[sy:sy + size, sx:sx + size].T


def metrics(prediction, target):
    difference = prediction.astype(np.float64) - target
    return {"mae": float(np.abs(difference).mean()),
            "psnr": float(-10 * np.log10(max(float((difference ** 2).mean()), np.finfo(np.float64).eps))),
            "ssim": float(structural_similarity(target, prediction, data_range=1, win_size=7))}


records = []
for index, record in enumerate(info["records"], 1):
    case = record["case_id"]
    with np.load(args.predictions / f"{case}.npz", allow_pickle=False) as arrays:
        assert str(arrays["axis_order"]) == "ZYX"
        target, prediction = arrays["octa_gt"], arrays["octa"]
    assert target.shape == prediction.shape == (256, 256, 256)
    assert np.isfinite(target).all() and np.isfinite(prediction).all()
    # Dataset section 3.3: FULL is the axial mean, not full-depth maximum.
    # Transpose the en-face maps from YX to official supplied BMP orientation XY.
    gt_full = target.mean(axis=0, dtype=np.float64).T.astype(np.float32)
    pred_full = prediction.mean(axis=0, dtype=np.float64).T.astype(np.float32)
    gt_max, pred_max = target.max(axis=0).T, prediction.max(axis=0).T
    official_path = args.data / "official_projections/3M/OCTA(FULL)" / f"{case}.bmp"
    with Image.open(official_path) as image:
        official = np.asarray(image.convert("L"), dtype=np.float32) / 255
    crop = aligned_official(official, record["geometry"])
    assert crop.shape == gt_full.shape
    lo, hi = float(gt_full.min()), float(np.percentile(gt_full, 99))
    if hi <= lo:
        raise ValueError(f"Constant FULL projection for {case}")
    scores = metrics(pred_full, gt_full)
    records.append({"case_id": case, "disease": record["disease"], **scores,
                    "common_display_window": [lo, hi],
                    "official_source": str(official_path.resolve()),
                    "geometry": record["geometry"]})

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.5), constrained_layout=True)
    panels = [
        (official, "官方 FULL：完整视野", (0, 1)),
        (crop, "官方 FULL：对齐模型视野", (0, 1)),
        (gt_full, "真实体积：FULL 均值投影", (lo, hi)),
        (pred_full, "生成体积：FULL 均值投影", (lo, hi)),
        (gt_max, "旧图真实：全深度最大值", (0, 1)),
        (pred_max, "旧图生成：全深度最大值", (0, 1)),
        (gt_full, "真实 FULL：固定[0,1]显示", (0, 1)),
        (pred_full, "生成 FULL：固定[0,1]显示", (0, 1)),
    ]
    for ax, (array, title, window) in zip(axes.flat, panels):
        ax.imshow(array, cmap="gray", vmin=window[0], vmax=window[1], interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    sy, sx, _ = record["geometry"]["crop_start_yxz"]
    ry, rx, _ = record["geometry"]["resized_shape"]
    size = record["geometry"]["crop_size"]
    height, width = official.shape  # official image rows=X, columns=Y
    axes[0, 0].add_patch(Rectangle((sy * width / ry - .5, sx * height / rx - .5),
        size * width / ry, size * height / rx, linewidth=1.4, edgecolor="#29a9dd", facecolor="none"))
    fig.suptitle(f"{case} {record['disease']} | 第{info['checkpoint_epoch']}轮 | OCTA-500 FULL 对照\n"
                 "蓝框为模型中心视野；上排真实/生成共用真实FULL的min至P99显示窗；显示拉伸不参与指标",
                 fontsize=12)
    fig.savefig(args.output / f"{case}.png", dpi=150)
    plt.close(fig)
    print(f"FULL {index}/{len(info['records'])}: {case} | PSNR={scores['psnr']:.3f} SSIM={scores['ssim']:.4f}", flush=True)
    del target, prediction

aggregate = {key: {"mean": float(np.mean([r[key] for r in records])),
                   "std": float(np.std([r[key] for r in records], ddof=0))}
             for key in ("mae", "psnr", "ssim")}
result = {"checkpoint_epoch": info["checkpoint_epoch"], "checkpoint_sha256": info["checkpoint_sha256"],
          "split": info["split"], "cases": len(records), "metrics": aggregate, "records": records,
          "protocol": {"source": "https://arxiv.org/pdf/2012.07261 section 3.3 (B4)",
              "projection": "arithmetic mean along all axial samples, ZYX axis=0",
              "orientation": "XY, matching official BMP; transpose of old YX display",
              "coverage": "256 center crop after lateral resize to 400; 64% lateral FOV per axis",
              "metrics": "prediction vs GT on identical crop, raw normalized [0,1]; SSIM win_size=7",
              "display": "common GT min-to-P99 window for GT/prediction; official BMP kept in its supplied display scale",
              "official_crop": "official BMP linearly resized and center-cropped only for visual alignment; not a metric target",
              "model_changed": False, "used_to_select_checkpoint": False}}
(args.output / "full_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"cases": len(records), "metrics": aggregate}, ensure_ascii=False), flush=True)
