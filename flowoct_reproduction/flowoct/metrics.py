"""Explicit [0,1] volume and axial max-projection metrics."""

import numpy as np
from skimage.metrics import structural_similarity


def image_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or min(target.shape) < 7:
        raise ValueError("Metrics require matching arrays with every axis >=7")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite metric input")
    if min(prediction.min(), target.min()) < 0 or max(prediction.max(), target.max()) > 1:
        raise ValueError("Metrics require normalized [0,1] arrays")
    difference = prediction.astype(np.float64) - target
    mse = float(np.mean(difference ** 2))
    # Cap exact-match PSNR at the float64 epsilon floor so JSON remains finite.
    return {"mae": float(np.mean(np.abs(difference))),
            "psnr": float(-10 * np.log10(max(mse, np.finfo(np.float64).eps))),
            "ssim": float(structural_similarity(target, prediction, data_range=1.0, win_size=7))}


def volume_metrics(prediction_zyx, target_zyx):
    result = {"volume_" + k: v for k, v in image_metrics(prediction_zyx, target_zyx).items()}
    result.update({"axial_mip_" + k: v for k, v in image_metrics(
        prediction_zyx.max(axis=0), target_zyx.max(axis=0)).items()})
    return result


def summarize(rows):
    keys = [key for key in rows[0] if key.startswith(("volume_", "axial_mip_"))]
    return {"cases": len(rows), "aggregation": "equal weight per scan; population standard deviation",
            "metrics": {key: {"mean": float(np.mean([row[key] for row in rows])),
                              "std": float(np.std([row[key] for row in rows]))} for key in keys}}
