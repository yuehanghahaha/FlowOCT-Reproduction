"""Fixed case 10308: a separate from-scratch fitting diagnostic.

Uses the installed reproduction core without changing its model or losses.
Only this tool's output directory receives new checkpoints and reports.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = "/workspace/flowoct_work/flowoct_reproduction_a100"
STAGES = (70, 100, 200, 300)
CASE = "10308"
FORMAT = "flowoct-singlecase-v1"
LOG = logging.getLogger("singlecase")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def model_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stable_checkpoint(source, destination=None):
    import torch
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {source}. Logs cannot recreate weights.")
    for attempt in range(5):
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if sha_file(source) == digest:
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            break
        if attempt == 4:
            raise RuntimeError(f"Checkpoint keeps changing: {source}")
        time.sleep(.5)
    if destination is not None:
        with Path(destination).open("xb") as stream:
            stream.write(raw)
        if sha_file(destination) != digest:
            raise OSError("Frozen checkpoint checksum mismatch")
    return payload, digest


def atomic_checkpoint(path, payload):
    import torch
    temporary = Path(str(path) + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_maps(path, maps):
    temporary = Path(str(path) + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **maps)
    os.replace(temporary, path)


def preflight(project):
    import importlib.metadata
    import matplotlib
    from skimage.metrics import structural_similarity
    from flowoct.data import read_manifest
    from flowoct.runner import load_config
    expected = read_json(HERE / "expected_baseline.json")
    base = project / "runs/flowoct_3m_200e"
    cfg = load_config(base / "config.json")
    for key, value in expected["config"].items():
        if cfg[key] != value:
            raise ValueError(f"Formal baseline config changed: {key}")
    if sha_file(cfg["manifest"]) != expected["manifest_sha256"]:
        raise ValueError("Manifest differs from the recorded 140/10/50 split")
    current = {p.relative_to(project).as_posix(): sha_file(p)
               for p in sorted((project / "flowoct").glob("*.py"))}
    if current != expected["source_sha256"]:
        raise ValueError("Reproduction core differs from the version used in the existing experiments")
    for package, wanted in expected["packages"].items():
        actual = importlib.metadata.version(package)
        if actual != wanted:
            raise ValueError(f"Environment mismatch: {package} expected {wanted}, got {actual}")
    recorded = read_json(base / "environment.json")
    sources = {k.replace("\\", "/"): v for k, v in recorded["source_sha256"].items()}
    if sources != current or recorded["packages"] != expected["packages"]:
        raise ValueError(f"Recorded training environment/core differs: {base}")
    rows = read_manifest(cfg["manifest"], "3M")
    selected = [r for r in rows if r["case_id"] == CASE]
    if len(selected) != 1 or selected[0]["split"] != "train" or selected[0]["disease"] != "NORMAL":
        raise ValueError("Fixed case10308 must be the original NORMAL training case")
    return cfg, selected, expected


def load_fixed_item(rows, cfg):
    from flowoct.data import PairedVolumeDataset
    # Preserve original membership; only this private loader's transforms change.
    dataset = PairedVolumeDataset(rows, "train", cfg)
    dataset.split = "validation"
    item = dataset[0]
    expected_start = [(cfg["load_size"] - cfg["crop_size"]) // 2] * 2 + [0]
    if item["geometry"]["flipped_z"] or item["geometry"]["crop_start_yxz"] != expected_start:
        raise ValueError("Fixed center crop/no flip check failed")
    return item


def item_proof(item):
    return {"case_id": item["case_id"], "source_split": "train", "geometry": item["geometry"],
            "oct_sha256": array_hash(item["A"].numpy()), "octa_sha256": array_hash(item["B"].numpy())}


@contextmanager
def evaluation_mode(model):
    from flowoct.runner import rng_state, restore_rng
    state, training = rng_state(), model.training
    model.eval()
    try:
        yield
    finally:
        restore_rng(state)
        model.train(training)


def correlation(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denominator) if denominator > 0 else None


def scores(gt, prediction):
    from scipy.ndimage import gaussian_filter
    from skimage.metrics import structural_similarity
    mse = max(float(np.mean((prediction.astype(np.float64) - gt) ** 2)), np.finfo(np.float64).eps)
    span = float(np.ptp(gt))
    return {"mae": float(np.mean(np.abs(prediction.astype(np.float64) - gt))),
            "psnr_range1": float(-10 * np.log10(mse)),
            "ssim_range1": float(structural_similarity(gt, prediction, data_range=1., win_size=7)),
            "ssim_gt_span": float(structural_similarity(gt, prediction, data_range=span, win_size=7)) if span > 0 else None,
            "highpass_correlation": correlation(gt - gaussian_filter(gt, 1), prediction - gaussian_filter(prediction, 1)),
            "lowpass_correlation": correlation(gaussian_filter(gt, 3), gaussian_filter(prediction, 3))}


def views(volume_yxz):
    middle = volume_yxz.shape[0] // 2
    # Saved en-face arrays have XY order, matching the previous review package.
    return {"full": volume_yxz.mean(axis=2, dtype=np.float64).T.astype(np.float32),
            "mip": volume_yxz.max(axis=2).T.copy(),
            "slice_z": volume_yxz[:, :, middle].T.copy(),
            "slice_y": volume_yxz[middle, :, :].T.copy(),
            "slice_x": volume_yxz[:, middle, :].T.copy()}


def evaluate(model, item, cfg):
    import torch
    from flowoct.metrics import volume_metrics
    with evaluation_mode(model), torch.no_grad():
        # GT is deliberately not supplied to the complete generator.
        prediction = model.validation(item["A"].unsqueeze(0), None, cfg["inference_chunk"])
        prediction = prediction.detach().float().cpu().numpy()[0, 0].copy()
    gt = item["B"].numpy()[0]
    if prediction.shape != gt.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid complete-volume OCT-only prediction")
    metrics = volume_metrics(prediction.transpose(2, 0, 1).copy(), gt.transpose(2, 0, 1).copy())
    maps = {"gt_" + k: v for k, v in views(gt).items()}
    maps.update({"pred_" + k: v for k, v in views(prediction).items()})
    for kind in ("full", "mip"):
        metrics.update({kind + "_" + k: v for k, v in scores(maps["gt_" + kind], maps["pred_" + kind]).items()})
    return metrics, maps


def teacher_probe(model, item):
    """GT-mixed training-path diagnostic, explicitly distinct from generation."""
    import torch
    count = item["A"].shape[1]
    indices = list(range(0, count, max(1, count // 8)))[:8]
    middle = count // 2
    if middle not in indices:
        indices[-1] = middle
    a = item["A"][:, indices].permute(1, 0, 2, 3).contiguous().to(model.device)
    b = item["B"][:, indices].permute(1, 0, 2, 3).contiguous().to(model.device)
    records, maps = [], {}
    with evaluation_mode(model), torch.no_grad():
        for time_value in (0., .5, .9):
            t = torch.full((len(indices),), time_value, device=model.device)
            xt = (1 - time_value) * a + time_value * b
            w = model.netG(torch.cat(model.dwt_2d(xt), dim=1), t)
            velocity = model.idwt_2d(*(w[:, c:c + 1] for c in range(4)))
            endpoint = torch.clamp(xt + (1 - time_value) * velocity, 0, 1)
            records.append({"t": time_value, "sampled_y_indices": indices,
                            "raw_velocity_mse": float(torch.mean((velocity - (b - a)) ** 2)),
                            "cropped_velocity_mse": float(model._cropped_l2_loss(velocity, b - a)),
                            "endpoint_mae": float(torch.mean(torch.abs(endpoint - b))),
                            "cropped_endpoint_l1": float(model._cropped_l1_loss(endpoint, b)),
                            "uses_gt_in_input": time_value > 0})
            maps[f"teacher_t{time_value:g}"] = endpoint[indices.index(middle), 0].cpu().numpy().T.copy()
    return records, maps


def stage_report(model, item, cfg, folder, tag, metadata, evaluated=None):
    metrics, maps = evaluated if evaluated is not None else evaluate(model, item, cfg)
    probe, probe_maps = teacher_probe(model, item)
    maps.update(probe_maps)
    save_maps(folder / f"{tag}.npz", maps)
    row = {"tag": tag, **metadata, "model_state_sha256": model_hash(model.state_dict()), **metrics}
    write_json(folder / f"{tag}.json", {"metrics": row, "teacher_probe": probe})
    return row


def train_fit(cfg, item, root, epochs, expected, resume=False):
    import torch
    from flowoct.runner import make_model, rng_state, restore_rng
    checkpoint_dir = root / "checkpoints"
    folder = root / "review"
    proof = item_proof(item)
    history, best_psnr, best_epoch, start = [], float("-inf"), None, 1
    model = make_model(cfg, True)
    if resume:
        payload, _ = stable_checkpoint(checkpoint_dir / "latest.pt")
        if payload.get("format") != FORMAT or payload["fixed_input"] != proof:
            raise ValueError("Resume checkpoint format or fixed input differs")
        if payload["helper_sha256"] != sha_file(__file__) or payload["baseline"] != expected:
            raise ValueError("Diagnostic code or baseline changed since checkpoint")
        for key, value in payload["config"].items():
            if key not in ("device", "epochs") and cfg.get(key) != value:
                raise ValueError(f"Resume config differs: {key}")
        if epochs < payload["epoch"]:
            raise ValueError("Total epochs is smaller than the saved epoch")
        model.load_state_dict(payload["model"], strict=True)
        model.optimizer_G.load_state_dict(payload["optimizer"])
        history = payload["history"]
        best_psnr, best_epoch = payload["best_fit_psnr"], payload["best_fit_epoch"]
        start = payload["epoch"] + 1
        restore_rng(payload["rng"])
        events_path = folder / "resume_events.json"
        events = read_json(events_path) if events_path.exists() else []
        events.append({"time_utc": datetime.now(timezone.utc).isoformat(), "resume_after_epoch": start - 1,
                       "requested_total_epochs": epochs, "device": cfg["device"]})
        write_json(events_path, events)
        LOG.info("RESUME after epoch%d: best_fit_epoch=%d PSNR=%.6f", start - 1, best_epoch, best_psnr)
        del payload
    else:
        checkpoint_dir.mkdir()
        initial_hash = model_hash(model.state_dict())
        stage_report(model, item, cfg, folder, "initial_epoch0", {"epoch": 0, "global_step": 0, "case_exposures": 0})
        if initial_hash != model_hash(model.state_dict()):
            raise ValueError("Initial evaluation modified model weights")
        write_json(folder / "initialization.json", {"mode": "fresh random initialization", "seed": cfg["seed"],
                   "initial_model_sha256": initial_hash, "pretrained_weights_loaded": False})
    device = model.device
    batch = {"A": item["A"].unsqueeze(0), "B": item["B"].unsqueeze(0)}
    for epoch in range(start, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        tic = time.perf_counter()
        model.set_input(batch)
        # Original core performs the complete forward/backward/Adam step.
        model.forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - tic
        train_peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.
        loss = {"total": float(model.loss.detach()), **model.get_current_loss()}
        loss.update({"weighted_velocity": model.vweight * loss["vloss"],
                     "weighted_l1": model.L1weight * loss["L1loss"],
                     "weighted_contrast": model.contrastweight * loss["contrastloss"],
                     "weighted_projection": model.projweight * loss["proj_loss"]})
        tic = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics, maps = evaluate(model, item, cfg)
        eval_peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.
        improved = metrics["volume_psnr"] > best_psnr
        if improved:
            best_psnr, best_epoch = metrics["volume_psnr"], epoch
        record = {"epoch": epoch, "global_step": epoch, "case_exposures": epoch,
                  **loss, **metrics, "best_fit_epoch": best_epoch, "best_fit_psnr": best_psnr,
                  "train_seconds": train_seconds, "eval_seconds": time.perf_counter() - tic,
                  "train_peak_allocated_gib": train_peak, "eval_peak_allocated_gib": eval_peak}
        history.append(record)
        metadata = {"epoch": epoch, "global_step": epoch, "case_exposures": epoch}
        # A saved report always corresponds to the same weights as its .pt.
        report = None
        if epoch in STAGES or epoch == epochs or improved:
            report = stage_report(model, item, cfg, folder, "best_fit" if improved else f"epoch{epoch}",
                                  metadata, evaluated=(metrics, maps))
        if epoch in STAGES or epoch == epochs:
            if improved:
                shutil.copyfile(folder / "best_fit.npz", folder / f"epoch{epoch}.npz")
                stage = read_json(folder / "best_fit.json")
                stage["metrics"]["tag"] = f"epoch{epoch}"
                write_json(folder / f"epoch{epoch}.json", stage)
        payload = {"format": FORMAT, "epoch": epoch, "global_step": epoch,
                   "model": model.state_dict(), "optimizer": model.optimizer_G.state_dict(),
                   "rng": rng_state(), "config": cfg, "fixed_input": proof,
                   "helper_sha256": sha_file(__file__), "baseline": expected, "history": history,
                   "best_fit_psnr": best_psnr, "best_fit_epoch": best_epoch,
                   "selection": "Training case volume PSNR after OCT-only 3-step generation; evaluated every epoch; strict greater-than"}
        if improved:
            atomic_checkpoint(checkpoint_dir / "best_fit.pt", payload)
        if epoch in STAGES or epoch == epochs:
            atomic_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", payload)
        # Latest is the epoch-boundary commit; it is written after stage assets.
        atomic_checkpoint(checkpoint_dir / "latest.pt", payload)
        write_csv(folder / "training_history.csv", history)
        write_json(root / "status.json", {"state": "training", "epoch": epoch, "total_epochs": epochs,
                                         "best_fit_epoch": best_epoch, "best_fit_psnr": best_psnr})
        LOG.info("FIT epoch %d/%d step=%d loss=%.6f volume_PSNR=%.6f FULL_span_SSIM=%.6f best_epoch=%d peak=%.2fGiB",
                 epoch, epochs, epoch, loss["total"], metrics["volume_psnr"], metrics["full_ssim_gt_span"],
                 best_epoch, train_peak)
        del payload, maps, report
    del model
    return history


def draw_reports(folder, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [f"epoch{e}" for e in STAGES] + ["best_fit"]
    records, data = [], []
    for tag in tags:
        record = read_json(folder / f"{tag}.json")["metrics"]
        records.append(record)
        with np.load(folder / f"{tag}.npz", allow_pickle=False) as archive:
            data.append({k: archive[k].copy() for k in archive.files})
    write_csv(folder / "comparison_metrics.csv", records)
    titles = ["GT"] + [f"Epoch {r['epoch']}\n" + ("best fit on train case" if t == "best_fit" else f"{r['global_step']} updates")
                         for t, r in zip(tags, records)]
    for name, kinds in (("projections", ("full", "full_roi", "mip")),
                        ("central_slices", ("slice_z", "slice_y", "slice_x"))):
        fig, axes = plt.subplots(3, 6, figsize=(18, 9), constrained_layout=True)
        for row_index, kind in enumerate(kinds):
            key = "full" if kind == "full_roi" else kind
            gt = data[0]["gt_" + key]
            if any(not np.array_equal(d["gt_" + key], gt) for d in data):
                raise ValueError("GT changed between diagnostic stages")
            low, high = float(gt.min()), float(np.percentile(gt, 99))
            if high <= low:
                high = low + 1e-6
            arrays = [gt] + [d["pred_" + key] for d in data]
            for column, value in enumerate(arrays):
                if kind == "full_roi":
                    value = value[40:104, 144:208]
                axes[row_index, column].imshow(value, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
                axes[row_index, column].axis("off")
                if row_index == 0:
                    axes[row_index, column].set_title(titles[column], fontsize=10)
                if column == 0:
                    label = {"full": "FULL: Z mean", "full_roi": "Fixed detail (x144,y40,64x64)",
                             "mip": "Z maximum", "slice_z": "Central Z / en-face",
                             "slice_y": "Central Y / B-scan", "slice_x": "Central X"}[kind]
                    axes[row_index, column].set_title(label, fontsize=10)
        fig.suptitle(f"Training case {CASE} | Fixed center crop | OCT-only, 3 Euler steps\n"
                     "Shared GT display window per row; no prediction normalization, sharpening or denoising.")
        fig.savefig(folder / f"{name}_comparison.png", dpi=140, facecolor="white")
        plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), constrained_layout=True)
    best = data[-1]
    gt = best["gt_slice_y"]
    for ax, key, title in zip(axes, ("gt_slice_y", "pred_slice_y", "teacher_t0", "teacher_t0.5", "teacher_t0.9"),
                              ("GT", "OCT-only: 3 steps", "t=0 one-step (no GT input)", "t=0.5 (50% GT input)", "t=0.9 (90% GT input)")):
        ax.imshow(best[key], cmap="gray", vmin=gt.min(), vmax=np.percentile(gt, 99), interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Best-fit epoch {records[-1]['epoch']}: central Y plane | GT-mixed path is a diagnostic only")
    fig.savefig(folder / "best_fit_training_path_probe.png", dpi=140, facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    xs = [r["epoch"] for r in history]
    for ax, key, title in zip(axes.flat, ("volume_psnr", "full_ssim_gt_span", "total", "full_highpass_correlation"),
                              ("Training case volume PSNR: selection criterion", "FULL SSIM with GT span (diagnostic)",
                               "Training loss", "FULL high-pass correlation (diagnostic)")):
        ax.plot(xs, [r[key] for r in history], linewidth=1.2)
        for e in STAGES:
            ax.axvline(e, color="gray", alpha=.2, linewidth=.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Single-case epoch = optimizer update = case exposure")
        ax.grid(alpha=.15)
    fig.savefig(folder / "training_curves.png", dpi=140, facecolor="white")
    plt.close(fig)
    return records


def build_archive(review, archive):
    temporary = Path(str(archive) + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
        for path in sorted(review.rglob("*")):
            if path.is_file():
                if path.suffix == ".pt":
                    raise ValueError("Weights must not enter the lightweight review archive")
                bundle.write(path, path.relative_to(review))
    with zipfile.ZipFile(temporary) as bundle:
        bad = bundle.testzip()
        if bad:
            raise OSError(f"Archive integrity failure: {bad}")
    os.replace(temporary, archive)
    Path(str(archive) + ".sha256").write_text(f"{sha_file(archive)}  {archive.name}\n", encoding="ascii")


def finish(root, history):
    folder = root / "review"
    records = draw_reports(folder, history)
    best = read_json(folder / "best_fit.json")["metrics"]
    checkpoint_metadata = {}
    for tag, filename, epoch in [(f"epoch{e}", f"epoch_{e:04d}.pt", e) for e in STAGES] + [
            ("best_fit", "best_fit.pt", best["epoch"]), ("latest", "latest.pt", history[-1]["epoch"])]:
        payload, digest = stable_checkpoint(root / "checkpoints" / filename)
        if payload["format"] != FORMAT or payload["epoch"] != epoch or payload["global_step"] != epoch:
            raise ValueError(f"Saved diagnostic checkpoint history mismatch: {filename}")
        state_hash = model_hash(payload["model"])
        if tag != "latest" and read_json(folder / f"{tag}.json")["metrics"]["model_state_sha256"] != state_hash:
            raise ValueError(f"Saved report and checkpoint weights differ: {tag}")
        checkpoint_metadata[tag] = {"file": filename, "epoch": epoch, "global_step": epoch,
                                    "sha256": digest, "model_state_sha256": state_hash}
        del payload
    if best["epoch"] != max(history, key=lambda r: r["volume_psnr"])["epoch"]:
        raise ValueError("Selected best-fit report differs from the first maximum in full history")
    write_json(folder / "checkpoint_metadata.json", checkpoint_metadata)
    write_json(folder / "summary.json", {
        "case_id": CASE, "source_split": "train", "completed_epochs": history[-1]["epoch"],
        "optimizer_updates": history[-1]["global_step"], "best_fit_epoch": best["epoch"],
        "best_fit_volume_psnr": best["volume_psnr"], "compared_checkpoints": records,
        "selection": "Highest full-volume PSNR on this same training crop, measured every epoch with OCT-only 3-step generation",
        "limits": "A one-case fitting diagnostic, not validation/test accuracy. A blurry endpoint does not identify a unique cause."})
    notes = f"""# 单病例拟合检查结果

病例：{CASE}，原划分为训练集 NORMAL。全新随机初始化，固定中心256立方裁剪，无翻转。
已完成 {history[-1]['epoch']} 轮，即 {history[-1]['global_step']} 次优化器更新。
单病例最佳拟合轮：{best['epoch']}，该训练病例三维 PSNR={best['volume_psnr']:.6f}。

- projections_comparison.png：GT / 70 / 100 / 200 / 300 / 实际最佳，FULL、固定局部和Z最大投影。
- central_slices_comparison.png：相同轮次的三个方向中心切片。
- training_curves.png、training_history.csv：每轮损失、完整生成指标、更新次数和峰值显存。
- best_fit_training_path_probe.png：区分只输入OCT生成与含GT的训练路径探针。
- comparison_metrics.csv、各阶段.json/.npz：指标、原始二维浮点图、训练路径记录。

每轮训练后仅输入 OCT，进行完整256立方生成，再与GT计算三维PSNR。严格超过历史最佳才更新best_fit.pt。
含GT的t=0.5/0.9探针有助于检查训练路径与自由生成的差距；它们不是模型只输入OCT生成的结果，也不参与选优。
GT范围SSIM和高通相关度用于观察细节，不等于论文指标或血管分割精度。
1轮=1病例=1次更新；原140病例实验的1轮=140次更新，两者不能直接按轮数比较训练总量。
本实验没有独立验证/测试集，单病例PSNR不能当作模型泛化成绩。
检查点留在上级checkpoints/，本压缩包只含二维图和记录。
"""
    (folder / "README_结果说明.md").write_text(notes, encoding="utf-8")
    for handler in logging.getLogger().handlers:
        handler.flush()
    if (root / "run.log").exists():
        shutil.copyfile(root / "run.log", folder / "run.log")
    archive = root.parent / f"{root.name}_epoch{history[-1]['epoch']}_review.zip"
    build_archive(folder, archive)
    write_json(root / "status.json", {"state": "complete", "epoch": history[-1]["epoch"],
                                     "best_fit_epoch": best["epoch"], "archive": str(archive),
                                     "archive_mib": archive.stat().st_size / 2**20})
    LOG.info("DOWNLOAD THIS ZIP: %s", archive)
    LOG.info("Archive %.2f MiB SHA256=%s", archive.stat().st_size / 2**20, sha_file(archive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "fit", "resume"))
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    parser.add_argument("--output", type=Path, help="New independent run directory; for resume, existing diagnostic directory")
    parser.add_argument("--epochs", type=int, default=400, help="Total single-case epochs, one optimizer update each")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.epochs < max(STAGES):
        parser.error("At least300 epochs are required to compare all four requested stages")
    project = args.project.expanduser().resolve()
    if not (project / "run.py").is_file():
        raise FileNotFoundError(f"Project not found: {project}")
    sys.path.insert(0, str(project))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True)
    cfg, rows, expected = preflight(project)
    cfg = dict(cfg, device=args.device, epochs=args.epochs)
    LOG.info("CHECK passed: original core/config/environment/manifest; case10308 belongs to train/NORMAL")
    if args.command == "check":
        item = load_fixed_item(rows, cfg)
        print(json.dumps({"check": "passed", "fixed_case": item_proof(item), "epochs": args.epochs,
                          "optimizer_updates": args.epochs, "sampled_y_planes_per_update": 64,
                          "network_chunk": 0, "amp": False, "initialization": "fresh random seed2026",
                          "selection": "Same training-case volume PSNR; evaluated every epoch",
                          "note": "CPU checks only. This does not verify GPU training memory."}, ensure_ascii=False, indent=2), flush=True)
        return
    from flowoct.runner import environment, setup
    import torch
    device = setup(cfg)
    item = load_fixed_item(rows, cfg)
    if args.command == "resume" and args.output is None:
        parser.error("resume requires --output pointing to the existing single-case experiment")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")
    root = args.output.expanduser().resolve() if args.output else project.parent / "experiments" / f"flowoct_singlecase10308_{stamp}"
    if args.command == "fit":
        root.mkdir(parents=True, exist_ok=False)
        (root / "review").mkdir()
    elif not (root / "checkpoints/latest.pt").is_file():
        raise FileNotFoundError(root / "checkpoints/latest.pt")
    file_handler = logging.FileHandler(root / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    LOG.info("EXPERIMENT DIRECTORY: %s", root)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        LOG.info("GPU %s free %.2f/%.2f GiB; training uses64 planes, FP32, network_chunk0",
                 torch.cuda.get_device_name(device), free / 2**30, total / 2**30)
    if args.command == "fit":
        write_json(root / "review/environment.json", environment(device))
        write_json(root / "review/fixed_input.json", item_proof(item))
        write_json(root / "review/protocol.json", {
            "config": cfg, "baseline": expected, "helper_sha256": sha_file(__file__),
            "case_id": CASE, "membership": "Original train/NORMAL; fixed center crop via private validation preprocessing; manifest unchanged",
            "initialization": "Fresh random initialization; no original checkpoint loaded; seed2026",
            "epoch_definition": "One presentation of the fixed case and one Adam update per epoch",
            "updates": args.epochs, "fixed_stages": list(STAGES), "best_fit_selection": "Same training crop volume PSNR, OCT-only3-step full generation every epoch; strict >",
            "unchanged": "Original netG, Haar, losses+ROIs+weights, Adam, random64 Y sampling, random t, FP32, network_chunk0, 3-step inference_chunk32",
            "teacher_probe": "8 fixed Y planes, t=0/0.5/0.9; at t>0 input includes GT; diagnostic only, never selected by these scores",
            "projection": "Physical Z mean=FULL and Z max; saved/displayed XY, same as previous review",
            "coverage": "Center256 cube after400x400x256 resize; not full native3mm/640Z",
            "display": "GT min-P99 shared across epochs; fixed FULL ROI x144,y40,64x64; nearest; no enhancement",
            "limits": "Not an independent validation/test study;400 updates need not guarantee exact fitting"})
    history = train_fit(cfg, item, root, args.epochs, expected, resume=args.command == "resume")
    finish(root, history)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Stopped. Resume the independent diagnostic from its own latest.pt. Original training runs were not changed.")
        raise SystemExit(1)
