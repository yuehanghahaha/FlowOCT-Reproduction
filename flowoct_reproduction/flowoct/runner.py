"""Auditable single-GPU train/validate/checkpoint/predict/evaluate workflow."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (PairedVolumeDataset, collate_volumes, inspect_header,
                   read_manifest, sha256)
from .metrics import summarize, volume_metrics


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def load_config(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        cfg = json.load(stream)
    if sorted(cfg["axis_order"]) != ["X", "Y", "Z"]:
        raise ValueError("axis_order must be a permutation of ZYX")
    if cfg["batch_size"] != 1:
        raise ValueError("Use batch_size=1: upstream projection otherwise mixes patients")
    if cfg["load_size"] < cfg["crop_size"] or cfg["crop_size"] < 32 or cfg["crop_size"] % 16:
        raise ValueError("Require load_size >= crop_size, crop_size >=32 and a multiple of 16")
    for key in ("epochs", "inference_chunk", "val_every", "save_every", "threads"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["workers"] < 0 or cfg["network_chunk"] < 0 or not 0 <= cfg["train_slices"] <= cfg["crop_size"]:
        raise ValueError("Invalid worker or training slice count")
    if cfg["lr"] <= 0 or not 0 <= cfg["beta"] < 1:
        raise ValueError("Invalid Adam settings")
    return cfg


def environment(device):
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "monai", "torchio", "scikit-image", "PyWavelets", "SimpleITK", "einops")}
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                              for path in sorted((ROOT / "flowoct").glob("*.py"))},
            "device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def setup(cfg):
    torch.set_num_threads(cfg["threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu for diagnostics")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(cfg["seed"])
    # Epoch-boundary replay is tested on the same software/hardware. This does
    # not promise identical floating point results across different GPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    versions = environment(device)["packages"]
    expected = {"torch": "2.5.1", "numpy": "1.26.4", "monai": "1.4.0", "torchio": "0.20.4"}
    for package, version in expected.items():
        if versions[package].split("+")[0] != version:
            raise RuntimeError(f"Baseline requires {package}=={version}; got {versions[package]}")
    return device


def make_model(cfg, training):
    from .model import FlowOCT
    device = torch.device(cfg["device"])
    opt = SimpleNamespace(gpu_ids=[device.index or 0] if device.type == "cuda" else [],
                          isTrain=training, checkpoints_dir=".", name="unused",
                          lr=cfg["lr"], beta=cfg["beta"], train_slices=cfg["train_slices"],
                          network_chunk=cfg["network_chunk"])
    return FlowOCT(opt)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, cfg, generator, training=False):
    return DataLoader(dataset, batch_size=1, shuffle=training, num_workers=cfg["workers"],
                      generator=generator, worker_init_fn=seed_worker, collate_fn=collate_volumes,
                      persistent_workers=False, pin_memory=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path, model, cfg, epoch, global_step, best_psnr, manifest_hash, generators):
    payload = {"format": "flowoct-reproduction-v1", "model": model.state_dict(),
               "optimizer": model.optimizer_G.state_dict(), "config": cfg,
               "epoch": epoch, "global_step": global_step, "best_psnr": best_psnr,
               "manifest_sha256": manifest_hash, "rng": rng_state(),
               "loader_rng": {key: value.get_state() for key, value in generators.items()}}
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    # Includes Python/NumPy optimizer and RNG objects. Load only your own
    # checkpoints; this project does not download or load third-party weights.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "flowoct-reproduction-v1":
        raise ValueError("Not a full reproduction checkpoint; upstream two-file weights differ")
    return payload


def to_zyx(tensor):
    return tensor.detach().float().cpu().numpy()[0, 0].transpose(2, 0, 1).copy()


@torch.no_grad()
def validate(model, loader, cfg):
    model.eval()
    scores = []
    for batch in loader:
        prediction = model.validation(batch["A"], None, cfg["inference_chunk"])
        scores.append(volume_metrics(to_zyx(prediction), to_zyx(batch["B"])))
    return summarize(scores)


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def train(cfg, output, resume=None, limits=None):
    cfg = dict(cfg)
    cfg["diagnostic_limits"] = dict(limits) if limits else None
    device = setup(cfg)
    rows = read_manifest(cfg["manifest"], cfg["fov"])
    manifest_hash = sha256(cfg["manifest"])
    output = Path(output).resolve()
    checkpoint = read_checkpoint(resume) if resume else None
    if checkpoint:
        if output != Path(resume).resolve().parent:
            raise ValueError("Resume into the checkpoint's existing run directory")
        allowed_changes = {"epochs", "device"}
        if cfg["device"] != checkpoint["config"]["device"]:
            print("Device changed: exact numerical replay is only verified on the same GPU.", flush=True)
        for key, value in cfg.items():
            if key not in allowed_changes and checkpoint["config"].get(key) != value:
                raise ValueError(f"Resume config changed: {key}")
        if manifest_hash != checkpoint["manifest_sha256"]:
            raise ValueError("Manifest changed since checkpoint")
        # Latest is the authoritative epoch-boundary state. Older snapshots
        # need a new run with an explicit protocol, rather than stale future logs.
        if Path(resume).resolve() != output / "latest.pt":
            raise ValueError("Resume from latest.pt; use best.pt for evaluation")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "environment.json", environment(device))
        shutil.copyfile(cfg["manifest"], output / "manifest.csv")
    write_json(output / "config.json", cfg)
    train_dataset = PairedVolumeDataset(rows, "train", cfg, (limits or {}).get("train", 0))
    val_dataset = PairedVolumeDataset(rows, "validation", cfg, (limits or {}).get("validation", 0))
    write_json(output / "training_scope.json", {
        "limited_diagnostic": bool(limits),
        "train_cases": [row["case_id"] for row in train_dataset.rows],
        "validation_cases": [row["case_id"] for row in val_dataset.rows]})
    generators = {key: torch.Generator().manual_seed(cfg["seed"] + offset)
                  for offset, key in enumerate(("train", "validation"))}
    loaders = {"train": make_loader(train_dataset, cfg, generators["train"], True),
               "validation": make_loader(val_dataset, cfg, generators["validation"])}
    model = make_model(cfg, True)
    start_epoch, global_step, best_psnr = 1, 0, float("-inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        model.optimizer_G.load_state_dict(checkpoint["optimizer"])
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best_psnr = checkpoint["best_psnr"]
        for key, state in checkpoint["loader_rng"].items():
            generators[key].set_state(state.cpu())
        restore_rng(checkpoint["rng"])
    if start_epoch > cfg["epochs"]:
        raise ValueError("No remaining epochs; increase --epochs to continue")
    print(json.dumps({"run": str(output), "device": str(device), "start_epoch": start_epoch,
                      "train_cases": len(train_dataset), "validation_cases": len(val_dataset),
                      "sampled_slices": cfg["train_slices"] or cfg["crop_size"] // 4,
                      "generator_parameters": sum(p.numel() for p in model.netG.parameters())}), flush=True)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        tic = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses = []
        for step, batch in enumerate(loaders["train"], 1):
            model.set_input(batch)
            model.forward()
            global_step += 1
            loss = {"total": model.loss.item(), **model.get_current_loss()}
            losses.append(loss)
            append_jsonl(output / "steps.jsonl", {"epoch": epoch, "step": global_step,
                                                    "case_id": batch["metadata"][0]["case_id"], **loss})
            if step == 1 or step % cfg["log_every"] == 0:
                print(f"Epoch {epoch}/{cfg['epochs']} scan {step}/{len(train_dataset)} loss={loss['total']:.6f}", flush=True)
        row = {"epoch": epoch, "global_step": global_step,
               "loss": {key: float(np.mean([item[key] for item in losses])) for key in losses[0]},
               "train_seconds": time.perf_counter() - tic}
        improved = False
        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            tic = time.perf_counter()
            row["validation"] = validate(model, loaders["validation"], cfg)
            row["validation_seconds"] = time.perf_counter() - tic
            psnr = row["validation"]["metrics"]["volume_psnr"]["mean"]
            improved = psnr > best_psnr
            best_psnr = max(best_psnr, psnr)
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        row["best_validation_psnr"] = best_psnr if np.isfinite(best_psnr) else None
        # No scheduler, AMP or scaler is used, matching the public FP32 loop.
        save_checkpoint(output / "latest.pt", model, cfg, epoch, global_step, best_psnr,
                        manifest_hash, generators)
        if improved:
            shutil.copyfile(output / "latest.pt", output / "best.pt")
        if epoch % cfg["save_every"] == 0:
            shutil.copyfile(output / "latest.pt", output / f"epoch_{epoch:04d}.pt")
        append_jsonl(output / "epochs.jsonl", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return output


@torch.no_grad()
def predict(cfg, checkpoint_path, output, split="test", limit=0):
    device = setup(cfg)
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["manifest_sha256"] != sha256(cfg["manifest"]):
        raise ValueError("Prediction manifest differs from the training manifest")
    for key in ("axis_order", "crop_size", "load_size", "fov"):
        if cfg[key] != checkpoint["config"][key]:
            raise ValueError(f"Prediction preprocessing differs from training: {key}")
    rows = read_manifest(cfg["manifest"], cfg["fov"])
    dataset = PairedVolumeDataset(rows, split, cfg, limit)
    loader = make_loader(dataset, cfg, torch.Generator().manual_seed(cfg["seed"]))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = make_model(cfg, False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    records = []
    for batch in loader:
        meta = batch["metadata"][0]
        prediction = model.validation(batch["A"], None, cfg["inference_chunk"])
        arrays = {"oct": to_zyx(batch["A"]), "octa": to_zyx(prediction),
                  "octa_gt": to_zyx(batch["B"])}
        np.savez_compressed(output / f"{meta['case_id']}.npz", **arrays,
                            axis_order=np.asarray("ZYX"), case_id=np.asarray(meta["case_id"]))
        record = {key: meta[key] for key in ("case_id", "patient_id", "disease", "geometry")}
        record.update(volume_metrics(arrays["octa"], arrays["octa_gt"]))
        records.append(record)
        print(f"Saved {meta['case_id']} {arrays['octa'].shape}", flush=True)
    write_json(output / "prediction.json", {"checkpoint": str(Path(checkpoint_path).resolve()),
               "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
               "manifest_sha256": checkpoint["manifest_sha256"], "split": split,
               "limited_diagnostic": bool(limit or checkpoint["config"].get("diagnostic_limits")),
               "config": cfg, "environment": environment(device),
               "coverage": "center crop after resize; not restored full FOV or native axial resolution",
               "records": records, "summary": summarize(records)})
    return output


def evaluate(predictions, output):
    predictions, output = Path(predictions).resolve(), Path(output).resolve()
    info = json.loads((predictions / "prediction.json").read_text(encoding="utf-8"))
    expected = {record["case_id"] for record in info["records"]}
    actual = {path.stem for path in predictions.glob("*.npz")}
    if not expected or actual != expected:
        raise ValueError("Prediction files do not match the recorded case set")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for record in info["records"]:
        with np.load(predictions / f"{record['case_id']}.npz", allow_pickle=False) as data:
            if data["axis_order"].item() != "ZYX":
                raise ValueError("Expected saved ZYX arrays")
            score = volume_metrics(data["octa"], data["octa_gt"])
        rows.append({"case_id": record["case_id"], "patient_id": record["patient_id"],
                     "disease": record["disease"], **score})
    with (output / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["protocol"] = {"data_range": 1, "ssim_win_size": 7,
                           "ssim": "3D volume / 2D axial maximum projection; no anatomical layers",
                           "exact_match_psnr_cap_db": float(-10 * np.log10(np.finfo(np.float64).eps)),
                           "coverage": info["coverage"], "limited_diagnostic": info["limited_diagnostic"],
                           "split": info["split"], "checkpoint_sha256": info["checkpoint_sha256"]}
    summary["by_disease"] = {disease: summarize([row for row in rows if row["disease"] == disease])
                             for disease in sorted({row["disease"] for row in rows})}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def audit(cfg, output, deep=False):
    rows = read_manifest(cfg["manifest"], cfg["fov"])
    records = []
    for row in rows:
        record = inspect_header(row, cfg["axis_order"], cfg["fov"])
        if deep:
            with np.load(row["resolved_path"], allow_pickle=False) as data:
                record["ranges"] = {}
                for key in ("oct", "octa"):
                    a = data[key]
                    low, high = float(a.min()), float(a.max())
                    if not np.isfinite(a).all() or not 0 <= low < high <= 255:
                        raise ValueError(f"Invalid {key} values: {row['case_id']}")
                    record["ranges"][key] = [low, high]
            record["sha256"] = sha256(row["resolved_path"])
        records.append(record)
    result = {"fov": cfg["fov"], "counts": dict(Counter(row["split"] for row in rows)),
              "manifest_sha256": sha256(cfg["manifest"]), "audit": "all values and file hashes" if deep else "all paths, headers, metadata and split groups",
              "records": records}
    path = Path(output).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "train", "predict", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--manifest", type=Path)
        sub.add_argument("--device")
        sub.add_argument("--output", required=True, type=Path)
        if command == "audit":
            sub.add_argument("--deep", action="store_true")
        if command == "smoke":
            sub.add_argument("--full-size", action="store_true", help="Keep formal resize/crop settings; still one train/validation scan")
        if command in ("train", "predict"):
            sub.add_argument("--inference-chunk", type=int)
        if command == "train":
            sub.add_argument("--resume", type=Path)
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--train-slices", type=int)
            sub.add_argument("--network-chunk", type=int)
        if command == "predict":
            sub.add_argument("--checkpoint", required=True, type=Path)
            sub.add_argument("--split", choices=("validation", "test"), default="test")
            sub.add_argument("--limit", type=int, default=0, help="Diagnostic only; 0 = complete split")
    sub = commands.add_parser("evaluate")
    sub.add_argument("--predictions", required=True, type=Path)
    sub.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.predictions, args.output)
        return
    cfg = load_config(args.config)
    for key in ("manifest", "device", "epochs", "train_slices", "network_chunk", "inference_chunk"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    cfg["manifest"] = str(Path(cfg["manifest"]).resolve())
    if cfg["inference_chunk"] < 1:
        raise ValueError("inference_chunk must be positive")
    # Revalidate command-line overrides through the same checks.
    if args.command == "train":
        if cfg["epochs"] < 1 or cfg["network_chunk"] < 0 or not 0 <= cfg["train_slices"] <= cfg["crop_size"] or cfg["inference_chunk"] < 1:
            raise ValueError("Invalid training overrides")
        train(cfg, args.output, args.resume)
    elif args.command == "predict":
        if args.limit < 0:
            raise ValueError("limit must be nonnegative")
        predict(cfg, args.checkpoint, args.output, args.split, args.limit)
    elif args.command == "audit":
        audit(cfg, args.output, args.deep)
    else:
        # Uses actual server data, but only one train and validation scan at 32^3.
        # The isolated output is explicitly marked diagnostic throughout.
        cfg.update(epochs=1, train_slices=0, workers=0, save_every=1, val_every=1)
        if not args.full_size:
            cfg.update(load_size=32, crop_size=32)
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        run = train(cfg, output / "train", limits={"train": 1, "validation": 1})
        predictions = predict(cfg, run / "best.pt", output / "predictions", "validation", 1)
        evaluate(predictions, output / "evaluation")
        print("Real-data diagnostic completed; not a paper performance reproduction.", flush=True)
