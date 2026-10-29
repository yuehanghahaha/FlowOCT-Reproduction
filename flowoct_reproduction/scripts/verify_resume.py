"""Compare two uninterrupted epochs with one epoch plus a complete resume."""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flowoct.runner import load_config, read_checkpoint, train, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    cfg.update(load_size=32, crop_size=32, epochs=2, workers=0,
               train_slices=0, save_every=1, val_every=1)
    limits = {"train": 1, "validation": 1}
    uninterrupted = train(cfg, args.output / "uninterrupted", limits=limits)
    gc.collect()
    interrupted = train({**cfg, "epochs": 1}, args.output / "resumed", limits=limits)
    gc.collect()
    train(cfg, interrupted, resume=interrupted / "latest.pt", limits=limits)
    first, second = [read_checkpoint(path / "latest.pt") for path in (uninterrupted, interrupted)]
    unequal = [name for name in first["model"] if not torch.equal(first["model"][name], second["model"][name])]
    optimizer_equal = True
    for index, state in first["optimizer"]["state"].items():
        for key, value in state.items():
            other = second["optimizer"]["state"][index][key]
            optimizer_equal &= torch.equal(value, other) if torch.is_tensor(value) else value == other
    result = {"status": "PASS" if not unequal and optimizer_equal else "FAIL",
              "config": cfg, "diagnostic_only": True, "epochs": 2,
              "unequal_model_tensors": unequal, "optimizer_equal": bool(optimizer_equal),
              "global_step_equal": first["global_step"] == second["global_step"] == 2,
              "best_validation_psnr_equal": first["best_psnr"] == second["best_psnr"],
              "comparison": "same GPU, real train/validation scans, 32^3; epoch-boundary resume"}
    write_json(args.output / "comparison.json", result)
    print(result, flush=True)
    if result["status"] != "PASS" or not result["global_step_equal"] or not result["best_validation_psnr_equal"]:
        raise AssertionError("Epoch-boundary replay differs")


if __name__ == "__main__":
    main()
