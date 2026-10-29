"""One real-data update at the configured training resolution; no long run."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flowoct.data import PairedVolumeDataset, collate_volumes, read_manifest
from flowoct.runner import environment, load_config, make_model, setup, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--network-chunk", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.network_chunk is not None:
        if args.network_chunk < 0:
            raise ValueError("network_chunk must be nonnegative")
        cfg["network_chunk"] = args.network_chunk
    device = setup(cfg)
    dataset = PairedVolumeDataset(read_manifest(cfg["manifest"], cfg["fov"]), "train", cfg, 1)
    batch = collate_volumes([dataset[0]])
    model = make_model(cfg, True)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    tic = time.perf_counter()
    model.set_input(batch)
    try:
        model.forward()
    except torch.cuda.OutOfMemoryError as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {"status": "CUDA_OUT_OF_MEMORY", "config": cfg,
                   "environment": environment(device), "error": str(error),
                   "seconds_until_failure": time.perf_counter() - tic})
        raise
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {"status": "PASS", "config": cfg, "environment": environment(device), "case_id": batch["metadata"][0]["case_id"],
              "input_shape": list(batch["A"].shape),
              "sampled_slices": cfg["train_slices"] or cfg["crop_size"] // 4,
              "network_chunk": cfg["network_chunk"], "seconds": time.perf_counter() - tic,
              "loss": model.loss.item(), **model.get_current_loss(),
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
              "generator_parameters": sum(p.numel() for p in model.netG.parameters())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
