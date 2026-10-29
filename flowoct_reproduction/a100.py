"""A100 entry point: discover existing data and use the unchanged training loop."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def find_data(fov, explicit=None):
    """Locate CSV roots only; do not traverse large volume directories."""
    filename = f"split_{fov.upper()}_z256.csv"
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        bases = (Path.cwd(), ROOT, ROOT.parent, ROOT.parent.parent, Path.home())
        candidates = []
        for base in bases:
            candidates.extend((base, base / "OCTA500_v1", base / "OCTA500_common_v1",
                               base / "data/OCTA500_v1", base / "data/OCTA500_common_v1"))
    found = sorted({path.resolve() for path in candidates
                    if (path / filename).is_file()})
    if len(found) != 1:
        detail = "\n".join(str(path) for path in found) or "No matching CSV found."
        raise ValueError(
            f"Need exactly one data directory containing {filename}.\n{detail}\n"
            "Extract this project inside your data directory, or pass "
            "--data-dir /absolute/path/to/the/CSV/directory. "
            "A directory containing only native data is not supported.")
    data = found[0]
    if not (data / "z256" / fov.upper()).is_dir():
        raise FileNotFoundError(data / "z256" / fov.upper())
    return data


def make_config(fov, data, device):
    cfg = json.loads((ROOT / "configs" / f"{fov}.json").read_text(encoding="utf-8-sig"))
    expected = {"epochs": 200, "network_chunk": 0, "inference_chunk": 32,
                "train_slices": 0, "batch_size": 1, "crop_size": 256}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Baseline template changed; expected 200 epochs, 64 full slices and FP32.")
    cfg.update(manifest=str(data / f"split_{fov.upper()}_z256.csv"), device=device)
    return cfg


def write_config(path, cfg):
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def invoke(*arguments):
    command = [sys.executable, "-u", *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def check_gpu(cfg):
    # Exit the probe before training so its CUDA context does not occupy VRAM.
    source = """import json, sys
try:
    from flowoct.runner import setup
    import torch
except ImportError:
    raise SystemExit('Activate the FlowOCT environment and install requirements.txt first.')
cfg = json.loads(sys.argv[1])
device = setup(cfg)
free, total = torch.cuda.mem_get_info(device)
print(json.dumps({'device': str(device), 'gpu': torch.cuda.get_device_name(device),
                  'free_gib': round(free / 2**30, 2), 'total_gib': round(total / 2**30, 2),
                  'network_chunk': 0, 'sampled_slices': 64, 'epochs': 200}), flush=True)
"""
    subprocess.run([sys.executable, "-c", source, json.dumps(cfg)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train", "resume", "test"))
    parser.add_argument("--fov", choices=("3m", "6m"), default="3m")
    parser.add_argument("--data-dir", type=Path, help="CSV directory; normally found automatically")
    parser.add_argument("--device", default="cuda:0", help="Index within the GPUs allocated to you")
    parser.add_argument("--run-name", help="Default: flowoct_3m_200e or flowoct_6m_200e")
    args = parser.parse_args()
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("Use an allocated CUDA device, for example cuda:0")
    name = args.run_name or f"flowoct_{args.fov}_200e"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        parser.error("run-name must be a simple directory name")
    run = ROOT / "runs" / name

    if args.command in ("resume", "test"):
        if args.data_dir is not None:
            parser.error("Resume/test reuse the saved run configuration; omit --data-dir")
        saved = run / "config.json"
        cfg = json.loads(saved.read_text(encoding="utf-8"))
        if cfg["fov"] != args.fov.upper():
            raise ValueError("The saved run belongs to a different FOV")
        if cfg["network_chunk"] != 0 or cfg.get("diagnostic_limits"):
            raise ValueError("Use a formal unchunked run, not a chunked or limited diagnostic run")
        if args.command == "resume":
            invoke(ROOT / "run.py", "train", "--config", saved, "--device", args.device,
                   "--output", run, "--resume", run / "latest.pt")
        else:
            predictions = ROOT / "results" / name
            evaluation = ROOT / "evaluations" / name
            if predictions.exists() or evaluation.exists():
                raise FileExistsError("Test outputs already exist; they will not be overwritten")
            invoke(ROOT / "run.py", "predict", "--config", saved, "--device", args.device,
                   "--checkpoint", run / "best.pt", "--output", predictions)
            invoke(ROOT / "run.py", "evaluate", "--predictions", predictions, "--output", evaluation)
        return

    if args.command == "train" and run.exists():
        raise FileExistsError(f"Run already exists: {run}. Use resume or choose --run-name.")
    data = find_data(args.fov, args.data_dir)
    cfg = make_config(args.fov, data, args.device)
    print("Using existing data:", data, flush=True)
    check_gpu(cfg)
    if args.command == "check":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")
        output = ROOT / "verification" / f"a100_{args.fov}_{stamp}"
        output.mkdir(parents=True, exist_ok=False)
        config = output / "config.json"
        write_config(config, cfg)
        invoke(ROOT / "run.py", "audit", "--config", config, "--output", output / "audit.json")
        invoke(ROOT / "scripts/benchmark.py", "--config", config, "--output", output / "benchmark.json")
        invoke(ROOT / "run.py", "smoke", "--config", config, "--full-size", "--output", output / "fullsize")
        print("Checks passed. Diagnostic records:", output, flush=True)
        print(f"Start the separate 200-epoch run with: python a100.py train --fov {args.fov}", flush=True)
    else:
        config = ROOT / "configs" / f"{args.fov}_a100.json"
        write_config(config, cfg)
        invoke(ROOT / "run.py", "train", "--config", config, "--output", run)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"A100 launcher stopped: {error}")
