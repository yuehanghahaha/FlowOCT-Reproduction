"""Compare disposable one-step models; do not load or modify research weights."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "working"))
sys.path.insert(0, str(ROOT / "flowoct_reproduction"))
from flowoct.model import FlowOCT


def main():
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location(
        "author_with_only_roi_fix", ROOT / "working/models/FlowOCT_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    opt = SimpleNamespace(gpu_ids=[0], isTrain=True, checkpoints_dir=str(HERE),
                          name="unused", lr=0.0002, beta=0.5, continue_train=False,
                          network_chunk=0, train_slices=0)
    torch.manual_seed(20261028)
    author = module.FlowOCT(opt)
    current = FlowOCT(opt)
    current.load_state_dict(author.state_dict(), strict=True)
    torch.manual_seed(123)
    batch = {"A": torch.rand(1, 1, 16, 32, 32, device="cuda:0"),
             "B": torch.rand(1, 1, 16, 32, 32, device="cuda:0")}
    losses = []
    for model in (author, current):
        model.train()
        model.set_input(batch)
        torch.manual_seed(456)
        torch.cuda.manual_seed_all(456)
        model.forward()
        losses.append({"total": model.loss.item(), **model.get_current_loss()})
    parameter_differences = []
    gradient_differences = []
    for (an, ap), (cn, cp) in zip(author.named_parameters(), current.named_parameters()):
        assert an == cn
        parameter_differences.append(float((ap - cp).abs().max()))
        if ap.grad is not None:
            gradient_differences.append(float((ap.grad - cp.grad).abs().max()))
    with torch.no_grad():
        inp = batch["A"][0, :, :4].permute(1, 0, 2, 3)
        infer_delta = float((author.generate(inp) - current.generate(inp)).abs().max())
        dwt_deltas = [float((a-b).abs().max()) for a, b in
                      zip(author.dwt_2d(inp), current.dwt_2d(inp))]
        reconstruction_delta = float((current.idwt_2d(*current.dwt_2d(inp))-inp).abs().max())
    # Demonstrate the raw author's channel-index bug independently of training.
    proj = torch.zeros(1, 32, 32)
    proj[:, 12:20, 4:10] = 1
    old_roi = proj[12:20, 4:10]
    fixed_roi = proj[:, 12:20, 4:10]
    result = {
        "scope": "One disposable FP32 update on synthetic 1x1x16x32x32, four sampled slices; same initial parameters and RNG; then three-step inference. No research checkpoint loaded.",
        "baseline": "Author code with only the two projection channel indices fixed (working/)",
        "device": torch.cuda.get_device_name(0),
        "versions": {"torch": torch.__version__},
        "losses": {"roi_fixed_author": losses[0], "current": losses[1]},
        "max_loss_difference": max(abs(losses[0][k]-losses[1][k]) for k in losses[0]),
        "max_parameter_difference_after_adam": max(parameter_differences),
        "max_gradient_difference": max(gradient_differences),
        "max_inference_difference_after_adam": infer_delta,
        "dwt_subband_max_differences": dwt_deltas,
        "haar_roundtrip_max_difference": reconstruction_delta,
        "raw_index_probe": {"old_shape": list(old_roi.shape), "fixed_shape": list(fixed_roi.shape),
                            "old_is_empty": old_roi.numel() == 0},
        "limitations": "Small synthetic input and one update cannot establish identical full-size A100 training trajectories or validate the author's unavailable data protocol. The raw original model has an intentionally repaired ROI in this comparison."
    }
    (HERE / "training_parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
