"""FP32 parity checks against the pinned author code; no training or model edits."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torchio as tio


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "flowoct_reproduction/reference/upstream"
sys.path.insert(0, str(REFERENCE))
sys.path.insert(0, str(ROOT / "flowoct_reproduction"))
from flowoct.model import FlowOCT


def stats(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {"max_absolute_difference": float(np.abs(delta).max()),
            "mean_absolute_difference": float(np.abs(delta).mean())}


def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "Use the local GPU: the original wavelet picks CUDA whenever available"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(Path(r"D:\data\FSdata\runs\flowoct_3m_200e\best.pt"), map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 193
    spec = importlib.util.spec_from_file_location("pinned_author_model", REFERENCE / "models/FlowOCT_model.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    opt = SimpleNamespace(gpu_ids=[0], isTrain=True, checkpoints_dir=str(HERE), name="unused",
                          lr=.0002, beta=.5, continue_train=False, network_chunk=0, train_slices=0)
    model, author = FlowOCT(opt), source.FlowOCT(opt)
    model.load_state_dict(checkpoint["model"], strict=True)
    author.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    author.eval()
    saved_path = Path(r"D:\data\FSdata\results\flowoct_3m_best_after200\10305.npz")
    with np.load(saved_path, allow_pickle=False) as data:
        assert data["axis_order"].item() == "ZYX"
        oct_volume = data["oct"]
        pred_volume = data["octa"]
    planes = [64, 128, 192]
    inp = torch.from_numpy(oct_volume[:, planes, :].transpose(1, 2, 0).copy())[:, None].cuda()
    with torch.no_grad():
        wrapped = model.generate(inp)
        original = author.generate(inp)
        chunked = model._generate_3d(inp.permute(1, 0, 2, 3)[None], chunk_size=1)[0].permute(1, 0, 2, 3)
        roundtrip = model.idwt_2d(*model.dwt_2d(inp))
    result = {
        "scope": "Fixed case 10305, three full 256x256 B-scans at Y=64,128,192, local GPU FP32, checkpoint193",
        "device": torch.cuda.get_device_name(0),
        "numeric_settings": {"matmul_tf32": False, "cudnn_tf32": False, "cudnn_deterministic": True},
        "upstream_vs_wrapper": stats(original.cpu().numpy(), wrapped.cpu().numpy()),
        "chunk1_vs_chunk3": stats(chunked.cpu().numpy(), wrapped.cpu().numpy()),
        "local_regeneration_vs_saved_a100": stats(wrapped[:, 0].cpu().numpy(), pred_volume[:, planes, :].transpose(1, 2, 0)),
        "haar_roundtrip": stats(roundtrip.cpu().numpy(), inp.cpu().numpy()),
        "generation_steps": model.generation_steps,
        "limits": "No new training, no whole-test inference rerun, and no proof of exact author offline preprocessing or training parity",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    assert result["upstream_vs_wrapper"]["max_absolute_difference"] < 1e-6
    assert result["chunk1_vs_chunk3"]["max_absolute_difference"] < 1e-4
    assert result["local_regeneration_vs_saved_a100"]["max_absolute_difference"] < 1e-3
    assert result["haar_roundtrip"]["max_absolute_difference"] < 1e-6
    del checkpoint, model, author, inp, wrapped, original, chunked, oct_volume, pred_volume
    torch.cuda.empty_cache()
    print("Inference parity checks passed", flush=True)

    # Compare native640 -> author TorchIO resize with the actual common z256 -> same resize.
    result["resize_checks"] = []
    for case in ("10305", "10325"):
        data_root = ROOT / "data/OCTA500_common_v1"
        arrays = []
        for representation in ("native", "z256"):
            with np.load(data_root / representation / "3M" / f"{case}.npz", allow_pickle=False) as data:
                arr = data["octa"].astype(np.float32)
            arr = tio.Resize((400, 400, 256))(arr.transpose(1, 2, 0).copy()[None])
            arr = np.asarray(arr, dtype=np.float32)[0, 72:328, 72:328, :]
            lo, hi = float(arr.min()), float(arr.max())
            arrays.append((arr - lo)/(hi-lo))
        native, z256 = arrays
        with np.load(saved_path.parent / f"{case}.npz", allow_pickle=False) as data:
            saved_gt = data["octa_gt"].transpose(1, 2, 0)
        check = {
            "case_id": case,
            "native_direct_vs_common_route_normalized_gt": stats(native, z256),
            "common_route_vs_saved_gt": stats(z256, saved_gt),
            "full_mean_native_direct_vs_common_route": stats(native.mean(axis=2, dtype=np.float64), z256.mean(axis=2, dtype=np.float64)),
        }
        assert check["common_route_vs_saved_gt"]["max_absolute_difference"] == 0
        result["resize_checks"].append(check)
        print("Resize checked " + case + ": " + json.dumps(check), flush=True)
        del arrays, native, z256, saved_gt, arr
    (HERE / "source_execution_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
