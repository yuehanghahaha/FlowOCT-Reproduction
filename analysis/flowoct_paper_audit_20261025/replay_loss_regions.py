"""Read-only loss-region diagnostics on three training cases; never update weights."""
import hashlib
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "flowoct_reproduction"))
from flowoct.data import PairedVolumeDataset, read_manifest
from flowoct.model import FlowOCT
from flowoct.roi import auto_crop_octa_images


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def region_stats(pred, target, octa, power):
    records = []
    losses, low_losses, low_fractions = [], [], []
    for index, (fake, real, gt) in enumerate(zip(pred, target, octa)):
        _, _, info = auto_crop_octa_images(fake, real)
        xm, ym, xM, yM = info["expanded_bbox"]
        region = (slice(ym, yM+1), slice(xm, xM+1))
        difference = fake[region].astype(np.float64) - real[region]
        error = np.abs(difference) ** power
        low = gt[region] <= .01
        losses.append(float(error.mean()))
        low_losses.append(float(np.mean(error * low)))
        low_fractions.append(float(low.mean()))
        records.append({
            "slice_in_sampled_batch": index,
            "bbox_xm_ym_xM_yM": [int(xm), int(ym), int(xM), int(yM)],
            "area_fraction": float(error.size/fake.size),
            "target_row_threshold_fallback": bool(not np.any(real.mean(axis=1) > real.max()*.05)),
        })
    return {
        "cases_or_slices": len(records),
        "loss_equal_weight_per_slice": float(np.mean(losses)),
        "mean_roi_area_fraction": float(np.mean([r["area_fraction"] for r in records])),
        "full_image_roi_count": sum(r["area_fraction"] == 1 for r in records),
        "target_no_row_above_threshold_count": sum(r["target_row_threshold_fallback"] for r in records),
        "low_gt_fraction_inside_roi_mean": float(np.mean(low_fractions)),
        "low_gt_share_of_loss": float(np.sum(low_losses)/np.sum(losses)),
        "records": records,
    }


def main():
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    path = Path(r"D:\data\FSdata\runs\flowoct_3m_200e\best.pt")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = saved["config"].copy()
    assert saved["epoch"] == 193 and cfg["network_chunk"] == 0
    rows = read_manifest(ROOT / "data/OCTA500_common_v1/split_3M_z256.csv", "3M")
    selected = [next(r for r in rows if r["split"] == "train" and r["disease"] == disease)
                for disease in ("NORMAL", "DR", "AMD")]
    dataset = PairedVolumeDataset(selected, "train", cfg)
    model = FlowOCT(SimpleNamespace(gpu_ids=[0], isTrain=True, checkpoints_dir=str(HERE),
                                   name="unused", lr=cfg["lr"], beta=cfg["beta"],
                                   train_slices=0, network_chunk=0))
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    before = state_hash(model)
    steps = [float(s["step"]) for s in saved["optimizer"]["state"].values() if "step" in s]
    assert steps and min(steps) == max(steps) == saved["global_step"]
    result = {
        "checkpoint_epoch": saved["epoch"], "checkpoint_global_step": saved["global_step"],
        "optimizer_parameter_states": len(steps), "optimizer_step_min_max": [min(steps), max(steps)],
        "scope": "Three training cases, first manifest occurrence in NORMAL/DR/AMD; one seeded random crop/flip each and 64 sampled Y B-scans, evaluated with checkpoint193",
        "limits": "Diagnostic states are not recorded historical training batches. No optimizer update, no new mask used in training, no result-based case selection.",
        "diagnostic_execution": "GPU FP32 eval/no_grad; velocity network processed in groups of4 for local GPU memory; official training config remains network_chunk0",
        "low_signal_definition": "OCTA GT<=0.01 is for descriptive accounting only, not the paper's unknown foreground mask",
        "records": [],
    }
    del saved
    for index in range(len(dataset)):
        item = dataset[index]
        a, b = item["A"].to(device), item["B"].to(device)  # C,Y,X,Z
        sampled_y = torch.randperm(a.shape[1], device=device)[:a.shape[1]//4]
        aa = a[:, sampled_y].permute(1, 0, 2, 3).contiguous()
        bb = b[:, sampled_y].permute(1, 0, 2, 3).contiguous()
        times = torch.rand(aa.shape[0], device=device)
        ti = times[:, None, None, None]
        xt = (1-ti)*aa + ti*bb
        velocity_target = bb-aa
        with torch.no_grad():
            wavelets = torch.cat(model.dwt_2d(xt), dim=1)
            predicted_w = torch.cat([model.netG(wavelets[start:start+4], times[start:start+4])
                                     for start in range(0, len(aa), 4)])
            velocity = model.idwt_2d(*(predicted_w[:, c:c+1] for c in range(4)))
            reconstructed = torch.clamp(xt+(1-ti)*velocity, 0, 1)
            actual_l2 = float(model._cropped_l2_loss(velocity, velocity_target))
            actual_l1 = float(model._cropped_l1_loss(reconstructed, bb))
            velocity_target_w = torch.cat(model.dwt_2d(velocity_target), dim=1)
            full_wavelet_mse = float(torch.mean((predicted_w-velocity_target_w)**2))
            full_pixel_mse = float(torch.mean((velocity-velocity_target)**2))
            assert abs(full_wavelet_mse-full_pixel_mse) < 1e-6
            projected_true = bb.max(dim=0)[0]
            projected_fake = reconstructed.max(dim=0)[0]
            _, _, projection_bbox = auto_crop_octa_images(projected_fake[0].cpu().numpy(), projected_true[0].cpu().numpy())
            xm, ym, xM, yM = projection_bbox["expanded_bbox"]
            original_indexed_fake = projected_fake[ym:yM+1, xm:xM+1]
            original_indexed_true = projected_true[ym:yM+1, xm:xM+1]
            fixed_indexed_fake = projected_fake[:, ym:yM+1, xm:xM+1]
            fixed_indexed_true = projected_true[:, ym:yM+1, xm:xM+1]
            projection_fix = {
                "bbox_xm_ym_xM_yM": [int(xm), int(ym), int(xM), int(yM)],
                "original_indexed_shape": list(original_indexed_fake.shape),
                "fixed_indexed_shape": list(fixed_indexed_fake.shape),
                "original_indexed_l1": float(torch.mean(torch.abs(original_indexed_fake-original_indexed_true))) if original_indexed_fake.numel() else None,
                "fixed_indexed_l1": float(torch.mean(torch.abs(fixed_indexed_fake-fixed_indexed_true))),
            }
        gt = bb[:, 0].cpu().numpy()
        velocity_regions = region_stats(velocity[:, 0].cpu().numpy(), velocity_target[:, 0].cpu().numpy(), gt, 2)
        l1_regions = region_stats(reconstructed[:, 0].cpu().numpy(), gt, gt, 1)
        assert abs(actual_l2-velocity_regions["loss_equal_weight_per_slice"]) < 1e-6
        assert abs(actual_l1-l1_regions["loss_equal_weight_per_slice"]) < 1e-6
        row = {"case_id": item["case_id"], "disease": item["disease"], "geometry": item["geometry"],
               "sampled_y_indices": sampled_y.cpu().tolist(), "time_min_max": [float(times.min()), float(times.max())],
               "velocity_roi": velocity_regions, "l1_roi": l1_regions,
               "actual_helper_losses": {"velocity_mse": actual_l2, "l1": actual_l1},
               "unmasked_haar_energy_check": {"pixel_mse": full_pixel_mse, "wavelet_mse": full_wavelet_mse},
               "projection_index_fix": projection_fix,
               "sampled_plane_axis_order": "XZ", "batch_max_projection_axis_order": "XZ"}
        result["records"].append(row)
        print(json.dumps({"case_id": row["case_id"], "disease": row["disease"],
                          "velocity_roi": {k:v for k,v in velocity_regions.items() if k != "records"},
                          "l1_roi": {k:v for k,v in l1_regions.items() if k != "records"},
                          "projection_index_fix": projection_fix,
                          "unmasked_haar_energy_check": row["unmasked_haar_energy_check"]}, ensure_ascii=False), flush=True)
        del item, a, b, aa, bb, xt, ti, times, velocity_target, wavelets, predicted_w, velocity, reconstructed, gt
        del velocity_target_w, projected_true, projected_fake, original_indexed_fake, original_indexed_true, fixed_indexed_fake, fixed_indexed_true
    after = state_hash(model)
    assert before == after
    result["model_state_unchanged"] = before == after
    result["model_state_sha256"] = before
    # Isolate the projection component: swapping sampled Y planes is invisible to its max.
    true = np.zeros((2, 3, 4), dtype=np.float32)  # Y,X,Z
    fake = np.zeros_like(true)
    true[0, :, 1] = 1
    fake[1, :, 1] = 1
    y_loss = float(np.abs(true.max(axis=0)-fake.max(axis=0)).mean())
    z_loss = float(np.abs(true.max(axis=2)-fake.max(axis=2)).mean())
    assert y_loss == 0 and z_loss == 1
    result["projection_counterexample"] = {
        "current_y_batch_max_l1": y_loss, "physical_z_max_l1": z_loss,
        "scope": "Synthetic demonstration of the projection term only; the other training losses can detect this difference",
    }
    (HERE / "loss_region_replay.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Read-only loss-region checks completed; model state unchanged", flush=True)


if __name__ == "__main__":
    main()
