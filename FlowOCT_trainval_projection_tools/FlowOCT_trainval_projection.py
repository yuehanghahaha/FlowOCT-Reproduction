"""Compare existing epoch70/140/best weights on fixed train10 + validation10.

Read-only model evaluation. No training, no test cases, no saved 3D volumes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

DEFAULT_PROJECT = "/workspace/flowoct_work/flowoct_reproduction_a100"
MANIFEST_SHA256 = "24e12fdb4dece29f27c565b4515ba0b31417505b3af893f99e61a416b05ad4da"
TRAIN = {"10308": "NORMAL", "10310": "AMD", "10354": "NORMAL", "10373": "DR",
         "10386": "NORMAL", "10413": "NORMAL", "10426": "NORMAL", "10463": "NORMAL",
         "10474": "CNV", "10481": "NORMAL"}
VALIDATION = {"10302": "CNV", "10319": "NORMAL", "10338": "NORMAL", "10357": "NORMAL",
              "10396": "NORMAL", "10422": "NORMAL", "10447": "DR", "10456": "AMD",
              "10495": "NORMAL", "10500": "NORMAL"}
TAGS = ("epoch70", "epoch140", "best_at_start")
PROJECTIONS = ("full", "mip")
METRICS = ("mae", "psnr_range1", "ssim_range1", "psnr_gt_span", "ssim_gt_span")
ROI = (144, 40, 64)
LOG = logging.getLogger("flowoct_review")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def model_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_snapshot(source, destination=None):
    """Read a stable complete checkpoint even if training is replacing best.pt."""
    import torch
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {source}. A log cannot recreate weights.")
    for attempt in range(5):
        try:
            raw = source.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            if digest != sha_file(source):
                raise OSError("Source changed while reading")
        except Exception as error:
            if attempt == 4:
                raise RuntimeError(f"Could not read stable checkpoint {source}: {error}") from error
            time.sleep(.5)
            continue
        if checkpoint.get("format") != "flowoct-reproduction-v1":
            raise ValueError(f"Unexpected checkpoint format: {source}")
        if destination is not None:
            with Path(destination).open("xb") as stream:
                stream.write(raw)
            if sha_file(destination) != digest:
                raise OSError("Frozen checkpoint checksum mismatch")
        return checkpoint, digest


def selected_cases(rows, splits):
    indexed = {row["case_id"]: row for row in rows}
    selected = []
    for split in splits:
        expected = TRAIN if split == "train" else VALIDATION
        for case, disease in expected.items():
            if case not in indexed or indexed[case]["split"] != split or indexed[case]["disease"] != disease:
                raise ValueError(f"Fixed case metadata differs: {split}/{case}/{disease}")
            selected.append(indexed[case])
    if {r["case_id"] for r in rows if r["split"] == "validation"} != set(VALIDATION):
        raise ValueError("The original validation case set changed")
    groups = [r.get("leakage_group") or r["patient_id"] for r in selected]
    if len(groups) != len(set(groups)):
        raise ValueError("Selected cases contain a repeated image/patient group")
    return selected


def center_dataset(dataset_class, rows, split, config):
    # Filter using the real source membership first. Only this private instance's
    # preprocessing mode changes; the manifest and source membership stay intact.
    dataset = dataset_class(rows, split, config)
    dataset.split = "validation"
    return dataset


def check_checkpoint(checkpoint, config, manifest_hash, expected_epoch=None):
    epoch = int(checkpoint["epoch"])
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError(f"Expected epoch {expected_epoch}, got {epoch}")
    if checkpoint["global_step"] != epoch*140 or checkpoint["manifest_sha256"] != manifest_hash:
        raise ValueError("Checkpoint training history or manifest differs")
    keys = ("fov", "axis_order", "load_size", "crop_size", "batch_size", "train_slices",
            "network_chunk", "inference_chunk", "lr", "beta", "seed")
    if any(checkpoint["config"][key] != config[key] for key in keys):
        raise ValueError("Checkpoint model/preprocessing settings differ")
    if checkpoint["config"].get("diagnostic_limits"):
        raise ValueError("A limited diagnostic checkpoint cannot be used")


def fixed_checkpoint(base, extended, epoch):
    candidates = [run/f"epoch_{epoch:04d}.pt" for run in (base, extended)]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Fixed checkpoint is missing from both run directories: "+", ".join(map(str,candidates)))


def check_packages(expected):
    from importlib.metadata import version
    for package, wanted in expected.items():
        actual = version(package)
        if actual != wanted:
            raise ValueError(f"Training environment differs: {package} expected {wanted}, got {actual}")


def preflight(project, splits):
    # Fail before any long inference if CPU plotting/scoring dependencies are missing.
    import matplotlib
    from skimage.metrics import structural_similarity
    from flowoct.data import read_manifest
    from flowoct.runner import load_config
    base = project / "runs/flowoct_3m_200e"
    extended = project / "runs/flowoct_3m_400e_from200"
    config = load_config(base / "config.json")
    formal = {"fov": "3M", "axis_order": "ZYX", "load_size": 400, "crop_size": 256,
              "batch_size": 1, "network_chunk": 0, "train_slices": 0, "inference_chunk": 32}
    if any(config[key] != value for key, value in formal.items()):
        raise ValueError("Expected the unchanged formal 3M 256-cube baseline")
    if sha_file(config["manifest"]) != MANIFEST_SHA256:
        raise ValueError("Manifest differs from the original experiment")
    rows = read_manifest(config["manifest"], "3M")
    cases = selected_cases(rows, splits)
    current_source = {p.relative_to(project).as_posix(): sha_file(p)
                      for p in sorted((project/"flowoct").glob("*.py"))}
    for run in (base, extended):
        recorded_environment = json.loads((run/"environment.json").read_text(encoding="utf-8"))
        check_packages(recorded_environment["packages"])
        recorded = recorded_environment["source_sha256"]
        recorded = {key.replace("\\", "/"): value for key, value in recorded.items()}
        if current_source != recorded:
            raise ValueError(f"Core source differs from the trained version: {run}")
    sources = [("epoch70", fixed_checkpoint(base,extended,70), 70),
               ("epoch140", fixed_checkpoint(base,extended,140), 140),
               ("best_at_start", extended/"best.pt", None)]
    observed = []
    for tag, path, epoch in sources:
        payload, digest = checkpoint_snapshot(path)
        check_checkpoint(payload, config, MANIFEST_SHA256, epoch)
        observed.append({"tag": tag, "source": str(path), "observed_epoch": int(payload["epoch"]),
                         "observed_sha256": digest})
        del payload
    return config, cases, sources, current_source, observed


def project_zyx(volume):
    if volume.ndim != 3 or not np.isfinite(volume).all() or volume.min() < 0 or volume.max() > 1:
        raise ValueError("Expected a finite normalized ZYX volume")
    return {"full": volume.mean(axis=0, dtype=np.float64).T.astype(np.float32),
            "mip": volume.max(axis=0).T.copy()}


def predict_maps(model, item, chunk_size):
    import torch
    geometry = item["geometry"]
    if geometry["flipped_z"] or geometry["crop_start_yxz"] != [72, 72, 0]:
        raise ValueError("Expected unchanged center crop without augmentation")
    if geometry["model_axis_order"] != "YXZ":
        raise ValueError("Unexpected model axis order")
    with torch.no_grad():
        # Only OCT enters the generator. The paired OCTA is used afterward as GT.
        prediction = model.validation(item["A"].unsqueeze(0), None, chunk_size)
    generated = prediction.detach().float().cpu().numpy()[0, 0].transpose(2, 0, 1).copy()
    target = item["B"].numpy()[0].transpose(2, 0, 1).copy()
    if target.shape != generated.shape or target.shape != (256,)*3:
        raise ValueError("Expected a complete prepared 256-cube prediction")
    proof = {"geometry": geometry, "input_oct_sha256": array_hash(item["A"].numpy()),
             "gt_octa_sha256": array_hash(target)}
    return project_zyx(target), project_zyx(generated), proof


def scores(real, fake):
    from skimage.metrics import structural_similarity
    if real.shape != fake.shape or min(real.shape) < 7 or not np.isfinite(fake).all():
        raise ValueError("Invalid projection arrays")
    mse = float(np.mean((fake.astype(np.float64)-real)**2))
    mse = max(mse, np.finfo(np.float64).eps)
    span = float(np.ptp(real))
    return {"mae": float(np.mean(np.abs(fake.astype(np.float64)-real))),
            "psnr_range1": float(-10*np.log10(mse)),
            "ssim_range1": float(structural_similarity(real, fake, data_range=1., win_size=7)),
            "psnr_gt_span": float(10*np.log10(span**2/mse)) if span > 0 else None,
            "ssim_gt_span": float(structural_similarity(real, fake, data_range=span, win_size=7)) if span > 0 else None}


def metric_tables(records):
    summaries, paired = [], []
    splits = sorted({r["split"] for r in records})
    index = {(r["split"], r["case_id"], r["projection"], r["tag"]): r for r in records}
    for split in splits:
        for projection in PROJECTIONS:
            for tag in TAGS:
                group = [r for r in records if (r["split"],r["projection"],r["tag"]) == (split,projection,tag)]
                row = {"split": split, "projection": projection, "tag": tag, "cases": len(group)}
                for metric in METRICS:
                    values = [r[metric] for r in group if r[metric] is not None]
                    row[metric] = {"mean": float(np.mean(values)) if values else None,
                                   "std": float(np.std(values)) if values else None, "valid_cases": len(values)}
                summaries.append(row)
            cases = sorted({r["case_id"] for r in records if r["split"] == split})
            for older, newer in (("epoch70","epoch140"),("epoch140","best_at_start"),("epoch70","best_at_start")):
                for case in cases:
                    a, b = index[split,case,projection,older], index[split,case,projection,newer]
                    row = {"split":split,"case_id":case,"projection":projection,"comparison":f"{newer}_minus_{older}"}
                    for metric in METRICS:
                        row[metric+"_delta"] = b[metric]-a[metric] if a[metric] is not None and b[metric] is not None else None
                    paired.append(row)
    wins = []
    for split in splits:
        for projection in PROJECTIONS:
            for comparison in sorted({r["comparison"] for r in paired}):
                group = [r for r in paired if (r["split"],r["projection"],r["comparison"]) == (split,projection,comparison)]
                row = {"split":split,"projection":projection,"comparison":comparison}
                for metric in METRICS:
                    values = [r[metric+"_delta"] for r in group if r[metric+"_delta"] is not None]
                    row[metric+"_improved"] = sum(value < 0 if metric == "mae" else value > 0 for value in values)
                    row[metric+"_valid"] = len(values)
                wins.append(row)
    return {"groups":summaries,"paired_improvement_counts":wins}, paired


def save_case_arrays(path, arrays, case):
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **arrays, case_id=np.asarray(case), axis_order=np.asarray("XY"))


def draw_figures(review, cases, checkpoint_meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    titles = ["Real OCTA",f"Epoch {checkpoint_meta['epoch70']['epoch']}",
              f"Epoch {checkpoint_meta['epoch140']['epoch']}",f"Best epoch {checkpoint_meta['best_at_start']['epoch']}"]
    (review/"figures").mkdir(exist_ok=True)
    for split in dict.fromkeys(r["split"] for r in cases):
        selected = [r for r in cases if r["split"] == split]
        full_overview, full_axes = plt.subplots(len(selected),4,figsize=(12,len(selected)*3),squeeze=False,constrained_layout=True)
        mip_overview, mip_axes = plt.subplots(len(selected),4,figsize=(12,len(selected)*3),squeeze=False,constrained_layout=True)
        for index, row in enumerate(selected):
            case = row["case_id"]
            with np.load(review/"projections"/f"{split}_{case}.npz",allow_pickle=False) as saved:
                arrays = {key:saved[key] for key in saved.files if key not in ("case_id","axis_order")}
            fig, axes = plt.subplots(3,4,figsize=(12,9),constrained_layout=True)
            for projection, plot_row, overview in (("full",0,full_axes),("mip",2,mip_axes)):
                real = arrays["gt_"+projection]
                lo, hi = float(real.min()), max(float(np.percentile(real,99)),float(real.min())+1e-6)
                values = [real]+[arrays[tag+"_"+projection] for tag in TAGS]
                for col, (array,title) in enumerate(zip(values,titles)):
                    axes[plot_row,col].imshow(array,cmap="gray",vmin=lo,vmax=hi,interpolation="nearest")
                    axes[plot_row,col].set_title(f"{title} | {'FULL mean' if projection=='full' else 'Z maximum'}",fontsize=9)
                    overview[index,col].imshow(array,cmap="gray",vmin=lo,vmax=hi,interpolation="nearest")
                    overview[index,col].set_title(f"{case} {row['disease']} | {title}",fontsize=8)
                    overview[index,col].axis("off")
                    if projection == "full":
                        x,y,size = ROI
                        axes[0,col].add_patch(Rectangle((x-.5,y-.5),size,size,fill=False,edgecolor="#ff9800",linewidth=1))
                        axes[1,col].imshow(array[y:y+size,x:x+size],cmap="gray",vmin=lo,vmax=hi,interpolation="nearest")
                        axes[1,col].set_title("Same fixed FULL detail",fontsize=9)
            for axis in axes.flat:
                axis.axis("off")
            fig.suptitle(f"{split} | {case} | {row['disease']} | OCT-only inference, center crop\n"
                         "GT and all epochs share the GT display window. No sharpening or denoising.",fontsize=11)
            fig.savefig(review/"figures"/f"{split}_{case}.png",dpi=130,facecolor="white")
            plt.close(fig)
        for fig, label in ((full_overview,"FULL"),(mip_overview,"Z_MAX")):
            fig.suptitle(f"{split}: all {len(selected)} cases | "+" / ".join(titles),fontsize=12)
            fig.savefig(review/f"{split}_{label}_overview.png",dpi=120,facecolor="white")
            plt.close(fig)


def build_archive(review, archive):
    partial = archive.with_suffix(archive.suffix+".partial")
    if archive.exists() or partial.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(partial,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as bundle:
        for path in sorted(review.rglob("*")):
            if path.is_file():
                bundle.write(path,path.relative_to(review))
    with zipfile.ZipFile(partial) as bundle:
        bad = bundle.testzip()
        if bad:
            raise OSError(f"Archive integrity check failed: {bad}")
    partial.replace(archive)
    Path(str(archive)+".sha256").write_text(f"{sha_file(archive)}  {archive.name}\n",encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(DEFAULT_PROJECT))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--splits",nargs="+",choices=("train","validation"),default=["train","validation"])
    parser.add_argument("--check-only",action="store_true",help="CPU checks only; no inference or output directory")
    args = parser.parse_args()
    args.splits = list(dict.fromkeys(args.splits))
    project = args.project.expanduser().resolve()
    if not (project/"run.py").is_file():
        raise FileNotFoundError(f"Project not found: {project}")
    sys.path.insert(0,str(project))
    from flowoct.data import PairedVolumeDataset
    from flowoct.runner import environment, make_model, setup
    config,cases,sources,source_hashes,observed = preflight(project,args.splits)
    print(json.dumps({"check":"passed","source_splits":args.splits,"cases":len(cases),
                      "inferences":len(cases)*3,"checkpoints":observed},indent=2),flush=True)
    if args.check_only:
        print("Check only: no inference, no checkpoint copies, no training.",flush=True)
        return
    import torch
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")
    output = args.output.expanduser().resolve() if args.output else project.parent/"uploads"/f"flowoct_trainval_70_140_best_{stamp}"
    archive = Path(str(output)+"_review.zip")
    if archive.exists() or Path(str(archive)+".partial").exists():
        raise FileExistsError(archive)
    output.mkdir(parents=False,exist_ok=False)
    frozen,review = output/"frozen_checkpoints",output/"review"
    frozen.mkdir(); review.mkdir(); (review/"projections").mkdir()
    LOG.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",force=True,
                        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(output/"run.log",encoding="utf-8")])
    checkpoint_meta = {}
    for tag,source,expected in sources:
        destination = frozen/f"{tag}.pt"
        payload,digest = checkpoint_snapshot(source,destination)
        check_checkpoint(payload,config,MANIFEST_SHA256,expected)
        checkpoint_meta[tag] = {"source":str(source),"frozen_path":str(destination),"epoch":int(payload["epoch"]),
                                "global_step":int(payload["global_step"]),"best_validation_psnr":float(payload["best_psnr"]),
                                "sha256":digest,"model_state_sha256":model_hash(payload["model"])}
        LOG.info("Frozen %s: epoch=%d sha256=%s",tag,payload["epoch"],digest)
        del payload
    config["device"] = args.device
    device = setup(config)
    if device.type == "cuda":
        free,total = torch.cuda.mem_get_info(device)
        LOG.info("GPU %s: free %.2f / %.2f GiB",torch.cuda.get_device_name(device),free/2**30,total/2**30)
    write_json(review/"checkpoint_metadata.json",checkpoint_meta)
    write_json(review/"environment.json",environment(device))
    case_list = [{"source_split":r["split"],"case_id":r["case_id"],"patient_id":r["patient_id"],"disease":r["disease"]} for r in cases]
    write_csv(review/"case_list.csv",case_list)
    write_json(review/"protocol.json",{
        "config":config,"manifest_sha256":MANIFEST_SHA256,"source_sha256":source_hashes,
        "helper_script_sha256":sha_file(Path(__file__)),
        "selection":"Fixed seed2026 stratified train10: 7 NORMAL+1 DR+1 AMD+1 CNV, unique image groups; original validation10",
        "membership":"Source split is preserved. Private dataset instance uses validation preprocessing for both source splits.",
        "inference":"OCT-only start, eval/no_grad, FP32, 3 steps, all 256 B-scans, inference_chunk32; no optimizer",
        "projection":"Saved physical ZYX: mean(axis0) for FULL; max(axis0) for Z maximum; displayed XY",
        "metric":"Raw arrays; MAE; PSNR/SSIM range1 and per-case GT span; win7. GT-span scores are diagnostics, not paper metrics.",
        "display":"All epochs share GT min-P99 per projection/case; nearest; no sharpening/denoising",
        "roi_display_xy":[144,40,64,64],"coverage":"Prepared center256 cube, about center1.92mm lateral, not native full3mm/640Z",
        "limits":"Training-fit vs validation diagnostic, not independent test performance or proof of a causal implementation fault",
        "expected_inferences":len(cases)*3,"checkpoint_selection":"Existing validation volume PSNR best.pt, frozen at run start"})
    metric_rows,proofs,arrays_by_case = [],{},{}
    completed = 0
    model = make_model(config,False)
    for split in args.splits:
        dataset = center_dataset(PairedVolumeDataset,cases,split,config)
        for tag in TAGS:
            payload,digest = checkpoint_snapshot(frozen/f"{tag}.pt")
            if digest != checkpoint_meta[tag]["sha256"]:
                raise ValueError("Frozen checkpoint changed")
            model.load_state_dict(payload["model"],strict=True)
            del payload
            model.eval(); model.requires_grad_(False)
            for index in range(len(dataset)):
                item = dataset[index]
                case = item["case_id"]
                real,fake,proof = predict_maps(model,item,config["inference_chunk"])
                key = f"{split}_{case}"
                if key in proofs and proofs[key] != proof:
                    raise ValueError(f"Input/GT or geometry changed between checkpoints: {case}")
                proofs[key] = proof
                arrays = arrays_by_case.setdefault(key,{})
                for projection in PROJECTIONS:
                    gt_key = "gt_"+projection
                    if gt_key in arrays and not np.array_equal(arrays[gt_key],real[projection]):
                        raise ValueError("GT projections changed")
                    arrays[gt_key],arrays[tag+"_"+projection] = real[projection],fake[projection]
                    metric_rows.append({"split":split,"case_id":case,"disease":item["disease"],"tag":tag,
                                        "checkpoint_epoch":checkpoint_meta[tag]["epoch"],"projection":projection,
                                        **scores(real[projection],fake[projection]),
                                        "gt_min":float(real[projection].min()),"gt_max":float(real[projection].max()),
                                        "display_gt_p99":float(np.percentile(real[projection],99))})
                save_case_arrays(review/"projections"/f"{key}.npz",arrays,case)
                write_csv(review/"metrics_per_case.csv",metric_rows)
                completed += 1
                write_json(output/"status.json",{"state":"running","completed_inferences":completed,"total_inferences":len(cases)*3})
                LOG.info("%d/%d %s %s %s epoch%d",completed,len(cases)*3,split,case,tag,checkpoint_meta[tag]["epoch"])
                del item,real,fake
            if model_hash(model.state_dict()) != checkpoint_meta[tag]["model_state_sha256"]:
                raise ValueError("Model state changed during evaluation")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if completed != len(cases)*3 or len(metric_rows) != len(cases)*6:
        raise ValueError("Incomplete comparisons")
    summary,paired = metric_tables(metric_rows)
    write_json(review/"summary.json",summary)
    write_csv(review/"paired_changes.csv",paired)
    write_json(review/"case_preprocessing_proof.json",proofs)
    LOG.info("Inference complete. Drawing comparisons on CPU.")
    draw_figures(review,cases,checkpoint_meta)
    (review/"README.txt").write_text(
        "Projection-only review: fixed train10 and/or original validation10, three checkpoints.\n"
        "figures/: GT / epoch70 / epoch140 / frozen best, FULL / fixed detail / Z maximum.\n"
        "projections/: raw 2D float32 maps, axis XY. No 3D volumes or weights in this package.\n"
        "GT-span SSIM/PSNR are sensitivity diagnostics, not paper foreground or volume scores.\n"
        "Source training cases use center-crop/no-flip preprocessing and OCT-only generation.\n"
        "Check checkpoint_metadata.json for the ACTUAL frozen best epoch, not an assumed313.\n",encoding="utf-8")
    LOG.info("All %d inferences complete; model weights were not updated.",completed)
    for handler in logging.getLogger().handlers:
        handler.flush()
    shutil.copy2(output/"run.log",review/"run.log")
    build_archive(review,archive)
    write_json(output/"status.json",{"state":"complete","completed_inferences":completed,"archive":str(archive),"archive_mib":archive.stat().st_size/2**20})
    LOG.info("DOWNLOAD THIS ZIP: %s",archive)
    LOG.info("Archive size: %.2f MiB; SHA256: %s",archive.stat().st_size/2**20,sha_file(archive))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Stopped. Existing training and original checkpoints were not modified.")
        raise SystemExit(1)
