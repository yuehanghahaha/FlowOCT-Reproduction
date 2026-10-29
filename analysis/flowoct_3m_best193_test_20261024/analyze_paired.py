"""Audit matched epoch-98/193 results and make paired scientific figures."""
import hashlib
import importlib.util
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = Path(r"D:\data\FSdata")
OLD = DATA / "results/flowoct_3m_best_preview"
NEW = DATA / "results/flowoct_3m_best_after200"
CURRENT = HERE / "FULL_review"
METRICS = ("volume_mae", "volume_psnr", "volume_ssim",
           "axial_mip_mae", "axial_mip_psnr", "axial_mip_ssim",
           "full_mae", "full_psnr", "full_ssim")


def score2d(pred, target):
    delta = pred.astype(np.float64) - target
    return {"mae": float(np.mean(np.abs(delta))),
            "psnr": float(-10*np.log10(max(float(np.mean(delta**2)), np.finfo(np.float64).eps))),
            "ssim": float(structural_similarity(target, pred, data_range=1.0, win_size=7))}


def figure_case(row):
    case = row["case_id"]
    with np.load(HERE / "comparison_arrays" / f"{case}.npz", allow_pickle=False) as maps:
        arrays = {key: maps[key] for key in maps.files}
    gt, old, new = [arrays[k] for k in ("gt_full", "pred_full_98", "pred_full_193")]
    lo, hi = float(gt.min()), float(np.percentile(gt, 99))
    hi = max(hi, lo+1e-6)
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 7.2), constrained_layout=True)
    panels = [
        (gt, "真实FULL", "gray", lo, hi),
        (old, "第98轮 FULL", "gray", lo, hi),
        (new, "第193轮 FULL", "gray", lo, hi),
        (np.abs(old-gt), "98轮绝对误差", "magma", 0, hi-lo),
        (np.abs(new-gt), "193轮绝对误差", "magma", 0, hi-lo),
        (arrays["gt_bscan"], "真实中心B-scan", "gray", 0, 1),
        (arrays["bscan_98"], "第98轮 B-scan", "gray", 0, 1),
        (arrays["bscan_193"], "第193轮 B-scan", "gray", 0, 1),
        (np.abs(arrays["bscan_98"]-arrays["gt_bscan"]), "98轮切片误差", "magma", 0, .1),
        (np.abs(arrays["bscan_193"]-arrays["gt_bscan"]), "193轮切片误差", "magma", 0, .1),
    ]
    for ax, (array, title, cmap, low, high) in zip(axes.flat, panels):
        artist = ax.imshow(array, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if ax is axes[0,4] or ax is axes[1,4]:
            fig.colorbar(artist, ax=ax, fraction=.045, pad=.02)
    fig.suptitle(
        f"{case} | {row['disease']} | 同一病例、同一真实体积、同一显示窗\n"
        f"FULL SSIM: {row['old']['full_ssim']:.4f} → {row['new']['full_ssim']:.4f}；"
        f"体积SSIM: {row['old']['volume_ssim']:.4f} → {row['new']['volume_ssim']:.4f}\n"
        "FULL为深度平均；GT及两模型共用GT min–P99显示窗，指标不使用显示拉伸。切片误差共用0–0.1色标。",
        fontsize=12)
    path = HERE / "paired_examples" / f"{case}.png"
    fig.savefig(path, dpi=145, facecolor="white")
    plt.close(fig)


def main():
    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    for folder in ("comparison_arrays", "paired_examples"):
        (HERE / folder).mkdir(exist_ok=True)

    oldinfo = json.loads((OLD/"prediction.json").read_text(encoding="utf-8"))
    newinfo = json.loads((NEW/"prediction.json").read_text(encoding="utf-8"))
    oldfull = json.loads((ROOT/"analysis/flowoct_3m_best_test_20261022/octa500_full/full_metrics.json").read_text(encoding="utf-8"))
    newfull = json.loads((CURRENT/"full_metrics.json").read_text(encoding="utf-8"))
    a = {r["case_id"]: r for r in oldinfo["records"]}
    b = {r["case_id"]: r for r in newinfo["records"]}
    fa = {r["case_id"]: r for r in oldfull["records"]}
    fb = {r["case_id"]: r for r in newfull["records"]}
    assert set(a) == set(b) == set(fa) == set(fb)
    assert oldinfo["config"] == newinfo["config"]
    assert oldinfo["manifest_sha256"] == newinfo["manifest_sha256"]
    assert oldinfo["environment"]["source_sha256"] == newinfo["environment"]["source_sha256"]
    assert oldinfo["environment"]["packages"] == newinfo["environment"]["packages"]
    selected_ssim = {next(iter(b)),
        min(b, key=lambda c:b[c]["volume_ssim"]-a[c]["volume_ssim"]),
        max(b, key=lambda c:b[c]["volume_ssim"]-a[c]["volume_ssim"]),
        next(c for c in b if b[c]["disease"]=="AMD")}
    bins = {name: {"count": 0, "98": {"abs":0., "sq":0., "signed":0.},
                         "193": {"abs":0., "sq":0., "signed":0.}}
            for name in ("GT<=0.01", "0.01<GT<0.1", "GT>=0.1")}
    rows = []
    metric_errors = []
    for i, case in enumerate(b, 1):
        assert a[case]["geometry"] == b[case]["geometry"]
        assert (a[case]["patient_id"],a[case]["disease"]) == (b[case]["patient_id"],b[case]["disease"])
        with zipfile.ZipFile(OLD/f"{case}.npz") as za, zipfile.ZipFile(NEW/f"{case}.npz") as zb:
            input_header_matches = (za.getinfo("oct.npy").CRC,za.getinfo("oct.npy").file_size) == (zb.getinfo("oct.npy").CRC,zb.getinfo("oct.npy").file_size)
        assert input_header_matches, f"Input OCT serialized CRC differs: {case}"
        with np.load(OLD/f"{case}.npz",allow_pickle=False) as oa, np.load(NEW/f"{case}.npz",allow_pickle=False) as nb:
            assert oa["axis_order"].item() == nb["axis_order"].item() == "ZYX"
            gt, old, gt_new, new = oa["octa_gt"], oa["octa"], nb["octa_gt"], nb["octa"]
        assert gt.shape == old.shape == gt_new.shape == new.shape == (256,256,256)
        assert np.array_equal(gt,gt_new), f"GT voxels differ: {case}"
        gt_hash = hashlib.sha256(gt).hexdigest()
        del gt_new
        gt_full = gt.mean(axis=0,dtype=np.float64).T.astype(np.float32)
        old_map = old.mean(axis=0,dtype=np.float64).T.astype(np.float32)
        new_map = new.mean(axis=0,dtype=np.float64).T.astype(np.float32)
        with np.load(CURRENT/"projections"/f"{case}.npz",allow_pickle=False) as saved:
            assert np.array_equal(saved["gt_full"],gt_full)
            assert np.array_equal(saved["pred_full"],new_map)
        scores_old = {key: a[case][key] for key in METRICS[:6]}
        scores_new = {key: b[case][key] for key in METRICS[:6]}
        for label, pred, fullmap, logged, full_logged, assigned in (
            ("98",old,old_map,a[case],fa[case],scores_old),
            ("193",new,new_map,b[case],fb[case],scores_new)):
            difference = pred.astype(np.float64) - gt
            recalculated = {"volume_mae": float(np.abs(difference).mean()),
                            "volume_psnr": float(-10*np.log10(max(float(np.mean(difference**2)),np.finfo(np.float64).eps)))}
            for name, mask in (
                ("GT<=0.01",gt<=.01),("0.01<GT<0.1",(gt>.01)&(gt<.1)),("GT>=0.1",gt>=.1)):
                count = int(mask.sum())
                if label == "98":
                    bins[name]["count"] += count
                diff = difference[mask]
                bins[name][label]["abs"] += float(np.abs(diff).sum())
                bins[name][label]["sq"] += float(np.sum(diff**2))
                bins[name][label]["signed"] += float(diff.sum())
                del mask, diff
            del difference
            if case in selected_ssim:
                recalculated["volume_ssim"] = float(structural_similarity(gt,pred,data_range=1.0,win_size=7))
            mip = score2d(pred.max(axis=0),gt.max(axis=0))
            recalculated.update({"axial_mip_"+key:value for key,value in mip.items()})
            for name,value in recalculated.items():
                error = abs(value-logged[name])
                metric_errors.append(error)
                assert error < 1e-7, (case,label,name,value,logged[name])
            full_scores = score2d(fullmap,gt_full)
            for name,value in full_scores.items():
                assert abs(value-full_logged[name]) < 1e-10, (case,label,name)
                assigned["full_"+name] = value
        np.savez_compressed(HERE/"comparison_arrays"/f"{case}.npz",
            gt_full=gt_full,pred_full_98=old_map,pred_full_193=new_map,
            gt_bscan=gt[:,128,:],bscan_98=old[:,128,:],bscan_193=new[:,128,:],
            full_axis_order=np.asarray("XY"),bscan_axis_order=np.asarray("ZX"))
        row = {"case_id":case,"patient_id":b[case]["patient_id"],"disease":b[case]["disease"],
               "old":scores_old,"new":scores_new,
               "delta":{key:scores_new[key]-scores_old[key] for key in METRICS},
               "gt_sha256":gt_hash,"gt_voxels_identical":True,"input_oct_zip_crc_and_size_identical":True}
        rows.append(row)
        print(f"AUDIT {i}/50 {case}: FULL SSIM {scores_old['full_ssim']:.4f}->{scores_new['full_ssim']:.4f}; GT identical",flush=True)
        del gt,old,new

    rng = np.random.default_rng(20261024)
    bootstrap_indices = rng.integers(0,len(rows),size=(5000,len(rows)))
    comparison = {}
    for metric in METRICS:
        old_values=np.array([r["old"][metric] for r in rows])
        new_values=np.array([r["new"][metric] for r in rows])
        delta=new_values-old_values
        interval=np.percentile(delta[bootstrap_indices].mean(axis=1),[2.5,97.5])
        comparison[metric]={"old_mean":float(old_values.mean()),"old_std":float(old_values.std()),
            "new_mean":float(new_values.mean()),"new_std":float(new_values.std()),
            "delta_mean":float(delta.mean()),"delta_median":float(np.median(delta)),
            "improved_cases":int(np.sum(delta<0 if metric.endswith("mae") else delta>0)),
            "paired_case_bootstrap_95ci_delta_mean":interval.tolist()}
    groups={}
    for disease,count in Counter(r["disease"] for r in rows).items():
        group=[r for r in rows if r["disease"]==disease]
        groups[disease]={"cases":count,"metrics":{metric:{
            "old_mean":statistics.mean(r["old"][metric] for r in group),
            "new_mean":statistics.mean(r["new"][metric] for r in group),
            "delta_mean":statistics.mean(r["delta"][metric] for r in group)} for metric in METRICS}}
    total_voxels=sum(item["count"] for item in bins.values())
    for item in bins.values():
        item["voxel_fraction"]=item["count"]/total_voxels
        for label in ("98","193"):
            for key in ("abs","sq","signed"):
                item[label][key+"_mean"]=item[label][key]/item["count"]
    chosen=[]
    criteria=[min(rows,key=lambda r:r["delta"]["full_ssim"]),
              max(rows,key=lambda r:r["delta"]["full_ssim"]),
              min(rows,key=lambda r:r["delta"]["volume_ssim"]),
              next(r for r in rows if r["disease"]=="AMD"),
              next(r for r in rows if r["disease"]=="CNV"),rows[0]]
    for row in criteria+sorted(rows,key=lambda r:r["delta"]["full_ssim"]):
        if row["case_id"] not in chosen:
            chosen.append(row["case_id"])
        if len(chosen)==6:break
    result={"source_old":str(OLD),"source_new":str(NEW),"cases":len(rows),
        "checkpoint_epochs":[oldinfo["checkpoint_epoch"],newinfo["checkpoint_epoch"]],
        "checkpoint_sha256":[oldinfo["checkpoint_sha256"],newinfo["checkpoint_sha256"]],
        "same_case_set":True,"same_manifest":True,"same_config":True,"same_source_hashes":True,
        "same_package_versions":True,"all_50_gt_voxel_arrays_identical":True,
        "input_oct_check":"All 50 serialized OCT NPY members have matching embedded ZIP CRC32 and uncompressed length; input OCT voxel arrays not reread.",
        "recalculation":{"volume_mae_psnr_cases":50,"mip_metrics_cases":50,
                         "full_metrics_cases":50,"volume_ssim_cases":sorted(selected_ssim),
                         "max_abs_error_vs_logged":max(metric_errors)},
        "comparison":comparison,"by_disease":groups,"records":rows,
        "intensity_bin_diagnostic":{"definition":"GT-normalized intensity bins; diagnostic only, not the paper foreground mask; pooled voxel weighting.",
                                   "bins":bins},
        "example_cases":chosen,
        "uncertainty":"Paired bootstrap over 50 case IDs, seed 20261024, 5000 draws. Case/patient labels follow the provided public-ID manifest; rare disease groups have 1 case.",
        "protocol":{"full_projection":"Mean over all prepared axial samples, ZYX axis=0, transposed to XY.",
                    "display":"Same GT min-to-P99 window for GT, epoch98 and epoch193; metrics before display scaling.",
                    "coverage":"Same prepared center 256^3 crop after 400x400 resize, not original complete 3mm FOV.",
                    "model_or_training_changed":False}}
    (HERE/"paired_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

    fig,axes=plt.subplots(10,5,figsize=(20,18),constrained_layout=True)
    for ax,row in zip(axes.flat,rows):
        with np.load(HERE/"comparison_arrays"/f"{row['case_id']}.npz",allow_pickle=False) as maps:
            gt,old,new=maps["gt_full"],maps["pred_full_98"],maps["pred_full_193"]
        lo,hi=float(gt.min()),float(np.percentile(gt,99))
        ax.imshow(np.concatenate([gt,old,new],axis=1),cmap="gray",vmin=lo,vmax=max(hi,lo+1e-6),interpolation="nearest")
        ax.set_title(f"{row['case_id']} {row['disease']} | ΔFULL SSIM {row['delta']['full_ssim']:+.4f}",
                     fontsize=7,color="#126b81" if row["delta"]["full_ssim"]>=0 else "#b74840")
        ax.axis("off")
    fig.suptitle("50例FULL平均投影：每格从左到右为 真实 / 第98轮 / 第193轮\n同病例共用显示窗；蓝标题表示FULL SSIM提高，红标题表示下降。",fontsize=16)
    fig.savefig(HERE/"FULL_50例配对总览.png",dpi=160,facecolor="white")
    plt.close(fig)
    for case in chosen:
        figure_case(next(r for r in rows if r["case_id"]==case))
    fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    colors={"NORMAL":"#7c8b98","DR":"#27749b","AMD":"#ba4a40","CNV":"#bf8d30"}
    for ax,metric,title in zip(axes.flat,("volume_psnr","volume_ssim","full_psnr","full_ssim"),
                              ("体积PSNR / dB","体积SSIM","FULL平均投影PSNR / dB","FULL平均投影SSIM")):
        for disease in groups:
            group=[r for r in rows if r["disease"]==disease]
            ax.scatter([r["old"][metric] for r in group],[r["new"][metric] for r in group],
                       s=25,alpha=.8,color=colors[disease],label=f"{disease} n={len(group)}")
        values=[r[version][metric] for r in rows for version in ("old","new")]
        low,high=min(values),max(values);pad=max((high-low)*.05,.001)
        ax.plot([low-pad,high+pad],[low-pad,high+pad],ls="--",color="#777",lw=1)
        ax.set_xlim(low-pad,high+pad);ax.set_ylim(low-pad,high+pad)
        ax.set_title(f"{title}：{comparison[metric]['improved_cases']}/50例提高",fontsize=11)
        ax.set_xlabel("第98轮");ax.set_ylabel("第193轮");ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle("同一50例测试集配对比较：虚线上方表示该指标提高",fontsize=14)
    fig.savefig(HERE/"配对指标比较.png",dpi=160,facecolor="white")
    plt.close(fig)
    print(json.dumps({"comparison":comparison,"by_disease":groups,"examples":chosen,
                      "intensity_bins":bins,"ssim_verified":sorted(selected_ssim)},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()

