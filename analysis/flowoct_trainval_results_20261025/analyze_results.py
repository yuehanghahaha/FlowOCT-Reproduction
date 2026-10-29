"""Independently audit the downloaded train/validation epoch70/140/313 maps."""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = Path(r"D:\data\FSdata\flowoct")
ARCHIVE = SOURCE / "flowoct_trainval_70_140_best_20261025_070855_529336_utc_review.zip"
EXPECTED_ARCHIVE = "2352a755215f55b4f4e0970ae0206118a301e433aa0c49a64531530bc8891755"
TAGS = ("epoch70", "epoch140", "best_at_start")
EPOCHS = (70, 140, 313)
PROJECTIONS = ("full", "mip")
METRICS = ("mae", "psnr_range1", "ssim_range1", "psnr_gt_span", "ssim_gt_span")


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0,1])


def score(real, fake):
    difference = fake.astype(np.float64)-real
    mse = max(float(np.mean(difference**2)), np.finfo(np.float64).eps)
    span = float(np.ptp(real))
    assert span > 0
    return {"mae":float(np.mean(np.abs(difference))),
            "psnr_range1":float(-10*np.log10(mse)),
            "ssim_range1":float(structural_similarity(real,fake,data_range=1.,win_size=7)),
            "psnr_gt_span":float(10*np.log10(span**2/mse)),
            "ssim_gt_span":float(structural_similarity(real,fake,data_range=span,win_size=7))}


def diagnostics(real, fake):
    real, fake = real.astype(np.float64), fake.astype(np.float64)
    hp_real, hp_fake = real-gaussian_filter(real,1), fake-gaussian_filter(fake,1)
    return {"highpass_correlation":corr(hp_real,hp_fake),
            "lowpass_correlation":corr(gaussian_filter(real,3),gaussian_filter(fake,3)),
            "highpass_std_pred_over_gt":float(hp_fake.std()/hp_real.std()),
            "signed_bias":float(np.mean(fake-real)),
            "gt_mean":float(real.mean()),"prediction_mean":float(fake.mean()),
            "gt_span":float(np.ptp(real)),
            "gradient_X_correlation":corr(np.diff(real,axis=0),np.diff(fake,axis=0)),
            "gradient_Y_correlation":corr(np.diff(real,axis=1),np.diff(fake,axis=1))}


def aggregate(rows, fields):
    return {key:{"mean":float(np.mean([r[key] for r in rows])),
                 "std":float(np.std([r[key] for r in rows],ddof=0)),
                 "min":float(min(r[key] for r in rows)),"max":float(max(r[key] for r in rows))}
            for key in fields}


def figures(results):
    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"train":"#1d6b9b","validation":"#d66b25"}
    labels = {"train":"训练病例10例","validation":"验证病例10例"}
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for axis,(projection,metric,title) in zip(axes.flat,(
            ("full","ssim_gt_span","FULL：GT灰度跨度诊断SSIM ↑"),
            ("full","mae","FULL：MAE ↓"),
            ("mip","ssim_gt_span","Z最大投影：GT跨度诊断SSIM ↑"),
            ("full","highpass_correlation","FULL：细纹理相关 ↑"))):
        for split in ("train","validation"):
            selected = [r for r in results["records"] if r["split"]==split and r["projection"]==projection]
            for case in sorted({r["case_id"] for r in selected}):
                values = [next(r[metric] for r in selected if r["case_id"]==case and r["tag"]==tag) for tag in TAGS]
                axis.plot(EPOCHS,values,color=colors[split],alpha=.16,linewidth=1)
            means = [float(np.mean([r[metric] for r in selected if r["tag"]==tag])) for tag in TAGS]
            axis.plot(EPOCHS,means,"o-",color=colors[split],linewidth=2.5,label=labels[split])
            for epoch,value in zip(EPOCHS,means):
                axis.annotate(f"{value:.4f}" if metric=="mae" else f"{value:.3f}",
                              (epoch,value),xytext=(0,8 if split=="validation" else -15),
                              textcoords="offset points",ha="center",color=colors[split],fontsize=9)
        axis.set_title(title,fontsize=12)
        axis.set_xlabel("实际训练轮次")
        axis.set_xticks(EPOCHS)
        axis.grid(alpha=.2)
        if axis is axes[0,0]:
            axis.legend()
    fig.suptitle("70 → 140 → 313轮：训练与验证投影均改善，细节差距仍存在\n细线为逐病例，粗线为10例均值；GT跨度SSIM与细纹理相关均为诊断指标",fontsize=14)
    fig.savefig(HERE/"训练与验证_三阶段趋势.png",dpi=150,facecolor="white")
    plt.close(fig)

    for split in ("train","validation"):
        selected=[r for r in results["records"] if r["split"]==split and r["projection"]=="full"]
        fig,ax=plt.subplots(figsize=(9.5,4.8),constrained_layout=True)
        cases=sorted({r["case_id"] for r in selected})
        x=np.arange(len(cases))
        for offset,tag,color,epoch in zip((-.25,0,.25),TAGS,("#91b7c9","#4e8ca6","#1f566e"),EPOCHS):
            values=[next(r["ssim_gt_span"] for r in selected if r["case_id"]==case and r["tag"]==tag) for case in cases]
            ax.bar(x+offset,values,width=.23,color=color,label=f"第{epoch}轮")
        ax.set_xticks(x,cases)
        ax.set_ylim(0,.56)
        ax.set_ylabel("FULL GT跨度诊断SSIM")
        ax.set_title(labels[split]+"：每例三个阶段逐步改善")
        ax.legend(ncol=3)
        ax.grid(axis="y",alpha=.2)
        fig.savefig(HERE/f"{split}_逐例结构变化.png",dpi=150,facecolor="white")
        plt.close(fig)

    # Fixed normal cases selected by list position, plus train/validation AMD pair.
    examples = [("train","10308","训练 NORMAL，固定首例"),
                ("validation","10319","验证 NORMAL，固定首例"),
                ("train","10310","训练 AMD，单例"),
                ("validation","10456","验证 AMD，单例")]
    fig,axes=plt.subplots(4,4,figsize=(12,12.5),constrained_layout=True)
    for row,(split,case,label) in enumerate(examples):
        with np.load(SOURCE/"projections"/f"{split}_{case}.npz",allow_pickle=False) as data:
            values=[data["gt_full"]]+[data[tag+"_full"] for tag in TAGS]
        gt=values[0]
        lo,hi=float(gt.min()),float(np.percentile(gt,99))
        for col,(array,title) in enumerate(zip(values,(label+"\n真实FULL","第70轮","第140轮","第313轮"))):
            axes[row,col].imshow(array,cmap="gray",vmin=lo,vmax=hi,interpolation="nearest")
            if col:
                value=next(r["ssim_gt_span"] for r in results["records"] if (r["split"],r["case_id"],r["projection"],r["tag"])==(split,case,"full",TAGS[col-1]))
                title+=f"\n诊断SSIM={value:.3f}"
            axes[row,col].set_title(title,fontsize=10)
            axes[row,col].axis("off")
    fig.suptitle("训练病例也存在细血管粘连与纹理不匹配；三阶段都有改善\n每行GT和三个模型共用GT灰度窗；类别同配额，病例并非疾病严重程度匹配",fontsize=13)
    fig.savefig(HERE/"固定训练验证病例_FULL对照.png",dpi=145,facecolor="white")
    plt.close(fig)


def main():
    verified=[]
    if ARCHIVE.is_file():
        digest=sha_file(ARCHIVE)
        assert digest==EXPECTED_ARCHIVE
        with zipfile.ZipFile(ARCHIVE) as archive:
            assert archive.testzip() is None
            for member in archive.infolist():
                if member.is_dir(): continue
                path=(SOURCE/member.filename).resolve()
                assert path.is_relative_to(SOURCE.resolve()) and path.is_file()
                assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(archive.read(member)).digest()
                verified.append(member.filename)
        archive_scope="ZIP hash, CRC and all extracted-file bytes checked in this execution"
    else:
        earlier=read_json(HERE/"archive_verified_before_reanalysis.json")
        assert earlier["sha256"]==EXPECTED_ARCHIVE and earlier["extracted_files_verified"]==54
        verified=[str(p.relative_to(SOURCE)) for p in SOURCE.rglob("*") if p.is_file()]
        archive_scope=earlier["scope"]
    assert len(verified)==54
    cases=read_csv(SOURCE/"case_list.csv")
    protocol=read_json(SOURCE/"protocol.json")
    checkpoints=read_json(SOURCE/"checkpoint_metadata.json")
    environment=read_json(SOURCE/"environment.json")
    proofs=read_json(SOURCE/"case_preprocessing_proof.json")
    logged=read_csv(SOURCE/"metrics_per_case.csv")
    logged_index={(r["split"],r["case_id"],r["projection"],r["tag"]):r for r in logged}
    assert len(logged_index)==len(logged)==120
    assert len(cases)==len({r["case_id"] for r in cases})==20
    assert sorted(Counter(r["source_split"] for r in cases).items())==[("train",10),("validation",10)]
    membership={r["case_id"]:r for r in read_csv(ROOT/"data/OCTA500_common_v1/split_3M_z256.csv")}
    for row in cases:
        original=membership[row["case_id"]]
        assert (original["split"],original["disease"],original["patient_id"])==(row["source_split"],row["disease"],row["patient_id"])
    for tag,epoch in zip(TAGS,EPOCHS):
        assert checkpoints[tag]["epoch"]==epoch and checkpoints[tag]["global_step"]==epoch*140
    sources_match={}
    for name,digest in protocol["source_sha256"].items():
        current=sha_file(ROOT/"flowoct_reproduction"/name)
        assert current==digest==environment["source_sha256"][name]
        sources_match[name]=True
    helper_sha=sha_file(ROOT/"FlowOCT_trainval_projection_tools/FlowOCT_trainval_projection.py")
    assert helper_sha==protocol["helper_script_sha256"]
    event_text=(SOURCE/"run.log").read_text(encoding="utf-8")
    events=re.findall(r" (\d+)/60 (train|validation) (\d+) (epoch70|epoch140|best_at_start) epoch(\d+)",event_text)
    assert [int(e[0]) for e in events]==list(range(1,61))
    assert len({(e[1],e[2],e[3]) for e in events})==60
    assert {e[1] for e in events[:30]}=={"train"} and {e[1] for e in events[30:]}=={"validation"}
    for count,split,case,tag,epoch in events:
        assert int(epoch)==checkpoints[tag]["epoch"]
    records,errors=[],[]
    keys={"gt_full","gt_mip"}|{f"{tag}_{projection}" for tag in TAGS for projection in PROJECTIONS}
    for row in cases:
        split,case=row["source_split"],row["case_id"]
        with np.load(SOURCE/"projections"/f"{split}_{case}.npz",allow_pickle=False) as data:
            assert data["axis_order"].item()=="XY" and data["case_id"].item()==case
            assert set(data.files)==keys|{"case_id","axis_order"}
            for key in keys:
                assert data[key].shape==(256,256) and data[key].dtype==np.float32
                assert np.isfinite(data[key]).all() and 0<=data[key].min()<=data[key].max()<=1
            for projection in PROJECTIONS:
                gt=data["gt_"+projection]
                for tag in TAGS:
                    prediction=data[tag+"_"+projection]
                    assert not np.array_equal(prediction,gt)
                    values=score(gt,prediction)
                    log=logged_index[split,case,projection,tag]
                    assert int(log["checkpoint_epoch"])==checkpoints[tag]["epoch"]
                    for metric in METRICS:
                        error=abs(values[metric]-float(log[metric]));errors.append(error)
                        assert error<1e-10,(case,projection,tag,metric,error)
                    records.append({"split":split,"case_id":case,"disease":row["disease"],"tag":tag,"epoch":checkpoints[tag]["epoch"],
                                    "projection":projection,**values,**diagnostics(gt,prediction)})
                assert not np.array_equal(data["epoch70_"+projection],data["epoch140_"+projection])
                assert not np.array_equal(data["epoch140_"+projection],data["best_at_start_"+projection])
        proof=proofs[f"{split}_{case}"]
        assert proof["geometry"]["crop_start_yxz"]==[72,72,0] and not proof["geometry"]["flipped_z"]
    assert len(list((SOURCE/"projections").glob("*.npz")))==20
    assert len(list((SOURCE/"figures").glob("*.png")))==20
    groups=[]
    fields=METRICS+("highpass_correlation","lowpass_correlation","highpass_std_pred_over_gt","signed_bias","gt_mean","prediction_mean","gt_span")
    for split in ("train","validation"):
        for projection in PROJECTIONS:
            for tag in TAGS:
                selected=[r for r in records if (r["split"],r["projection"],r["tag"])==(split,projection,tag)]
                groups.append({"split":split,"projection":projection,"tag":tag,"epoch":checkpoints[tag]["epoch"],"cases":len(selected),"metrics":aggregate(selected,fields)})
    log_summary=read_json(SOURCE/"summary.json")
    for group in groups:
        logged_group=next(r for r in log_summary["groups"] if (r["split"],r["projection"],r["tag"])==(group["split"],group["projection"],group["tag"]))
        for key in METRICS:
            for stat in ("mean","std"):
                error=abs(group["metrics"][key][stat]-logged_group[key][stat]); errors.append(error)
                assert error<1e-10
    by_key={(r["split"],r["case_id"],r["projection"],r["tag"]):r for r in records}
    paired=[]
    logged_paired=read_csv(SOURCE/"paired_changes.csv")
    assert len(logged_paired)==120
    lp={(r["split"],r["case_id"],r["projection"],r["comparison"]):r for r in logged_paired}
    for row in cases:
        split,case=row["source_split"],row["case_id"]
        for projection in PROJECTIONS:
            for old,new in (("epoch70","epoch140"),("epoch140","best_at_start"),("epoch70","best_at_start")):
                a,b=by_key[split,case,projection,old],by_key[split,case,projection,new]
                values={k:b[k]-a[k] for k in fields}
                comparison=f"{new}_minus_{old}"
                for metric in METRICS:
                    assert abs(values[metric]-float(lp[split,case,projection,comparison][metric+"_delta"]))<1e-10
                paired.append({"split":split,"case_id":case,"disease":row["disease"],"projection":projection,"older":old,"newer":new,"comparison":comparison,"delta":values})
    changes=[]
    for split in ("train","validation"):
        for projection in PROJECTIONS:
            for comparison in sorted({r["comparison"] for r in paired}):
                selected=[r for r in paired if (r["split"],r["projection"],r["comparison"])==(split,projection,comparison)]
                changes.append({"split":split,"projection":projection,"comparison":comparison,
                    "metrics":{metric:{"mean_delta":float(np.mean([r["delta"][metric] for r in selected])),
                        "median_delta":float(np.median([r["delta"][metric] for r in selected])),
                        "improved":sum(r["delta"][metric]<0 if metric=="mae" else r["delta"][metric]>0 for r in selected),"cases":len(selected)} for metric in fields}})
    disease_groups=[]
    for split in ("train","validation"):
        for disease in sorted({r["disease"] for r in cases}):
            selected=[r for r in records if r["split"]==split and r["disease"]==disease and r["projection"]=="full"]
            disease_groups.append({"split":split,"disease":disease,"cases":len(selected)//3,
                                  "stages":{tag:aggregate([r for r in selected if r["tag"]==tag],METRICS+("highpass_correlation",)) for tag in TAGS}})
    result={"source":str(SOURCE),"checkpoint_epochs":list(EPOCHS),"case_count":20,"metrics_recomputed":120,
            "records":records,"groups":groups,"paired_changes":changes,"paired_records":paired,"disease_groups":disease_groups,
            "verification":{"archive_sha256":EXPECTED_ARCHIVE,"matches_server_log":True,"archive_integrity":"passed","archive_files_checked":len(verified),
                "archive_verification_scope":archive_scope,
                "current_extracted_file_sha256":{str(p.relative_to(SOURCE)):sha_file(p) for p in sorted(SOURCE.rglob("*")) if p.is_file() and p.suffix!=".zip"},
                "source_membership_checked_against_original_manifest":True,"source_hash_matches":sources_match,"helper_sha256":helper_sha,
                "events_complete_and_sequential":True,"all_maps_float32_finite_0_1":True,"original_paired_rows_recomputed":len(logged_paired),"metric_max_absolute_difference":max(errors),
                "checkpoint_scope":"Downloaded projection metadata reports frozen weight identities; original weight files are not in this package and have not been rehashed locally"},
            "protocol":{"comparison":"Same case/stage is paired; train vs validation are different patients, same disease quotas, not severity-matched",
                "full":"Physical depth arithmetic mean, prepared center cube, saved XY maps","gt_span":"Diagnostic only, not paper foreground or volume score",
                "texture":"highpass=image-Gaussian(sigma1); lowpass=Gaussian(sigma3); Pearson correlation, not vessel accuracy",
                "limits":"Only 10 training patients sampled; validation cases already used for checkpoint selection; no independent test or full3D re-audit",
                "source_predictions_modified":False}}
    (HERE/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    with (HERE/"independent_metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    figures(result)
    print(json.dumps({"verification":result["verification"],"groups":[{"split":g["split"],"projection":g["projection"],"epoch":g["epoch"],
        **{k:g["metrics"][k]["mean"] for k in ("mae","ssim_gt_span","highpass_correlation","lowpass_correlation")}} for g in groups]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
