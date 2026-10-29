"""Write the verified paired-result report and a portable image review package."""
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=Path(r"D:\data\FSdata")
LABELS={
    "volume_mae":"体积 MAE ↓","volume_psnr":"体积 PSNR ↑ / dB","volume_ssim":"体积 SSIM ↑",
    "axial_mip_mae":"最大投影 MAE ↓","axial_mip_psnr":"最大投影 PSNR ↑ / dB","axial_mip_ssim":"最大投影 SSIM ↑",
    "full_mae":"FULL平均投影 MAE ↓","full_psnr":"FULL平均投影 PSNR ↑ / dB","full_ssim":"FULL平均投影 SSIM ↑",
}

def metric_table(comparison,keys):
    lines=["| 指标 | 第98轮 | 第193轮 | 193−98 | 改善病例 |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for key in keys:
        r=comparison[key]
        digits=5 if key.endswith("mae") else 4
        lines.append(f"| {LABELS[key]} | {r['old_mean']:.{digits}f} ± {r['old_std']:.{digits}f} | "
                     f"{r['new_mean']:.{digits}f} ± {r['new_std']:.{digits}f} | "
                     f"{r['delta_mean']:+.{digits}f} | {r['improved_cases']}/50 |")
    return "\n".join(lines)

def main():
    analysis=json.loads((HERE/"paired_analysis.json").read_text(encoding="utf-8"))
    rows=analysis["records"];comparison=analysis["comparison"]
    newmeta=json.loads((DATA/"results/flowoct_3m_best_after200/prediction.json").read_text(encoding="utf-8"))
    weights=DATA/"runs/flowoct_3m_200e/best.pt"
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==newmeta["checkpoint_sha256"]
    source=DATA/"flowoct_3m_best_after200_review.tar"
    digest=hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            digest.update(block)
    with tarfile.open(source,"r:") as package:
        files=[m for m in package.getmembers() if m.isfile()]
        errors=[]
        for member in files:
            local=DATA/member.name
            if not local.is_file() or local.stat().st_size!=member.size:
                errors.append(member.name);continue
            if not member.name.endswith(".npz"):
                if hashlib.sha256(package.extractfile(member).read()).digest()!=hashlib.sha256(local.read_bytes()).digest():
                    errors.append(member.name)
        assert not errors
        max_end=max(m.offset_data+m.size for m in files)
    assert max_end<=source.stat().st_size
    validation={
        "source_archive":str(source),"source_archive_bytes":source.stat().st_size,
        "source_archive_sha256_local":digest.hexdigest(),
        "scope":"Local archive identity and structural/readability checks; no downloaded server SHA256 sidecar was present for end-to-end checksum matching.",
        "archive_files":len(files),"npz_files":sum(m.name.endswith(".npz") for m in files),
        "all_extracted_file_sizes_match_archive":True,"all_non_npz_file_hashes_match_archive":True,
        "checkpoint_sha256_matches_prediction":True,
        "actual_checkpoint_read":{"format":"flowoct-reproduction-v1","epoch":193,"global_step":27020,
                                  "best_validation_psnr":29.741366907744993,"network_chunk":0,"epochs":200},
        "same_cases_config_manifest_sources":True,"all_50_gt_arrays_equal":True,
        "array_recalculation":analysis["recalculation"],
    }
    (HERE/"input_verification.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")

    font=Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    examples=[max(rows,key=lambda r:r["delta"]["full_ssim"]),min(rows,key=lambda r:r["delta"]["full_ssim"])]
    fig,axes=plt.subplots(2,3,figsize=(10,8),constrained_layout=True)
    for axesrow,row in zip(axes,examples):
        with np.load(HERE/"comparison_arrays"/f"{row['case_id']}.npz",allow_pickle=False) as maps:
            gt,old,new=maps["gt_full"],maps["pred_full_98"],maps["pred_full_193"]
        lo,hi=float(gt.min()),float(np.percentile(gt,99))
        for ax,array,title in zip(axesrow,(gt,old,new),
            (f"{row['case_id']} {row['disease']}\n真实FULL",
             f"第98轮\nSSIM {row['old']['full_ssim']:.4f}",
             f"第193轮\nSSIM {row['new']['full_ssim']:.4f}")):
            ax.imshow(array,cmap="gray",vmin=lo,vmax=max(hi,lo+1e-6),interpolation="nearest")
            ax.set_title(title,fontsize=11);ax.axis("off")
    fig.suptitle("FULL平均投影：改善与下降示例\n按FULL SSIM差选择最大改善/最大下降病例；同一病例共用显示窗，非总体平均。",fontsize=12)
    fig.savefig(HERE/"代表病例_FULL对照.png",dpi=150,facecolor="white")
    plt.close(fig)

    groups=analysis["by_disease"]
    group_lines=["| 类别 | 例数 | FULL PSNR：98 → 193 / dB | FULL SSIM：98 → 193 | 体积SSIM：98 → 193 |",
                 "| --- | ---: | ---: | ---: | ---: |"]
    for disease in ("NORMAL","DR","AMD","CNV"):
        r=groups[disease];m=r["metrics"]
        group_lines.append(f"| {disease} | {r['cases']} | {m['full_psnr']['old_mean']:.4f} → {m['full_psnr']['new_mean']:.4f} | "
                           f"{m['full_ssim']['old_mean']:.4f} → {m['full_ssim']['new_mean']:.4f} | "
                           f"{m['volume_ssim']['old_mean']:.4f} → {m['volume_ssim']['new_mean']:.4f} |")
    case_lines=["| 病例 | 类别 | FULL SSIM：98 → 193 | 体积SSIM：98 → 193 | 查看 |",
                "| --- | --- | ---: | ---: | --- |"]
    for case in analysis["example_cases"]:
        r=next(row for row in rows if row["case_id"]==case)
        case_lines.append(f"| {case} | {r['disease']} | {r['old']['full_ssim']:.4f} → {r['new']['full_ssim']:.4f} | "
                          f"{r['old']['volume_ssim']:.4f} → {r['new']['volume_ssim']:.4f} | [配对图](paired_examples/{case}.png) |")
    bins=analysis["intensity_bin_diagnostic"]["bins"]
    bin_lines=["| GT强度范围 | 体素占比 | 第98轮MAE | 第193轮MAE |",
               "| --- | ---: | ---: | ---: |"]
    for name,r in bins.items():
        bin_lines.append(f"| {name} | {r['voxel_fraction']*100:.2f}% | {r['98']['abs_mean']:.6f} | {r['193']['abs_mean']:.6f} |")
    ci_lines=["| 指标 | 平均差：193−98 | 病例配对bootstrap 95%区间 |",
              "| --- | ---: | --- |"]
    for key in ("volume_psnr","volume_ssim","full_psnr","full_ssim"):
        r=comparison[key];lo,hi=r["paired_case_bootstrap_95ci_delta_mean"]
        ci_lines.append(f"| {LABELS[key]} | {r['delta_mean']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    report=f"""# FlowOCT：200轮预算内最佳模型的测试与FULL投影分析

分析日期：2026-10-24。测试模型实际为第193轮，不是最后第200轮。病例：与第98轮完全相同的50例3M测试病例。

本次结论：FULL平均投影整体改善，体积PSNR也提高；体积SSIM下降，最大投影指标没有同步改善。因此不能将新模型概括为全面优于第98轮。当前结果支持分别报告体积、FULL平均投影与最大投影。

## 已生成的图和数据

- [第193轮50例FULL与中心B-scan对照图](FULL_review/figures)
- [第98/193轮全部50例FULL配对总览](FULL_50例配对总览.png)
- [改善与下降代表图](代表病例_FULL对照.png)
- [两轮测试指标配对分布](配对指标比较.png)
- [6例详细配对图](paired_examples)
- [完整配对数据与核对记录](paired_analysis.json)
- [原包、模型及解压记录核对](input_verification.json)

每张第193轮单病例PNG包含真实FULL、预测FULL、FULL绝对误差、真实中心B-scan、预测中心B-scan及其误差。配对图同时显示GT、第98轮和第193轮。

## 数据与模型核对

- 实际读取best.pt：第193轮，global_step=27020，最佳验证体积PSNR=29.7413669 dB，预算200轮，network_chunk=0。
- best.pt SHA-256与prediction.json一致：{newmeta['checkpoint_sha256']}。
- 本地tar包含50个NPZ及模型、配置和评价记录；61个解压文件的数量/尺寸与tar一致，全部非NPZ文件的内容哈希与tar一致。
- 两轮测试病例、病例/疾病标签、manifest哈希、完整运行配置、核心源码指纹和依赖版本一致。
- 全部50例真实OCTA体素逐元素一致；全部50例输入OCT的NPY成员CRC32及长度一致。没有重读全部输入OCT体素做逐元素比较。
- 全部50例体积MAE/PSNR、最大投影指标、FULL指标在本地重算；体积SSIM独立重算了{len(analysis['recalculation']['volume_ssim_cases'])}例：{', '.join(analysis['recalculation']['volume_ssim_cases'])}。重算指标与保存值最大绝对差为{analysis['recalculation']['max_abs_error_vs_logged']:.3g}。
- 没有重训、修改模型或修饰预测体积；本次是对已有预测进行投影、统计和绘图。

本地原包身份SHA-256：{digest.hexdigest()}。未取得服务器校验侧文件，因此这里的哈希用于本地追溯，不声称已完成与服务器原包的端到端校验。

## FULL平均投影结果

逐病例计算后等权平均；±为50例间总体标准差。

{metric_table(comparison,('full_mae','full_psnr','full_ssim'))}

FULL MAE下降约{(1-comparison['full_mae']['new_mean']/comparison['full_mae']['old_mean'])*100:.2f}%，PSNR提高0.6291 dB，SSIM提高0.01155。
50例中33例FULL MAE/PSNR改善，36例FULL SSIM改善，14例SSIM下降。SSIM差的中位数为+0.00321，小于均值+0.01155，表明少数较大改善对均值有贡献。
即使去掉AMD和CNV单例，正常及DR组的FULL平均指标也仍有改善，见分组表。

这些FULL指标是“预测体积的深度平均”与“同病例真实体积的深度平均”比较，范围为当前预处理后的中心体块。这里不是直接对官方显示BMP评分，也不是论文表格里的full volume指标；41.54 dB不能直接与论文体积PSNR比较。

## 体积与最大投影

{metric_table(comparison,('volume_mae','volume_psnr','volume_ssim','axial_mip_mae','axial_mip_psnr','axial_mip_ssim'))}

体积PSNR在47/50例提高，但体积SSIM仅9/50例提高、41例下降。体积MAE均值上升约{(comparison['volume_mae']['new_mean']/comparison['volume_mae']['old_mean']-1)*100:.2f}%。
最大投影PSNR平均下降0.2236 dB，最大投影SSIM基本持平。不能从FULL改善推断所有深度结构和极值血管信号都改善。

## 为什么FULL改善、体积SSIM却下降

指标关注点不同。体积PSNR来自三维MSE，FULL先沿深度平均再评价，三维SSIM则保留局部三维邻域信息。平均投影会合并不同深度的信号，降低部分深度误差的可见性；某些不同位置的正负误差还可能在平均时抵消。

本次额外进行了GT强度分组诊断：

{chr(10).join(bin_lines)}

GT<=0.01的低信号体素占77.37%，其MAE从0.002017上升到0.003593，约增加78.1%；中高强度区域MAE则下降。
这说明新模型减少了部分较大误差，同时在大量低信号位置留下了更多小幅残留。它与“体积PSNR升高，但MAE/SSIM下降”并存的现象一致。

这些阈值只用于本次误差诊断，不是论文前景掩膜、血管分割或训练修改。它提供一个有数据支持的解释，不能单独证明SSIM下降的全部原因或判定过拟合。

## 图像观察与代表病例

{chr(10).join(case_lines)}

- **10325（AMD，单例）**：第193轮减轻了第98轮的大片过亮现象，中心低信号区域及总体强度更接近GT。FULL和体积指标均改善。但细血管和局部结构仍有模糊与不匹配；只有1例，不能据此声称AMD整体效果已提升。
- **10331（NORMAL）**：FULL SSIM下降最多，右侧偏亮残留更明显；三维SSIM也下降。是“训练更久并不保证每例更好”的反例。
- **10330（NORMAL）**：FULL两轮看起来相近，SSIM仅小幅下降，但体积SSIM从0.7934降到0.6198。中心切片误差图显示193轮低信号区域残留更明显，说明只看平面血管图会漏掉三维问题。
- **10433（DR）**：FULL SSIM从0.9268降到0.9059，作为DR组下降病例保留。
- **10367（CNV）**：FULL小幅改善，体积SSIM近似持平；只有1例，不能作组层面的结论。
- **10305（NORMAL）**：作为固定复核病例保留，FULL变化很小。

示例选择包含FULL SSIM最大改善、最大下降、体积SSIM最大下降、稀有疾病单例和固定复核病例。完整50例总览同时提供，避免只展示好看的病例。

![代表病例](代表病例_FULL对照.png)

## 疾病分组

{chr(10).join(group_lines)}

NORMAL有40例，DR有8例；AMD和CNV各1例。体积SSIM的下降主要体现在NORMAL组，不能把AMD单例的大幅改善作为整体结果的替代。

## 统计与可比性

{chr(10).join(ci_lines)}

上述区间按同一病例的193−98差值做5000次病例bootstrap，seed=20261024，仅作为当前50例的探索性不确定性描述。病例/患者标签使用现有公开ID清单，稀有疾病样本不足，不提供临床或跨中心泛化结论。

## 投影与显示规则

1. NPZ中的轴顺序为ZYX，FULL取axis=0的算术平均，再转置为XY显示。
2. 每个病例的GT、第98轮和第193轮共用GT FULL min至P99灰度窗。不同病例窗口可以不同，不能直接比较跨病例的屏幕亮度。
3. 指标在未做显示拉伸的归一化[0,1]数组上计算，SSIM窗口7。
4. 当前输出是400x400缩放后的中心256x256横向范围和256层准备深度，横向为原3M视野的64%，约1.92mm；没有恢复完整3mm原始视野。
5. FULL是全保存深度平均，并不表示原始完整横向视野，也不恢复原始640层深度采样。
6. 切片图只展示中心Y切片；全体积指标和逐体素GT核对覆盖全部数据，不能用一个切片概括每个深度的结构。

## 后续实验建议

保留当前200轮预算下、按验证体积PSNR选出的第193轮作为主测试记录，同时报告FULL改善与体积SSIM下降。第98轮作为训练阶段对照，不在看完测试指标后随意替换主模型。
400轮延长训练已按原计划单独记录；结束后仍按同一验证规则选择模型，再用同一50例和同一投影、显示及指标设置评价。当前结果不能保证再训练一定改善SSIM，也没有据此修改损失、学习率或后处理预测值。

## 文件说明

FULL_review/figures保存第193轮50例PNG；FULL_review/projections保存二维GT和预测FULL。
comparison_arrays保存50例GT、98/193 FULL数组和同一中心B-scan，便于复核配对图。它们是轻量二维数据，原三维NPZ保持在D:/data/FSdata。
paired_analysis.json保留全部逐例指标、配对差和诊断结果。input_verification.json保留来源核对范围。
"""
    report_path=HERE/"FlowOCT_200轮最佳模型测试分析.md"
    report_path.write_text(report,encoding="utf-8")
    readme="""FlowOCT 200-epoch-budget test review (checkpoint epoch 193)
Start with FlowOCT_200轮最佳模型测试分析.md.
FULL_review/figures: all 50 current FULL mean/B-scan PNG comparisons.
FULL_50例配对总览.png: GT | epoch98 | epoch193 for every case.
paired_examples/: six detailed matched examples, including improvements and declines.
comparison_arrays/: lightweight paired FULL arrays and central B-scans.
FULL_review/*.json and per_case.csv: original test records and computed FULL metrics.
paired_analysis.json: complete paired analysis; input_verification.json: audit scope.
The original full 3D NPZ files and model checkpoint are not duplicated in this review ZIP.
"""
    (HERE/"README_先读.txt").write_text(readme,encoding="utf-8")
    archive=HERE/"FlowOCT_200轮投影与分析.zip"
    if archive.exists():
        raise FileExistsError(archive)
    included=[
        report_path,HERE/"README_先读.txt",HERE/"paired_analysis.json",HERE/"input_verification.json",
        HERE/"FULL_50例配对总览.png",HERE/"代表病例_FULL对照.png",HERE/"配对指标比较.png"]
    for folder in ("FULL_review/figures","comparison_arrays","paired_examples"):
        included.extend(path for path in (HERE/folder).rglob("*") if path.is_file())
    for name in ("full_metrics.json","prediction.json","summary.json","per_case.csv"):
        included.append(HERE/"FULL_review"/name)
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as package:
        for path in included:
            package.write(path,path.relative_to(HERE))
    archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    Path(str(archive)+".sha256").write_text(f"{archive_hash}  {archive.name}\n",encoding="utf-8")
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert len([name for name in package.namelist() if name.startswith("FULL_review/figures/") and name.endswith(".png")])==50
        assert len([name for name in package.namelist() if name.startswith("comparison_arrays/") and name.endswith(".npz")])==50
        assert len([name for name in package.namelist() if name.startswith("paired_examples/") and name.endswith(".png")])==6
    print(json.dumps({"report":str(report_path),"archive":str(archive),
                      "archive_mib":archive.stat().st_size/2**20,
                      "source_archive_sha256":digest.hexdigest(),
                      "all_50_gt_equal":True,"zip_integrity":"passed",
                      "full_figure_count":50,"paired_example_count":6},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

