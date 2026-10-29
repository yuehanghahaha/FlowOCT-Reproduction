"""Summarize the supplied FlowOCT console log without changing training files."""

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


SOURCE = Path(r"D:\data\FSdata\flowoct_3m_train.log")
OUTPUT = Path(__file__).resolve().parent
METRICS = (
    "volume_mae", "volume_psnr", "volume_ssim",
    "axial_mip_mae", "axial_mip_psnr", "axial_mip_ssim",
)


def metric(row, name):
    return row["validation"]["metrics"][name]["mean"]


def nonfinite(value, path=""):
    if isinstance(value, dict):
        return [p for key, item in value.items() for p in nonfinite(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [p for item in value for p in nonfinite(item, path)]
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def main():
    raw = SOURCE.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    parsed, malformed = [], []
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("{"):
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                malformed.append(number)
    rows = [r for r in parsed if "epoch" in r and "validation" in r]
    starts = [r for r in parsed if "start_epoch" in r]
    counts = Counter(r["epoch"] for r in rows)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    missing = [e for e in range(1, 201) if e not in counts]
    assert not duplicates and not missing and len(rows) == 200, "Inspect epoch coverage before comparing"
    assert [r["epoch"] for r in rows] == list(range(1, 201))
    by_epoch = {r["epoch"]: r for r in rows}
    best = max(rows, key=lambda r: metric(r, "volume_psnr"))
    reference = by_epoch[98]
    extrema = {}
    for name in METRICS:
        chosen = (min if name.endswith("mae") else max)(rows, key=lambda r: metric(r, name))
        extrema[name] = {"epoch": chosen["epoch"], "value": metric(chosen, name)}

    windows = []
    for lo, hi in ((1, 20), (41, 60), (81, 100), (101, 120),
                   (121, 140), (141, 160), (161, 180), (181, 200)):
        group = [r for r in rows if lo <= r["epoch"] <= hi]
        windows.append({
            "epochs": [lo, hi],
            "training_loss": {k: statistics.mean(r["loss"][k] for r in group) for k in group[0]["loss"]},
            "validation": {k: statistics.mean(metric(r, k) for r in group) for k in METRICS},
        })
    warnings = [{"line": i, "text": line} for i, line in enumerate(lines, 1) if "UserWarning:" in line]
    errors = [{"line": i, "text": line} for i, line in enumerate(lines, 1)
              if re.search(r"Traceback|KeyboardInterrupt|RuntimeError|OutOfMemoryError|out of memory|CUDA error", line)]
    running_best, consistency_errors, updates = -math.inf, [], []
    for r in rows:
        value = metric(r, "volume_psnr")
        if value > running_best:
            updates.append({"epoch": r["epoch"], "volume_psnr": value})
        running_best = max(running_best, value)
        if not math.isclose(running_best, r["best_validation_psnr"], abs_tol=1e-10):
            consistency_errors.append(r["epoch"])
    weighted = {key: best["loss"][key] * weight for key, weight in
                (("vloss", 10), ("L1loss", .5), ("contrastloss", .005), ("proj_loss", .1))}
    summary = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "completed_epochs": len(rows), "final_global_step": rows[-1]["global_step"],
        "validation_cases_each_epoch": sorted({r["validation"]["cases"] for r in rows}),
        "start_records": starts, "missing_epochs": missing, "duplicate_epochs": duplicates,
        "malformed_json_lines": malformed,
        "step_mismatches": [r["epoch"] for r in rows if r["global_step"] != r["epoch"] * 140],
        "nonfinite_records": [{"epoch": r["epoch"], "paths": nonfinite(r)} for r in rows if nonfinite(r)],
        "error_lines": errors, "warnings": warnings, "best_field_mismatches": consistency_errors,
        "selection_rule": "maximum mean validation volume_psnr; equal weight over 10 scans",
        "best_epoch": best["epoch"], "best_record": best,
        "reference_epoch_98": reference,
        "epoch_98_to_best": {k: {"before": metric(reference, k), "after": metric(best, k),
                                 "delta": metric(best, k) - metric(reference, k)} for k in METRICS},
        "metric_extrema": extrema, "windows": windows, "best_updates": updates,
        "best_epoch_weighted_losses": weighted,
        "runtime": {
            "training_hours": sum(r["train_seconds"] for r in rows) / 3600,
            "validation_hours": sum(r["validation_seconds"] for r in rows) / 3600,
            "mean_epoch_minutes": statistics.mean(r["train_seconds"] + r["validation_seconds"] for r in rows) / 60,
            "excludes": "checkpoint saving, interruptions, queueing and other unlogged wall time",
        },
        "peak_allocated_gib": {"min": min(r["peak_allocated_gib"] for r in rows),
                               "max": max(r["peak_allocated_gib"] for r in rows)},
        "limitations": [
            "Validation summary only; the newly selected checkpoint has not been evaluated on the test set here.",
            "Per-case paired statistics and image quality cannot be recovered from this epoch summary log.",
            "axial_mip is a maximum projection, whereas OCTA-500 FULL uses a mean projection.",
            "Checkpoint identity is inferred from the saved-model policy and log; the server checkpoint was not loaded.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "epochs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    epochs = [r["epoch"] for r in rows]
    specs = [
        ("训练总损失 ↓", None, None),
        ("验证：体积 PSNR ↑", "volume_psnr", "dB"),
        ("验证：体积 SSIM ↑", "volume_ssim", None),
        ("验证：体积 MAE ↓", "volume_mae", None),
        ("验证：最大投影 PSNR ↑", "axial_mip_psnr", "dB"),
        ("验证：最大投影 SSIM ↑", "axial_mip_ssim", None),
    ]
    for ax, (title, name, unit) in zip(axes.flat, specs):
        values = [metric(r, name) if name else r["loss"]["total"] for r in rows]
        ax.plot(epochs, values, color="#8b9db0", alpha=.75, lw=.9, label="每轮记录")
        ax.plot(epochs[19:], [statistics.mean(values[i-19:i+1]) for i in range(19, len(values))],
                color="#1c6486", lw=2, label="最近20轮均值")
        ax.axvline(100.5, color="#2d8c84", ls="--", lw=1, alpha=.7)
        if name:
            ax.scatter([98], [metric(reference, name)], s=26, color="#b07a31", zorder=5, label="第98轮")
            ax.scatter([best["epoch"]], [metric(best, name)], s=36, color="#bd4d50", zorder=5, label="第193轮")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("训练轮次")
        if unit:
            ax.set_ylabel(unit)
        ax.grid(alpha=.18)
        ax.set_xlim(1, 201)
        if name == "volume_psnr":
            ax.annotate(f"第193轮：{metric(best, name):.4f} dB", xy=(193, metric(best, name)),
                        xytext=(105, 27.1), arrowprops={"arrowstyle": "->", "color": "#bd4d50"},
                        color="#923b3d", fontsize=10)
    axes[0, 1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("FlowOCT 3M：完整200轮训练日志\n每轮验证10例；虚线标记第100轮后的续训；投影曲线为最大投影", fontsize=15)
    fig.savefig(OUTPUT / "validation_trends_200e.png", dpi=160, facecolor="white")
    plt.close(fig)

    comparison_rows = []
    for e in (98, 193, 197, 200):
        r = by_epoch[e]
        comparison_rows.append(f"| {e} | {metric(r, 'volume_mae'):.5f} | {metric(r, 'volume_psnr'):.4f} | "
                               f"{metric(r, 'volume_ssim'):.4f} | {metric(r, 'axial_mip_psnr'):.4f} | "
                               f"{metric(r, 'axial_mip_ssim'):.4f} |")
    window_rows = []
    for item in windows:
        if item["epochs"][0] >= 81:
            v = item["validation"]
            window_rows.append(f"| {item['epochs'][0]}–{item['epochs'][1]} | {v['volume_psnr']:.4f} | "
                               f"{v['volume_mae']:.5f} | {v['volume_ssim']:.4f} | {v['axial_mip_ssim']:.4f} |")
    psnr_gain = metric(best, "volume_psnr") - metric(reference, "volume_psnr")
    mae_reduction = (1 - metric(best, "volume_mae") / metric(reference, "volume_mae")) * 100
    hours = summary["runtime"]["training_hours"] + summary["runtime"]["validation_hours"]
    report = f"""# FlowOCT 3M：200轮日志分析

分析日期：2026-10-24。来源：`{SOURCE}`，SHA-256：`{summary['source_sha256']}`。

本次读取下载日志及本地训练/评价实现，生成分析图和报告；没有运行训练或重写模型权重。

## 完成情况与选模

日志包含连续、无重复的1–200轮完整训练与验证记录，最终global_step=28000；每轮验证10例。
启动记录从第1轮开始，中途从latest.pt恢复到第101轮。轮次和global_step连续；这能说明记录衔接正常，不能单凭日志证明续训数值与不中断运行逐位相同。

按既定的验证集平均体积PSNR选模，第{best['epoch']}轮最好，为{metric(best, 'volume_psnr'):.6f} dB。
按当前runner保存规则，训练目录中的best.pt应对应第193轮，latest.pt和epoch_0200.pt应对应第200轮。这里未加载服务器checkpoint，正式测试前应核对checkpoint的epoch字段。

以下均为同一日志中的验证集均值，不能和此前50例测试集结果混为同一组数据。

| 轮次 | 体积MAE ↓ | 体积PSNR ↑ / dB | 体积SSIM ↑ | 最大投影PSNR ↑ / dB | 最大投影SSIM ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(comparison_rows)}

第193轮相较第98轮：体积PSNR提高{psnr_gain:.4f} dB，体积MAE下降{mae_reduction:.2f}%，体积SSIM提高{metric(best, 'volume_ssim') - metric(reference, 'volume_ssim'):.4f}。
最大投影SSIM仅提高{metric(best, 'axial_mip_ssim') - metric(reference, 'axial_mip_ssim'):.4f}，说明投影结构指标的收益小于体积指标收益；是否恢复更多细血管还需相同病例的图像和逐例指标核对。

## 后100轮的收益与波动

以下对每轮10例验证均值再做阶段平均，用于观察趋势；它不是独立重复实验，也不能据此计算跨病例显著性。

| 轮次范围 | 体积PSNR / dB | 体积MAE | 体积SSIM | 最大投影SSIM |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(window_rows)}

后期训练损失仍下降，验证阶段平均总体改善，没有出现持续的整体恶化。个别轮次的SSIM波动较大：第197轮为0.8672，第200轮为0.7932。
本地实现对验证数据采用固定中心裁剪、关闭翻转，并从OCT开始执行固定三步Euler推理；没有随机噪声初值。因此不能把这些轮间变化解释为随机验证裁剪；每轮权重变化及病例响应可能参与其中，但该日志不包含逐例预测，不能唯一归因。
第197轮体积SSIM最高，第191轮体积MAE最低，第199轮最大投影PSNR最高，第200轮最大投影SSIM最高。不同指标的最优轮次不一致，正式结果应继续采用事先约定的体积PSNR选模规则，避免事后挑指标换模型。

## 损失、显存与异常

第193轮训练平均总损失为{best['loss']['total']:.6f}。四项损失乘权重后的贡献分别为：速度项{weighted['vloss']:.6f}、L1项{weighted['L1loss']:.6f}、对比项{weighted['contrastloss']:.6f}、投影项{weighted['proj_loss']:.6f}。
这些项的尺度和定义不同，不能用原始数值大小直接判断哪项学得好。

全部已解析数值均有限，没有记录Traceback、显存不足或CUDA错误；best_validation_psnr与逐轮累计最大值一致。
日志有两条相同的TorchIO兼容性警告，分别出现在首次启动与续训启动。当前本地加载器将数据转为张量字典并使用自定义collate；这条通用警告本身不能证明批次错误，日志也不足以独立核验所有输入内容。

峰值allocated显存范围{summary['peak_allocated_gib']['min']:.4f}–{summary['peak_allocated_gib']['max']:.4f} GiB，没有随轮次持续增大的迹象；该数值不是nvidia-smi整卡占用，也不是reserved显存。
日志开头记录A100 80GB、network_chunk=0、每病例64张切片。

记录的训练耗时{summary['runtime']['training_hours']:.2f}小时，验证耗时{summary['runtime']['validation_hours']:.2f}小时，合计{hours:.2f}小时，平均每轮{summary['runtime']['mean_epoch_minutes']:.2f}分钟。
这些计时不含模型保存、中断、排队和其他未记录开销，因此不能等同于从启动到结束的自然时间。

## 后续评价

200轮预算已完成，建议将此结果作为本次预定训练的终点，用更新后的best.pt（按日志应为第193轮）在原50例测试集上重新推理，保留此前第98轮输出，进行逐病例配对比较。
仍可保留epoch_0200.pt作为最后一轮的补充结果，但不要用测试集指标决定主模型。

日志里的axial_mip是沿深度的最大投影，不是OCTA-500 FULL平均投影。本次日志没有新best模型的FULL平均投影、测试集评价或图像，不能声称这些测试结果已经改善，也不能仅据该日志认定达到论文完整复现水平。

![200轮趋势](validation_trends_200e.png)
"""
    (OUTPUT / "FlowOCT_200轮日志分析.md").write_text(report, encoding="utf-8")
    print(json.dumps({"completed_epochs": len(rows), "best_epoch": best["epoch"],
                      "best_volume_psnr": metric(best, "volume_psnr"),
                      "report": str(OUTPUT / "FlowOCT_200轮日志分析.md"),
                      "figure": str(OUTPUT / "validation_trends_200e.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
