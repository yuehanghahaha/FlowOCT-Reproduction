"""Show the same type of central orthogonal slices described in paper Figure3."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from skimage.metrics import structural_similarity

HERE = Path(__file__).resolve().parent
SOURCE = Path(r"D:\data\FSdata\results\flowoct_3m_best_after200")
font = Path(r"C:\Windows\Fonts\msyh.ttc")
font_manager.fontManager.addfont(str(font))
plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams["axes.unicode_minus"] = False
all_records = []
for case in ("10305", "10331"):
    with np.load(SOURCE / f"{case}.npz", allow_pickle=False) as data:
        assert data["axis_order"].item() == "ZYX"
        gt, pred = data["octa_gt"], data["octa"]
    views = [
        ("中心Y切片：Z×X", gt[:, 128, :], pred[:, 128, :]),
        ("中心Z切片：X×Y", gt[128].T, pred[128].T),
        ("中心X切片：Z×Y", gt[:, :, 128], pred[:, :, 128]),
        ("FULL均值：X×Y", gt.mean(axis=0, dtype=np.float64).T, pred.mean(axis=0, dtype=np.float64).T),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(9.8, 12.6), constrained_layout=True)
    for row, (title, real, fake) in enumerate(views):
        low, high = float(real.min()), float(np.percentile(real, 99))
        high = max(high, low+1e-6)
        mse = float(np.mean((fake.astype(np.float64)-real)**2))
        values = {"case_id":case, "view":title,
                  "psnr_range1":float(-10*np.log10(mse)),
                  "ssim_range1":float(structural_similarity(real,fake,data_range=1,win_size=7)),
                  "ssim_gt_observed_range":float(structural_similarity(real,fake,data_range=float(np.ptp(real)),win_size=7)),
                  "common_display_min_p99":[low,high]}
        all_records.append(values)
        for col, (array, name) in enumerate(((real,"真实OCTA"),(fake,"第193轮预测"),(np.abs(fake-real),"绝对误差"))):
            axes[row,col].imshow(array, cmap="gray" if col<2 else "magma", vmin=low if col<2 else 0,
                                 vmax=high if col<2 else high-low, interpolation="nearest")
            axes[row,col].axis("off")
            axes[row,col].set_title(title+"\n"+name,fontsize=9)
        axes[row,1].set_title(title+"\n第193轮预测；SSIM(range=1)="+f"{values['ssim_range1']:.3f}",fontsize=9)
    fig.suptitle(case+"：中心切片与FULL是不同的评价对象\n每行真实/预测共用灰度窗；中心索引均128；与论文病例及显示细节并不完全相同",fontsize=11)
    fig.savefig(HERE / f"{case}_三方向中心切片与FULL.png",dpi=150,facecolor="white")
    plt.close(fig)
    del gt,pred,views
(HERE / "central_view_metrics.json").write_text(json.dumps({"records":all_records,
    "limits":"Fixed prior-review cases; no matching paper case ID, no claim of identical paper slice/display protocol; raw images unchanged"},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(all_records,ensure_ascii=False,indent=2))
