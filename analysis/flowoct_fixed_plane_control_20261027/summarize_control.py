"""Scientific plot and bounded interpretation of the local fixed-plane control."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
res=json.loads((HERE/'result.json').read_text(encoding='utf-8'))
rows={r['step']:r for r in res['records']}
stages=[400,800,2000]
data={}
for n in stages:
    with np.load(HERE/f'step{n:04d}.npz') as z:data[n]={k:z[k].copy() for k in z.files}
gt=data[400]['gt'];assert all(np.array_equal(gt,data[n]['gt']) for n in stages)
font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
peak=int(np.argmax(gt.mean(axis=1)));y=max(0,min(192,peak-32));x=96
fig,axes=plt.subplots(2,4,figsize=(12,6.6),constrained_layout=True)
arrays=[gt]+[data[n]['prediction_clipped'] for n in stages]
titles=['GT']+[f"{n}次更新\nB-scan SSIM={rows[n]['clipped_ssim_range1']:.3f}" for n in stages]
for i in range(2):
    for j,a in enumerate(arrays):
        if i:a=a[y:y+64,x:x+64]
        axes[i,j].imshow(a,cmap='gray',vmin=float(gt.min()),vmax=float(np.percentile(gt,99)),interpolation='nearest')
        axes[i,j].axis('off')
        if i==0:axes[i,j].set_title(titles[j],fontsize=10)
fig.suptitle('本地简化对照：10308固定单张B-scan，原FlowOCT主干，t=0，仅全图速度MSE\n上：整张切片；下：按GT预先定位的固定组织局部。不是完整FlowOCT训练或三维结果。',fontsize=11)
fig.savefig(HERE/'固定单切片_400_800_2000步对照.png',dpi=160);plt.close(fig)
body='''# FlowOCT固定单切片对照：对“400轮应该学会”的补充核验

日期：2026-10-27。此项使用本地真实病例10308、固定Y=128的完整256×256 B-scan。

沿用原2D FlowOCT主干、Haar和euler_step，不改核心文件。重新随机初始化，Adam lr=0.0002、betas=(0.5,0.999)，FP32。
本项固定t=0，只最小化10*MSE(A+v_hat(A,0),B)，等价于全图速度MSE。训练损失前不clamp；显示及SSIM使用clamp到[0,1]后的结果。
不含随机抽片、随机t、ROI、L1、对比或投影项，不执行正式3步轨迹。它检查固定映射的拟合能力，不是正式复现分数，也不是对全体积任务的等预算消融。

| 更新次数 | 未clamp全图MSE | 未clamp全图MAE | B-scan SSIM（range1） | GT>0.05位置MAE |
|---|---:|---:|---:|---:|
'''
for n in [0,400,800,1200,1600,2000]:
    r=rows[n];body+=f"| {n} | {r['raw_mse']:.8f} | {r['raw_mae']:.6f} | {r['clipped_ssim_range1']:.4f} | {r['bright_gt_raw_mae']:.6f} |\n"
body+='''
GT>0.05仅为亮信号位置的诊断，不是血管分割掩膜。所有阶段使用同一GT和同一显示窗口；图中局部按GT行均值峰值定位，不按预测挑选。

## 能够支持什么

400次更新时，即便是单张固定切片、固定t和单一目标，也尚未达到高精度拟合；2000步仍能明显改善。
400至2000步，全图MSE降低约91.1%，亮GT区域的MAE降低约76.0%。网络参数确实更新，梯度有限且非零，2000次Adam更新已核对。
这反对“FlowOCT主干完全没有固定样本拟合能力”的说法，也表明不能把400步当成必定充分训练的门槛。

## 不能支持什么

SSIM约0.94不等于逐像素完美记忆，不证明整个256立方体或完整FlowOCT目标已经可以拟合；也不证明原实验模糊只由更新次数造成。
当前对照与原实验在批次、t分布、辅助目标、输出生成方式及硬件上均有区别，不能用两组分数差直接归因。
本地einops0.8.2与服务器0.8.1不同，已记录；原始输入体素哈希与服务器10308固定裁剪完全一致。

更准确的后续判断应是：主干可以继续学习固定配对；完整流程的更新预算、时间路径和辅助目标各自影响，需要继续受控核查。
本次未更改服务器代码、正式复现代码或已有权重。
'''
(HERE/'固定单切片对照_结果说明.md').write_text(body,encoding='utf-8')
print(json.dumps({'stages':{n:rows[n] for n in stages},'fixed_display_roi':[x,y,64,64],
 'mse_reduction400to2000':1-rows[2000]['raw_mse']/rows[400]['raw_mse'],
 'bright_mae_reduction400to2000':1-rows[2000]['bright_gt_raw_mae']/rows[400]['bright_gt_raw_mae']},ensure_ascii=False,indent=2))
