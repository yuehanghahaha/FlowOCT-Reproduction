# 在服务器导出FULL平均投影和轻量分析包

原测试流程只保存三维NPZ，并计算体积和轴向最大投影指标，没有自动保存PNG。本工具读取已完成的测试结果，生成50例FULL平均投影、中心B-scan对照图、二维浮点数据和指标包。

它只使用CPU及磁盘读取，不加载训练模型、不创建CUDA上下文、不更改权重或训练配置。可以在400轮续训继续运行时，从同一计算环境的另一个终端或tmux窗口启动。

## 上传和运行

将本地文件D:/data/code/keyan_fuxian/FlowOCT/FlowOCT_export_FULL_review.py上传到服务器现有flowoct_work/uploads目录。

在tmux里可用Ctrl+B、松手、按c新建窗口，然后执行：

```bash
cd /workspace/flowoct_work/flowoct_reproduction_a100
/workspace/.conda/envs/lyh_flowoct/bin/python /workspace/flowoct_work/uploads/FlowOCT_export_FULL_review.py \
  --predictions results/flowoct_3m_best_after200 \
  --evaluation evaluations/flowoct_3m_best_after200 \
  --output /workspace/flowoct_work/uploads/flowoct_3m_best_after200_FULL_review
```

这条命令读取已有预测和评价结果，不重新推理。每处理一例会打印FULL进度，结束后打印案例数、checkpoint轮次、包大小和FULL指标。

已有同名输出目录或ZIP时会停止，保留已有文件。另一次导出可换一个新output名称。

## 下载

完成后下载uploads里的：

- flowoct_3m_best_after200_FULL_review.zip
- flowoct_3m_best_after200_FULL_review.zip.sha256

保存到D:/data/FSdata。新的包用于投影和选定切片的分析；完整三维体积复核仍使用服务器上原来的NPZ。

## 包内文件

- figures/：每例一张六宫格PNG，包含真实FULL、预测FULL、FULL误差、真实中心B-scan、预测中心B-scan及其误差。
- projections/：每例一个轻量NPZ，保存未显示拉伸的float32二维gt_full、pred_full，以及XY轴说明。
- full_metrics.json：FULL的逐病例、总体及疾病分组MAE、PSNR、SSIM，另有显示窗和几何信息。
- prediction.json、summary.json、per_case.csv：从原测试结果原样复制的记录，保留原体积/最大投影指标。

不包含原三维NPZ和模型权重。2D数组未压缩时，两张256x256 float32图每例合计0.5MiB，50例合计25MiB；再加PNG和记录，通常是几十MB。实际大小由程序打印。

## 计算口径

FULL = 对保存的ZYX体积沿Z轴做算术平均，不是最大值。YX投影再转置为XY，与前次已核对的官方BMP显示方向一致。

这是预处理后中心体块的全保存深度投影，不会恢复原始完整横向视野或640层分辨率，也不是直接使用官方显示BMP作为指标真值。

GT和预测使用同一个GT min到P99显示窗；指标在显示拉伸之前的归一化[0,1]数据上计算，SSIM窗口为7。二维浮点数组保留，便于下载后复算。误差图有独立色标，中心B-scan只是一个切片示例，不能代替整卷检查。

## 本地验证

工具已用本地既有第98轮10305病例检查：均值数组和XY方向正确，结果区别于最大投影，FULL指标与前次核验结果一致，原三维文件没有改变，重复输出会停止，ZIP与校验值正确。单病例原三维NPZ约92.83MiB，导出的轻量ZIP约1.01MiB。

这只是导出工具验证，不是服务器第193轮新测试结果。新结果需要在服务器执行上面的命令生成。

