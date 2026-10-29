# 单病例拟合检查结果

病例：10308，原划分为训练集 NORMAL。全新随机初始化，固定中心256立方裁剪，无翻转。
已完成 400 轮，即 400 次优化器更新。
单病例最佳拟合轮：394，该训练病例三维 PSNR=28.915187。

- projections_comparison.png：GT / 70 / 100 / 200 / 300 / 实际最佳，FULL、固定局部和Z最大投影。
- central_slices_comparison.png：相同轮次的三个方向中心切片。
- training_curves.png、training_history.csv：每轮损失、完整生成指标、更新次数和峰值显存。
- best_fit_training_path_probe.png：区分只输入OCT生成与含GT的训练路径探针。
- comparison_metrics.csv、各阶段.json/.npz：指标、原始二维浮点图、训练路径记录。

每轮训练后仅输入 OCT，进行完整256立方生成，再与GT计算三维PSNR。严格超过历史最佳才更新best_fit.pt。
含GT的t=0.5/0.9探针有助于检查训练路径与自由生成的差距；它们不是模型只输入OCT生成的结果，也不参与选优。
GT范围SSIM和高通相关度用于观察细节，不等于论文指标或血管分割精度。
1轮=1病例=1次更新；原140病例实验的1轮=140次更新，两者不能直接按轮数比较训练总量。
本实验没有独立验证/测试集，单病例PSNR不能当作模型泛化成绩。
检查点留在上级checkpoints/，本压缩包只含二维图和记录。
