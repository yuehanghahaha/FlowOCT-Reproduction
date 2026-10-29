# FlowOCT 固定裁剪单病例拟合实验

版本日期：2026-10-26。按你的选择，**重新只训练一个固定病例**，比较这次新实验第70、100、200、300轮和实际最佳轮。不会调用原来多病例训练的这些轮次权重。

## 1. 实验要回答什么

检查当前 FlowOCT 代码在反复学习同一个 OCT/OCTA 配对、相同裁剪区域时，能否把这个病例的结构和细节生成清楚。这样可以进一步区分“多病例学习难度”和“连单病例也未充分拟合、或训练路径与完整生成有差距”。

这是一项训练拟合诊断，没有独立验证/测试成绩。400轮未拟合清楚，也不能单凭这一点就认定代码错误或模型容量不足。

## 2. 固定设置

| 项目 | 设置及理由 |
|---|---|
| 病例 | 原训练集 NORMAL 病例 **10308**，沿用上次固定10例中的病例，不根据这次实验结果挑选 |
| 初始化 | **从零随机初始化**，seed=2026；不加载旧模型或旧优化器 |
| 数据 | 现有 `data/OCTA500_v1/split_3M_z256.csv` 对应数据，无需 native 文件夹 |
| 预处理 | 沿用现有 ZYX→YXZ 适配，先400×400×256，再固定中心256³；起点YXZ=[72,72,0] |
| 归一化 | 沿用裁剪内 OCT、OCTA 分别 min-max 到[0,1] |
| 数据增强 | 关闭随机裁剪和随机翻转，配对区域始终相同；加载一次后反复使用 |
| 模型 | 当前2D DiffusionModelUNet与2D Haar，结构不变 |
| 每次训练 | 仍随机采样64张Y方向B-scan，随机采样t；**固定裁剪不意味着固定64张切片或固定t** |
| 损失 | 原代码：10×velocity + 0.5×L1 + 0.005×contrast + 0.1×projection；ROI规则不变 |
| 优化器 | Adam，lr=0.0002，betas=(0.5,0.999)，无调度器 |
| 显存执行 | FP32，`network_chunk=0`，64张一次过网络，无AMP、无激活重计算 |
| 总训练预算 | 默认400个单病例轮次，**每轮1次Adam更新** |
| 每轮评估 | 只输入OCT，生成完整256³，3步Euler、每步clamp[0,1]，推理分组32 |
| 必存阶段 | 第70、100、200、300轮；另存最终400轮、每轮latest及历史最佳 |
| 最佳拟合 | 每轮完整生成后，以**同一训练病例完整体积PSNR最高**更新 `best_fit.pt`，相等不替换 |

病例数量减少，不会降低单次64张切片前反向传播的显存要求。此方案保持原执行方式，使用你的A100。没有在本地6GB显卡上假装完成256³/64张的正式训练。

**轮数的含义：**原实验每轮140病例、140次更新；本实验每轮1病例、1次更新。因此本实验100轮=100次更新，原实验100轮=14,000次更新。两者每个病例在各自100轮中都曝光100次，但总训练量不同，不能把两组的“100轮”直接当成相同训练预算。日志同时保存轮数、更新次数、病例曝光次数。

400轮是首次诊断预算，不能保证一定拟合到近乎完美。若仍明显改善，可按第8节延长；不要同时修改损失、采样量和积分步数，否则无法判断变化来自哪里。

## 3. 先核对原代码如何选最佳

作者公开代码 `reference/upstream/util/train_utils.py`：逐例计算三维PSNR，再计算 `avg_psnr = psnr_total / num`，仅当 `avg_psnr > global_best_psnr` 时保存最佳。关键位置为该文件第53、61、73—76行。

当前复现代码 `flowoct/runner.py` 第240—249行：

```python
psnr = row["validation"]["metrics"]["volume_psnr"]["mean"]
improved = psnr > best_psnr
best_psnr = max(best_psnr, psnr)
# 先保存本轮latest；若improved，将本轮latest复制为best.pt
```

你原实验为10例验证集，每例等权，选三维PSNR的平均值最高者。**不是按训练loss最低、SSIM最高、FULL投影最好看或固定140轮来选。**每例先算PSNR再平均，不是把10例MSE平均后换算一次PSNR。

当前指标代码对完整归一化体积计算 `PSNR = -10 log10(MSE)`，`data_range=1`，没有额外的前景掩膜。作者函数使用该例GT最大值减最小值；在目前逐例min-max到[0,1]的非恒定体积上，这个跨度也是1。论文其它掩膜、层投影与公开代码的差异，并没有因本实验而解决。

PSNR考察整体像素误差，不直接按照细血管清晰度排序。因此 `best.pt` 可以是验证体积PSNR最高，却不是投影细节最好的轮次。固定轮次检查点里的 `best_psnr` 字段表示**截至那轮的历史最佳验证值**，不一定是该检查点自身那一轮的验证值。

本实验无独立验证集，所以另外命名为 `best_fit.pt`，按10308这个训练裁剪的完整生成PSNR选择；不能称为“验证集最佳”。代码中每轮均评估，没有只在70/100/200/300这几轮内挑最佳。

## 4. 上传和解压

把本地 `FlowOCT_singlecase_tools.zip` 上传到：

```text
/workspace/flowoct_work/uploads/FlowOCT_singlecase_tools.zip
```

在服务器终端执行。使用现有训练环境的绝对Python路径，不要求你先正确切换conda，也不依赖unzip：

```bash
/workspace/.conda/envs/lyh_flowoct/bin/python -m zipfile -e /workspace/flowoct_work/uploads/FlowOCT_singlecase_tools.zip /workspace/flowoct_work/uploads/
```

解压后应为 `uploads/FlowOCT_singlecase_tools/`。此工具需要服务器上原有 `flowoct_reproduction_a100` 和训练环境；压缩包没有重复携带模型核心代码、数据集或旧权重。

## 5. 先做检查

```bash
bash /workspace/flowoct_work/uploads/FlowOCT_singlecase_tools/FlowOCT_singlecase.sh check
```

检查内容：原代码8个核心文件SHA、软件版本、数据清单SHA、病例确属训练集NORMAL、完整预处理后裁剪位置和OCT/GT哈希。出现 `"check": "passed"` 才继续。

检查仅使用CPU，不训练、不创建实验目录、不代表已经验证GPU显存。若源代码或环境被改过，工具会明确停止，不会自动放宽设置或改小训练切片数。

## 6. 后台启动正式实验

确认A100有足够空闲显存；同一卡上旧训练还在运行时，先等待它结束或按原训练的操作保存暂停。

```bash
mkdir -p /workspace/flowoct_work/logs /workspace/flowoct_work/experiments
nohup bash /workspace/flowoct_work/uploads/FlowOCT_singlecase_tools/FlowOCT_singlecase.sh fit --output /workspace/flowoct_work/experiments/flowoct_singlecase10308_fixed_400e > /workspace/flowoct_work/logs/flowoct_singlecase10308_400e.log 2>&1 < /dev/null &
```

这条命令从随机权重开始训练400轮。实验目录已存在会停止，防止覆盖已有结果。原来的200/400轮训练目录、最佳模型、数据清单均不写入。

查看进度：

```bash
tail -n 30 -f /workspace/flowoct_work/logs/flowoct_singlecase10308_400e.log
```

例如 `FIT epoch 100/400 step=100 ... best_epoch=97`：已经完成100次单病例更新，至今同病例PSNR最好的是97轮。`Ctrl+C` 只退出这个日志查看，不会停止后台训练。

耗时取决于A100负载。每轮包含一次64张训练和一次完整体积生成、指标计算；不预先承诺分钟数。初期也会先输出第0轮随机模型结果，供检查输入和显示范围。

## 7. 结果内容与下载

所有检查点留在服务器独立目录：

```text
/workspace/flowoct_work/experiments/flowoct_singlecase10308_fixed_400e/
  checkpoints/
    latest.pt
    best_fit.pt
    epoch_0070.pt
    epoch_0100.pt
    epoch_0200.pt
    epoch_0300.pt
    epoch_0400.pt
  review/
    projections_comparison.png
    central_slices_comparison.png
    best_fit_training_path_probe.png
    training_curves.png
    comparison_metrics.csv
    training_history.csv
    checkpoint_metadata.json
    summary.json
    ...原始二维图、配置、输入哈希与记录
  status.json
  run.log
```

训练完成后，日志打印 `DOWNLOAD THIS ZIP:`。按本说明指定目录运行时，下载：

```text
/workspace/flowoct_work/experiments/flowoct_singlecase10308_fixed_400e_epoch400_review.zip
```

这个ZIP含完整评判需要的二维原始浮点图、PNG、指标、训练记录和参数，不含 `.pt` 或完整三维体积，避免再次下载几个GB。压缩率随内容变化，不承诺固定大小；实际大小和SHA256会打印在日志中。下载解压后交给我分析即可。

五组权重并排显示GT/70/100/200/300/实际最佳；若最佳恰好是这些阶段之一，重复列是正常的，不另找一个较差轮次凑数。第400轮保留独立权重、二维图和JSON，即使未出现在五组对照中，也可以继续追查。

FULL是沿**物理Z深度求平均**；Z最大投影另外展示。每例各阶段使用同一个GT亮度窗口，局部位置固定，不分别拉伸预测图，不锐化、去噪或更换投影范围。覆盖为中心256裁剪，约原3mm视野中央1.92mm，不等于原生全3mm和640层数据。

## 8. 暂停、续训和延长

每完成1轮就原子写入 `latest.pt`，包括网络、对比头、Adam状态、Python/NumPy/Torch/CUDA随机状态和历史记录。中断最多丢失当前未提交的一轮；至少完成第1轮并生成latest后才可续训。

如果必须停止，先找**这个单病例脚本**的进程：

```bash
pgrep -af '[F]lowOCT_singlecase.py'
```

确认列出的PID属于这个实验，再执行 `kill -TERM 对应PID`。不要终止原多病例训练进程。已提交检查点继续保留；没有必要使用 `kill -9`。

进程停止后，从这个实验的latest继续到总400轮：

```bash
nohup bash /workspace/flowoct_work/uploads/FlowOCT_singlecase_tools/FlowOCT_singlecase.sh resume --output /workspace/flowoct_work/experiments/flowoct_singlecase10308_fixed_400e --epochs 400 >> /workspace/flowoct_work/logs/flowoct_singlecase10308_400e.log 2>&1 < /dev/null &
```

`--epochs` 是目标总轮数，不是再加多少轮。不要同时启动两份进程续训同一实验。

若400轮仍在明显改善，沿用相同设置继续到总1000轮，把上条的 `--epochs 400` 改为 `--epochs 1000`。不重新初始化、不清空优化器；最佳拟合仍在整个已完成历史中选择。新结果包以实际完成轮数命名，400轮包保留。

## 9. 如何判断结果

优先查看训练曲线、FULL/局部、正交切片，而不是仅看一个SSIM数值。

| 观察 | 可以支持的判断 | 仍不能证明的事 |
|---|---|---|
| 单病例越训越清楚，原多病例仍模糊 | 代码至少具备拟合这一病例固定区域的能力；后续查多病例学习和泛化 | 不能证明论文已复现，也不能证明其它病例都能拟合 |
| 含GT的训练路径误差小，但OCT-only完整生成仍模糊 | 训练路径与自由生成有差距，值得查速度场、积分轨迹和目标 | t=0.9的输入本来就含90%GT，不能把它当成生成能力 |
| t=0单步较好，3步生成较差 | 后续积分后的状态可能偏离训练轨迹，值得做专门检查 | 不能直接断言“3步一定错”，此次保持原3步 |
| 连同病例400次更新也不清楚 | 需要结合收敛趋势、加权损失和路径探针进一步查 | 不能直接归因于参数少、二维结构、显存或某一个损失 |

`best_fit_training_path_probe.png` 使用固定8张Y切片，在t=0、0.5、0.9检查速度预测和训练式端点。t>0的输入为 `(1-t)*OCT+t*GT`，明确含GT，**不参与最佳模型选择**。t=0不含GT，等价于检查一次跨越全时段的单步预测；正式结果始终为只输入OCT的3步生成。

指标包括完整体积和投影MAE/PSNR/SSIM，另有GT跨度SSIM和高低频相关度。后两项是细节敏感性诊断，不是论文结果或血管分割准确率；高频也包含噪声。

## 10. 这次添加或改变了什么

**不修改服务器原核心文件。**新增一个独立训练外层和启动脚本，读取并校验原核心版本，再调用原 `model.forward()` 完成前向、反向及Adam更新。

| 项目 | 相对原多病例实验的变化 | 理由 |
|---|---|---|
| 数据范围 | 140训练病例→10308单病例 | 检查最基本的记忆/拟合能力 |
| 裁剪/翻转 | 随机→固定中心/无翻转，缓存同一配对 | 保证反复学的是同一区域，不混入位置变化 |
| 训练轮定义 | 每轮140步→每轮1步 | 明确记录单病例更新预算 |
| 选择规则 | 验证10例平均PSNR→该训练裁剪PSNR | 没有独立验证；用best_fit名称避免混淆 |
| 训练记录 | 增加各项加权损失、两阶段峰值显存、输入哈希、训练路径探针 | 判断是否在学习，以及哪里出现差距 |
| 保存方式 | 独立完整检查点，支持续训；自动二维轻量包 | 保护原实验并方便审阅 |

未改网络、波段、损失、ROI、随机64切片和t规则、学习率、优化器、FP32、`network_chunk=0`或3步生成。现有轴适配与公开代码/论文之间未解决的差异也原样保留，避免把“修复方案”混入这次诊断。

`expected_baseline.json` 只是原始数据清单/核心代码/环境的校验指纹；`VERIFICATION_本地核验.md` 说明正式运行前已核验了哪些内容和本地显存限制。
