# FlowOCT：训练10例、验证10例，70/140/当前最优轮投影比较

本工具使用你服务器现有的FlowOCT代码和权重，不重新训练，不修改原工程。三份权重均在启动时冻结；当前最优来自400轮续训目录的best.pt，实际轮次以读取结果为准，不假定一定是313。

先完成固定10例训练病例的三组推理，再完成原有10例验证病例的三组推理，共60次病例推理。每次生成完整准备体块后只保留FULL平均和物理Z最大值两种二维投影；不保存三维预测文件，不把权重放入下载包。

## 上传和解压

把FlowOCT_trainval_projection_tools.zip上传到：

    /workspace/flowoct_work/uploads/

在服务器终端执行：

```bash
unzip -n /workspace/flowoct_work/uploads/FlowOCT_trainval_projection_tools.zip -d /workspace/flowoct_work/uploads
```

工具文件夹为/workspace/flowoct_work/uploads/FlowOCT_trainval_projection_tools。不要覆盖或替换flowoct_reproduction_a100中的文件。

## 第一步：先检查文件与环境

```bash
bash /workspace/flowoct_work/uploads/FlowOCT_trainval_projection_tools/FlowOCT_trainval_projection.sh --check-only
```

该命令只进行CPU检查，不启动推理、不更新训练、不生成大文件。检查内容包括依赖、原始manifest、病例归属、训练时核心源码指纹、三份checkpoint的轮次和配置。

必须看到check: passed，并看到epoch70对应70、epoch140对应140，以及当前best的真实轮次。如果缺少epoch_0070.pt或epoch_0140.pt，请保留完整错误信息；日志不能代替模型权重。

默认检查的权重是：

- /workspace/flowoct_work/flowoct_reproduction_a100/runs/flowoct_3m_200e/epoch_0070.pt
- /workspace/flowoct_work/flowoct_reproduction_a100/runs/flowoct_3m_200e/epoch_0140.pt
- /workspace/flowoct_work/flowoct_reproduction_a100/runs/flowoct_3m_400e_from200/best.pt

若原200轮目录中缺少70或140轮快照，会再检查续训目录中随原run复制过来的同名快照，仍严格核验轮次、配置和manifest。两个目录都缺少时才停止，不用邻近轮次替代。

脚本固定使用/workspace/.conda/envs/lyh_flowoct/bin/python，因此不依赖当前是否激活了base、lyh_tmux或lyh_flowoct。若提示缺少matplotlib，只对该环境补装：

```bash
/workspace/.conda/envs/lyh_flowoct/bin/python -m pip install matplotlib==3.10.1
```

其他检查失败先查看具体原因，不重新创建数据划分或改动模型。

## 第二步：后台执行完整60次推理

检查通过后执行：

```bash
mkdir -p /workspace/flowoct_work/logs
nohup bash /workspace/flowoct_work/uploads/FlowOCT_trainval_projection_tools/FlowOCT_trainval_projection.sh >> /workspace/flowoct_work/logs/flowoct_trainval_projection_launcher.log 2>&1 < /dev/null &
```

查看进度：

```bash
tail -n 30 -f /workspace/flowoct_work/logs/flowoct_trainval_projection_launcher.log
```

Ctrl+C只退出这个tail查看命令，后台任务继续。日志会依次显示train和validation、病例ID、使用的权重和1/60至60/60的进度。不要重复启动相同任务。

也可以在已经安装好的tmux内直接运行同一个bash命令，不使用nohup：

```bash
bash /workspace/flowoct_work/uploads/FlowOCT_trainval_projection_tools/FlowOCT_trainval_projection.sh
```

tmux脱离操作是Ctrl+B后松开，再单独按字母d。两种启动方式选一种即可。

本工具会占用GPU；与训练并行会竞争资源、可能降低速度。它不会自动结束或暂停原训练。文件冻结只锁定本次比较使用的权重，原best.pt仍可由训练正常更新。

## 第三步：下载结果

完成后，启动日志末尾出现：

    DOWNLOAD THIS ZIP: /workspace/flowoct_work/uploads/flowoct_trainval_70_140_best_时间戳_review.zip

下载这一份_review.zip及其.sha256文件。ZIP内含：

- figures：20张病例并列PNG；每张四列为GT、70轮、140轮、最优轮；三行为FULL、固定局部放大、Z最大投影。
- train_FULL_overview.png、train_Z_MAX_overview.png以及对应两张validation总览。
- projections：20个二维NPZ，含对应GT和三组预测的FULL与最大投影，不包含完整三维体积。
- metrics_per_case.csv：60次病例推理×两种投影，共120条投影指标。
- paired_changes.csv、summary.json：各病例阶段变化、均值和改善病例数。
- case_list.csv、checkpoint_metadata.json、protocol.json、case_preprocessing_proof.json、environment.json、run.log：病例归属、冻结轮次、哈希、设置及过程记录。

ZIP预计几十MB，最终大小会打印在日志中。旁边没有_review后缀的同名目录含冻结权重和过程文件，留在服务器即可。

## 训练病例如何保证公平

训练病例在未查看模型结果前，使用seed2026按类别固定选择7例NORMAL、DR/AMD/CNV各1例，并确保影像组不重复：

    10308 10310 10354 10373 10386 10413 10426 10463 10474 10481

原有10例验证病例不变：

    10302 10319 10338 10357 10396 10422 10447 10456 10495 10500

训练病例仍标记为train。工具先用原加载器按真实归属筛选，再把这个独立加载器实例的预处理模式设为validation，关闭随机裁剪和翻转。它不会修改manifest，也不会把训练病例加入原验证集。

所有病例均只输入OCT，从初始状态完成三步生成；不会使用混有真实OCTA的训练中间重建图。GT只在生成后用于投影对照。原始模型、Haar、损失、轴假设和预处理函数源码均不改。

显示采用每例GT min至P99共同灰度窗、最近邻方式，不单独增强预测。局部坐标固定为XY显示图x=144、y=40、边长64。

指标保留MAE、range=1的PSNR/SSIM，同时增加按GT实际投影跨度的PSNR/SSIM敏感性诊断。该诊断不冒充论文foreground或三维体积指标；本次属于训练拟合与验证表现检查。

## 可选操作

默认一条命令完成训练和验证。若只想先做训练10例，可加--splits train；只做验证可加--splits validation。每次运行都会另建输出目录、重新冻结当时的best；分两次运行时可能选中不同最优轮。需要严格使用同一最优权重比较训练和验证时，使用默认的完整一次运行。

CLI支持--project指定项目路径、--device指定分配到的设备、--output指定新的输出目录。现有服务器按默认值即可，不需要手动编辑代码。

本地验证范围及记录由交付说明另行报告；本工具没有替你连接A100，也没有在A100上实际运行这60次推理。
