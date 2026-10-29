# 基于当前服务器数据的 FlowOCT 复现工程

## A100 当前入口：无需手工改配置

新包 `FlowOCT_A100_20261021.zip` 解压在你已上传的数据目录 `OCTA500_v1` 内，进入生成的 `flowoct_reproduction_a100`，激活环境后执行 `python a100.py check`，检查通过后执行 `python a100.py train`。数据路径自动定位，默认使用进程内 `cuda:0`。恢复用 `python a100.py resume`，最终测试用 `python a100.py test`；6M加 `--fov 6m`。完整上传与安装步骤在 [A100_START.md](A100_START.md)。

下面保留原3090服务器的路径和历史操作记录，A100使用上面的新入口。

本工程使用你已经放在服务器上的 `OCTA500_common_v1`，直接读取现有 CSV 和配对 NPZ。按你确定的复现口径，**模型、小波和损失采用作者公开二维代码，训练预算采用论文的 200 轮**，并补齐训练、恢复、推理和评价入口。

**当前默认关闭训练分批和激活重计算：`network_chunk=0`；正式尺寸仍抽取 64 张切片，一次送入主干。推理恢复作者每批最多 32 张的设置。** 早期 `network_chunk=4` 的验证结果单独保留，不能当作当前整批配置运行成功的证据。

原始代码锁定到 [cnu-medilab/FlowOCT](https://github.com/cnu-medilab/FlowOCT/tree/2309126956d2b80a95f2d307d85a50799cc658a6)，提交 `2309126956d2b80a95f2d307d85a50799cc658a6`。模型、小波和损失的改动逐项列在 [CHANGES.md](CHANGES.md)，逐行差异位于 `provenance/*.patch`。原项目的 `upstream/`、`working/`、`fm3d/` 保持原状。

作者公开实现与论文在二维/三维小波、独立 OCT 条件、投影和前景损失等处存在差异；本工程没有把这些差异悄悄改成自行设计的方法。**代码完整可运行，不代表已经完成论文表格的数值复现。** 论文未公开的完整评价流程、层范围和原始病例名单，无法从当前材料唯一恢复。

## 1. 服务器位置与现有数据

```text
/docsystem/niuke/lvyuehang/
├── OCTA500_common_v1/                   现有数据，只读使用
│   ├── split_3M_z256.csv                140 / 10 / 50
│   ├── split_6M_z256.csv                180 / 20 / 100
│   └── z256/{3M,6M}/病例.npz
├── flowoct_reproduction/                本工程
└── conda_envs/
    ├── lyh_fm3d/                        原有环境
    └── lyh_flowoct/                     FlowOCT 独立虚拟环境
```

每个 NPZ 含 `oct`、`octa`，数值仍是 0–255 单位的 float32，轴顺序是 `ZYX`：Z=轴向深度，Y=B-scan 序号，X=片内横向。3M 为 `256×304×304`，6M 为 `256×400×400`。已有离线处理仅把原始 Z=640 线性重采样为 256，未进行解剖 ROI 裁剪。

划分清单来自你已有的分层固定协议，数量与论文一致，具体 ID 不能称为作者原划分。清单中的同患者和 `leakage_group` 不允许跨集合。已知重复图像组 `10333/10388` 保留在训练集，继续沿用原清单，不重新划分或删除病例。图像重复检查不能证明所有自然人身份关系。

默认使用第二张卡 `cuda:1`。开跑前检查 GPU 占用；需要换卡时用 `--device cuda:0`。如果设置 `CUDA_VISIBLE_DEVICES=1`，进程内应写 `--device cuda:0`。

## 2. 安装和进入环境

服务器已准备独立环境。登录后执行：

```bash
cd /docsystem/niuke/lvyuehang/flowoct_reproduction
source /docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/activate
python -m pip check
```

重建该服务器环境时执行 `bash scripts/setup_server.sh`。脚本借用已有 `lyh_fm3d` 的只读基础包，在独立 venv 中安装 MONAI、TorchIO 等，不向正在训练的原环境安装包。此 venv 依赖原环境继续存在；迁移到另一台机器时请新建完整环境。

另一台 Linux 机器可这样安装，最后修改配置里的数据路径与 GPU：

```bash
conda create -n flowoct python=3.10 -y
conda activate flowoct
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

上述 CUDA wheel 需要兼容驱动；见 [PyTorch 2.5.1 官方安装表](https://pytorch.org/get-started/previous-versions/#v251)。所有实际版本保存在 `verification/server_environment_freeze.txt` 和每次运行的 `environment.json`。

## 3. 先检查，再做短程验证

```bash
python -m unittest discover -s tests -v
python run.py audit --config configs/3m.json --output verification/my_audit_3m.json
python run.py audit --config configs/6m.json --output verification/my_audit_6m.json
python run.py smoke --config configs/3m.json --output verification/my_smoke_3m
python run.py smoke --config configs/6m.json --output verification/my_smoke_6m
```

`audit` 检查全部清单条目、文件、NPZ 头、轴和身份元数据、数量及跨集合重复组；它不完整读取所有图像体素。加 `--deep` 会完整读取每个体，验证有限值和范围，并计算文件 SHA256，耗时更长。每次实际加载训练体也检查三维形状、有限值和范围。

`smoke` 用真实清单里的一个训练病例和一个验证病例，临时缩为 `32³`，完成训练更新、验证选模、保存/重新加载权重、验证集推理、指标导出。**不使用测试病例调试，不代表正式分辨率性能。** 每次使用新输出目录，避免覆盖旧结果。

正式尺寸的单步显存与速度检查：

```bash
python scripts/benchmark.py --config configs/3m.json --output verification/my_benchmark_3m.json
python scripts/benchmark.py --config configs/6m.json --output verification/my_benchmark_6m.json
```

它保留正式的 `256³` 输入、64 张抽片和全部损失，尝试一次反向更新后退出，不启动 200 轮训练。当前默认测试的是作者整批方式；24GB 3090 此前测试已出现显存不足，重试结果和原始日志见 [VERIFICATION.md](VERIFICATION.md)。显存不足会写入失败记录，不会自动减少抽片、改成分批或开启 AMP。

显存足够时，可做正式尺寸的一训练病例/一验证病例完整流程；恢复一致性检查使用小尺寸：

```bash
python run.py smoke --config configs/3m.json --full-size --output verification/my_fullsize_3m
python scripts/verify_resume.py --config configs/3m.json --output verification/my_resume_check
```

后者比较同卡上的连续两轮训练，与一轮后重新加载完整 checkpoint 再训练一轮的网络和 Adam 状态，使用真实数据的 `32³` 诊断尺寸。所有限制病例数量的运行都会记录诊断标记和实际病例名单。

## 4. 正式训练与完整流程

一条命令完成 3M 的训练、按验证 PSNR 选择最佳权重、完整测试集推理和评价：

```bash
bash scripts/run_experiment.sh 3m flowoct_3m_200e
```

6M 使用：

```bash
bash scripts/run_experiment.sh 6m flowoct_6m_200e
```

两份默认配置使用同一张 GPU，请顺序运行，并先通过正式尺寸的单步显存检查。一键脚本使用已激活环境中的 `python`，也可通过 `FLOWOCT_PYTHON` 指定解释器。迁移到 A100 时按实际位置修改两份配置的 `manifest` 和 `device`；尚未核对目标 A100 的显存和环境，不能保证单凭型号即可运行。

正式训练默认 200 epochs，来自论文的预算；作者仓库默认是 400，详见改动说明。训练每轮遍历一次全部训练病例，不额外设定 steps-per-epoch。损失权重保持公开代码的 `10 / 0.5 / 0.005 / 0.1`，不混用论文公式下的另一个系数体系。

需要分别运行或更改轮数时：

```bash
python -u run.py train --config configs/3m.json --output runs/flowoct_3m_200e
# 在另一台服务器上，可直接指定实际 CSV 路径与 GPU：
python -u run.py train --config configs/3m.json \
  --manifest /实际数据目录/split_3M_z256.csv --device cuda:0 \
  --output runs/flowoct_3m_200e
```

默认 `network_chunk=0` 走作者的 `self.netG(x_t_dwt, t)`，不调用激活重计算；`inference_chunk=32`，训练 200 轮。默认 FP32，没有 AMP、梯度裁剪、学习率调度和额外训练数据。

可选分批代码仍保留以便审阅历史改动，需要明确传 `--network-chunk 4` 才会开启。它保留全部 64 张片的统一损失，但改变浮点运算顺序，尚无完整收敛实验能确定其对最终指标的影响；当前方案不启用。若手动使用 `--train-slices 8`，对比负样本和投影范围会改变，属于另一实验。`--epochs 400` 仅用于选择作者脚本的轮数，不是当前采用的论文预算。

## 5. 中断后恢复

```bash
python -u run.py train --config runs/flowoct_3m_200e/config.json \
  --output runs/flowoct_3m_200e \
  --resume runs/flowoct_3m_200e/latest.pt
```

`latest.pt` 包含生成器、对比头、Adam 状态、已完成 epoch、global step、最佳验证分数、Python/NumPy/Torch/CUDA 随机状态、两个 DataLoader 的随机生成器状态及清单 SHA256。恢复时使用该 run 保存的 `config.json`，校验清单和训练配置，允许增加总 epochs 或明确换设备。旧分批实验不能在原 run 内切换成整批实验。相同软硬件下的下一步恢复已进行独立测试；不同设备不承诺逐位相同。

只在**完整 epoch 结束后**保存。若在一轮中途终止，该轮会从上一轮末尾重新执行；没有实现 batch 中途恢复。每轮更新 `latest.pt`，验证更优时更新 `best.pt`，每 10 轮保留一份历史快照。始终使用 `latest.pt` 恢复，`best.pt` 用于测试。

## 6. 推理、评价与看图

```bash
python run.py predict --config runs/flowoct_3m_200e/config.json \
  --checkpoint runs/flowoct_3m_200e/best.pt --output results/flowoct_3m_200e
python run.py evaluate --predictions results/flowoct_3m_200e \
  --output evaluations/flowoct_3m_200e
python scripts/visualize.py --input results/flowoct_3m_200e/10305.npz \
  --output evaluations/flowoct_3m_200e/10305.png
```

`10305` 是现有 3M 测试清单中的一个病例；自行更换清单时用真实测试病例文件名。推理入口使用配对测试数据以便评价，生成器计算不读取 GT。这里没有额外提供只有 OCT 的部署输入流程。

输出 NPZ 含 `oct`、预测 `octa`、真值 `octa_gt`、`axis_order=ZYX` 和 `case_id`，统一恢复到 ZYX **轴顺序**。输出仍是 resize 后中心裁剪得到的 `256³`，没有恢复原始 640 深度和完整横向 FOV，也没有把归一化强度伪装成原始强度。每个病例的裁剪坐标、翻转、归一化前 min/max 写在 `prediction.json` 中。

指标输出为 `per_case.csv` 和 `summary.json`：体素 MAE、PSNR、3D SSIM，以及沿 Z 的最大投影 MAE、PSNR、2D SSIM。固定 `data_range=1`、SSIM 窗口 7；逐扫描等权均值和总体标准差，同时报告各疾病组。精确相同时 PSNR 使用 float64 epsilon 下限，约 156.54 dB，以便 JSON 有限。

这些是**中心裁剪评价**，投影也只覆盖该裁剪。它们不能代表原始全体积、分层投影、论文前景 ROI 指标或病灶诊断性能。没有自行补造 LPIPS/FID、前景阈值、专家血管标签或诊断分类器；当前公开材料不足以唯一重建其完整协议。

## 7. 文件和记录

```text
configs/3m.json, 6m.json     正式参数及现有服务器路径
a100.py                    自动定位数据和生成配置的A100入口，不修改训练算法
A100_START.md              新代码包的上传、解压和直接运行步骤
flowoct/model.py            作者模型及明示修复/显存执行选项
flowoct/wavelets.py         作者矩阵小波，2D 矩阵跟随输入设备和类型
flowoct/wavelet_functions.py 原样保留的小波自动求导
flowoct/roi.py              原样保留的损失 ROI 工具
flowoct/data.py             CSV、重复组检查、轴转换、原流程预处理
flowoct/runner.py           训练/恢复/验证/测试/审计/评价
flowoct/metrics.py          明确固定协议的指标
scripts/                   安装、全流程、正式尺寸单步验证和看图
tests/                     轴、小波、泄漏、数值和恢复测试
reference/upstream/         锁定版本的相关作者原件
provenance/                 来源、清单指纹、逐行补丁和原数据协议
verification/              本次实际验证证据
```

训练后每个 run 还会生成 `config.json`、`environment.json`、`manifest.csv`、逐次更新 `steps.jsonl`、每轮 `epochs.jsonl` 及权重。配置修改会写进 run，恢复不会静默切换尺寸或损失设置。不要修改已经开始训练的 run 使用的源代码和数据。

本次没有启动 200/400 轮的正式训练；已完成的是交付工程、服务器适配和验证。实际检验范围及限制以 [VERIFICATION.md](VERIFICATION.md) 为准。
