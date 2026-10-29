# A100 新包：上传解压后直接运行，无需编辑代码或配置

适用于你当前的SSH：`ssh lvyuehang@10.10.101.35`。你已上传原common_v1除native外的数据，目录名为 `OCTA500_v1`；本包只读取z256和对应CSV。

## 1. 电脑PowerShell：上传这次的新代码包

```powershell
Set-Location 'D:\data\code\keyan_fuxian\FlowOCT'
scp .\FlowOCT_A100_20261021.zip .\FlowOCT_A100_20261021.zip.sha256 lvyuehang@10.10.101.35:
```

文件进入远程家目录，已有数据无需重传。

## 2. A100：在已上传的数据目录中解压

回到包含 `split_3M_z256.csv`、`split_6M_z256.csv` 和 `z256` 的 `OCTA500_v1` 目录。先检查：

```bash
pwd -P
ls -l split_3M_z256.csv split_6M_z256.csv
ls -ld z256/3M z256/6M
```

如果仍只显示 `total 0` 或找不到这些文件，先核对实际上传位置和上传是否完成。

校验新代码包（这个括号中的命令不会切换当前数据目录）：

```bash
(cd "$HOME" && sha256sum -c FlowOCT_A100_20261021.zip.sha256)
```

显示OK后，在当前数据目录解压：

```bash
python3 -m zipfile -e "$HOME/FlowOCT_A100_20261021.zip" .
cd flowoct_reproduction_a100
```

新包会生成独立的 `flowoct_reproduction_a100` 代码文件夹；保持它位于CSV的下一层，启动脚本就能自动定位实际数据绝对路径。

```text
OCTA500_v1/
├── split_3M_z256.csv
├── split_6M_z256.csv
├── z256/{3M,6M}/病例.npz
└── flowoct_reproduction_a100/
    ├── a100.py
    ├── requirements.txt
    ├── configs/
    └── flowoct/
```

## 3. 安装或激活环境

如果你还没有建立FlowOCT环境，先执行：

```bash
conda create -n flowoct python=3.10 -y
```

然后激活并安装锁定版本：

```bash
conda activate flowoct
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

若已建立并安装好同版本环境，只需激活。环境创建、网络和GPU分配仍按平台权限执行；遇到安装限制时保留报错，不改锁定版本。不要运行只针对旧3090服务器目录的 `scripts/setup_server.sh`。

## 4. 直接检查3M，自动完成路径配置

```bash
python a100.py check
```

这一个命令会：

1. 找到包含3M CSV的数据目录；若自动找到多个候选，列出候选并停止，避免选错数据。
2. 在进程内使用 `cuda:0`，检查依赖、GPU名称和可用显存。
3. 把实际路径和GPU编号写入本次检查配置，保留200轮、64张整批、FP32、推理批量32。
4. 审计全部3M清单，预期140/10/50。
5. 尝试正式256³、64张整批的一次完整参数更新，记录显存。
6. 跑一个训练病例与一个验证病例的正式尺寸保存、重载、推理和评价流程。

每次检查使用新的 `verification/a100_3m_时间戳/` 目录，保留配置、审计、单步JSON和完整流程诊断结果。检查失败即停止，不会减少切片、启用AMP、改回分批或自动开始200轮训练。

如果代码解压在其它地方，启动器也会检查当前目录、代码上级目录，以及这些位置和家目录中的常见 `OCTA500_v1` / `OCTA500_common_v1` 路径；不扫描整台服务器。找不到时可以用命令参数指定数据目录，无需编辑文件：

```bash
python a100.py check --data-dir /实际包含CSV的数据目录
```

若检查时使用了 `--data-dir`，正式train也带上同一参数。按第2步在CSV所在目录解压时，不需要这个参数。

仅在平台已分配的GPU计算环境内执行。若分配给进程的可见GPU不止一个，可以通过 `--device cuda:1` 指定另一个进程内编号；平台已有GPU可见范围不做修改。

## 5. 检查成功后，正式训练200轮

```bash
python a100.py train
```

自动生成 `configs/3m_a100.json` 并调用原训练入口。正式输出目录为 `runs/flowoct_3m_200e`，不会覆盖已有同名实验。已有实验用下面的resume；想另开一次可以用 `--run-name 新实验名`。

长训练采用平台作业方式；普通终端明确允许后台运行时，可选择以下后台命令替代上面的前台命令，两者只运行一个：

```bash
nohup python -u a100.py train > a100_3m_train.log 2>&1 < /dev/null &
tail -f a100_3m_train.log
```

日志里出现 `Epoch 1/200 scan 1/140 loss=...` 表示第一例更新完成。在tail里按Ctrl+C只停止看日志，后台训练继续。Slurm/PBS平台把 `python a100.py train` 放进平台提供的作业模板；nohup不会延长平台分配的资源时限。

## 6. 恢复和最终测试

恢复完整epoch边界的训练状态：

```bash
python a100.py resume
```

使用正式训练的best，对全部测试病例推理并评价：

```bash
python a100.py test
```

resume和test直接读取该run的 `config.json`，不重新搜索或切换数据目录。默认使用 `latest.pt` 恢复、`best.pt` 测试。测试输出为 `results/flowoct_3m_200e` 和 `evaluations/flowoct_3m_200e`，已经存在时停止，避免覆盖。

每轮完成训练和验证后更新latest，验证集平均体积PSNR严格提高时更新best，每10轮保留历史模型。完整200轮默认22个checkpoint文件；每份包括生成器、对比头、Adam、随机状态和配置。中断的半轮从上一完整轮末尾重新开始。

逐例step日志在中断前已经写入的半轮记录不会自动清理，恢复后可能重复；有效完成轮次以checkpoint为准。

## 7. 6M只需加一个参数

```bash
python a100.py check --fov 6m
python a100.py train --fov 6m
python a100.py resume --fov 6m
python a100.py test --fov 6m
```

按需要分阶段执行这些命令。两套默认都使用 `cuda:0`，顺序运行；6M正式目录为 `runs/flowoct_6m_200e`，预期180/20/100。

本次只增加自动部署入口和相应测试，不修改模型、损失、小波、数据预处理及训练循环。新增源码全文见 `EXTRA_CODE_REVIEW.md`，来源指纹见 `provenance/source_manifest.json`。

A100目标尚未连接或实际训练验证。包内保留的成功正式尺寸记录来自旧分批模式；新整批方式须以你执行check的结果为准。评价当前提供中心裁剪体积/轴向最大投影的MAE、PSNR和SSIM，论文完整LPIPS/FID、前景和分层协议尚未补齐。
