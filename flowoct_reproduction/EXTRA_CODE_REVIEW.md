# 新增代码完整汇总，以及整批与可选分批执行的区别

核对日期：2026-10-21。对照作者提交 `2309126956d2b80a95f2d307d85a50799cc658a6`。

**初次交付确实默认开启了我增加的分批重计算；现在按你的要求恢复作者整批方式，`network_chunk=0/inference_chunk=32`，训练 200 轮。** 分批分支仍可审阅，当前默认不启用，训练不会调用激活重计算。

本页集中解释区别，并贴出真实补丁和全部新增代码。正文、配置、脚本和附录指纹已更新到当前版本。模型、小波、加载器与损失源码未随本次默认参数调整而改变；以作者二维代码为模型基准，轮数采用论文的 200 轮。

2026-10-21按你的要求新增A100自动部署入口 `a100.py` 和相应测试，上传解压后不再手工编辑代码或JSON。它只定位已有数据、填写路径/设备并调用原运行入口；全文在附录B，说明见第8节。原有科学计算源码的字节哈希未改变。

## 1. 先看你最关心的 64 → 4

可选分批模式仍然选 64 张，分 16 组，每组 4 张经过同一个主干网络，然后合起来算一次总损失、更新一次参数。**当前默认已恢复 64 张一次送入网络。**

```text
作者及当前默认 network_chunk=0：
同一个病例抽 64 张 → 一次送入网络 → 64 张结果 → 全部损失 → 更新 1 次

可选 network_chunk=4（当前不启用）：
同一个病例抽 64 张 → 每 4 张一组，依次过网络，共 16 组
                                       ↓
                              合回 64 张结果
                                       ↓
                        全部损失 → 更新 1 次
```

| 要比较的量 | 作者及当前默认执行方式 | 可选分批执行方式 |
|---|---|---|
| 每个训练数据项的病例数 | 1 | 1 |
| 正式尺寸下抽取的二维切片数 | 64 | 64 |
| 主干一次处理的切片数 | 64 | 4 |
| 初次前向调用主干的组数 | 1 | 16 |
| 对比学习使用的特征 | 64 个预测 + 64 个真值 | 64 个预测 + 64 个真值，仍统一比较 |
| 训练投影使用的切片 | 全部抽到的 64 张 | 全部抽到的 64 张 |
| 一次数据项的参数更新次数 | 1 | 1 |
| 主干中间结果的处理 | 保留用于反向传播 | 使用时重算所需的中间结果 |

正式尺寸的体块是 `256³`，按作者规则取第一空间轴长度的四分之一，即 `256 // 4 = 64`。这个 64 不是配置里的病例 `batch_size`。病例 `batch_size` 仍为 1。

按当前 YXZ 轴假设，抽的是不同 Y/B-scan 位置的片；不能因为源码变量名叫 D 就直接称为轴向 Z 深度切片。

## 2. 原代码和我加的代码，直接对照

作者原来只有这一句：

```python
pred_dx_dwt = self.netG(x_t_dwt, t)
```

我改成下面这个可关闭的分支。这是当前 `flowoct/model.py` 中的实际代码，不是重新写的示意版本：

```python
        network_chunk = getattr(self.opt, 'network_chunk', 0)
        if network_chunk:
            pred_dx_dwt = torch.cat([
                checkpoint(self.netG, x_t_dwt[start:start + network_chunk],
                           t[start:start + network_chunk], use_reentrant=False)
                for start in range(0, x_t_dwt.shape[0], network_chunk)
            ], dim=0)
        else:
            pred_dx_dwt = self.netG(x_t_dwt, t)
```

这里两个新增操作的作用分别是：

- `start:start + network_chunk`：仅在显式设置 `network_chunk=4` 时，每次取 4 张及其对应时间 t，把 `[64,4,128,128]` 分为 16 个 `[4,4,128,128]`；第二个维度的 4 是四个小波子带，与每组 4 张是两个不同概念。当前默认走 `else` 中的整批调用。
- `checkpoint(...)`：减少为反向传播长期保留的主干中间激活，在反向需要时重新计算。代价是额外计算开销；耗时不保证和原方式一样。[PyTorch 官方原理说明](https://docs.pytorch.org/docs/2.14/checkpoint.html#torch.utils.checkpoint.checkpoint)
- `torch.cat(..., dim=0)`：按原顺序合回全部 64 张预测，后面的 IDWT、终点估计、四项损失继续使用完整结果。

这里的 `checkpoint(...)` 是**显存重计算函数**。磁盘里的 `latest.pt/best.pt` 是**训练状态文件**。它们名字相似，但用途不同。

分组循环内部没有 `optimizer.step()`。参数更新仍在全部损失之后：

```python
self.loss.backward()
self.optimizer_G.step()
```

对比头仍统一处理全部 64 张预测和全部 64 张真值；没有把 16 组各自的对比损失算完再平均。否则，负样本集合会变，这就会改变训练目标。

## 3. 为什么加它，能否说与原代码完全一样

加它的直接原因是显存实测：

| 在第二张 3090 上的检查 | 结果 |
|---|---|
| 关闭分组，64 张一次处理，正式尺寸 | 显存不足；当时已分配约 22.11 GiB，再申请 1 GiB 时失败 |
| 旧分批模式，每组 4 张并重计算，仍使用全部 64 张 | 3M、6M 单步训练均通过，峰值约 4.38 GiB |

证据分别是 [原整批运行记录](verification/benchmark_3m_unchunked.json)、[3M 分组运行记录](verification/benchmark_3m_256.json)、[6M 分组运行记录](verification/benchmark_6m_256.json)。这些是单步/短程运行证据，不是收敛性能结果。

2026-10-20 按新默认配置在空闲第二张 3090 上再次整批尝试，仍在对比头 ReLU 申请 1 GiB 时显存不足，尚未完成反向更新。见 [本次重试记录](verification/benchmark_3m_unchunked_retry.json) 和 [错误日志](verification/benchmark_3m_unchunked_retry.log)。没有证据证明分批一定会明显改变最终指标；关闭它是为了尽量沿用作者执行方式。

**数学上保留相同的样本集合、网络和损失目标，不等于保证浮点结果、权重轨迹或最终论文指标完全一致。** 当前主干使用 GroupNorm，实际检查有 56 个 GroupNorm、0 个 BatchNorm，未启用非零 dropout；这种网络各切片的主干计算不依赖同组其它切片的 BatchNorm 统计。但批量形状变化仍可能改变浮点运算顺序，Adam 也可能放大接近零的梯度差异。

已有的对照测试只验证了 **CPU 小尺寸人工输入** 上整批与分组的损失、梯度在设定误差内接近：测试输入第一空间轴为 8，抽 2 张，比较一次处理 2 张与每组 1 张；损失容差 `3e-5`，梯度 `atol=1e-6, rtol=5e-4`。没有完成“正式尺寸整批 64 张与分组 64 张训练 200 轮后效果相同”的验证，前者已在这张卡上显存不足。

另外，之前报告的“恢复后参数逐位一致”，比较的是**相同分组模式下连续训练与中断恢复**，不是拿分组模式与作者整批模式比较。这两项检查应当分开理解。

如果两张 3090 各跑一个基线，每个基线仍只有自己那张卡的显存；可选分组模式也只在一张卡上执行，没有把这 16 组分配到两张卡。

## 4. 模型文件到底改了几处

`flowoct/model.py` 来自作者的 `models/FlowOCT_model.py`。原文件 288 行，当前 303 行。逐行差异是 **增加 24 行、删除/替换 9 行，合计 6 个修改块**，包括注释和导入。

| 修改块 | 具体变化 | 理由 / 影响 |
|---|---|---|
| 导入 | 增加 PyTorch 的 `checkpoint` 导入；把 `util.*` 改为包内相对导入 | 前者用于显存重计算，后者使本工程独立运行 |
| 自动加载权重 | 删除构造函数中自动调用作者两份权重加载的分支 | 改由 `runner.py` 统一恢复网络、对比头、Adam 和随机状态；保存/加载协议已改变 |
| 抽片数量选项 | 增加 `train_slices` 及范围检查 | 默认 0 表示沿用作者 `D//4`；显式填其它值会改变训练采样，本次默认没有减少抽片数 |
| 网络执行 | 原来一行 `self.netG(...)` 改成上文分组/重计算分支 | 增加了可选分支；现默认 0，直接调用作者整批路径 |
| 投影损失索引 | `[y,x]` 改成 `[:,y,x]`，预测和真值各一处 | 原投影是 `(C,H,W)`，必须保留通道维；否则 ROI 非零起点会得到空区域，产生 NaN |
| 非有限损失检查 | 在反向前检查 `torch.isfinite(self.loss)` | 损失出现 NaN/Inf 时停止，不把异常值继续更新进模型 |

网络层结构、生成器通道、Haar 子带、四项损失的形式和权重、三步 Euler 积分都保留公开实现。这里“保留”不包含继续使用错误的投影索引。

完整补丁在本页附录 A，不必另外翻文件。

## 5. 所有新增文件，按实际文件列出

以下统计只计算交付中的代码、配置和依赖文件，不把说明文档、运行日志、作者原件快照计为新训练代码。

**15 个新增文件，共 1173 行，包含空行和注释。** 所有文件的完整内容都贴在附录 B。

| 新增文件 | 行数 | 加了什么 | 何时用到 |
|---|---:|---|---|
| `flowoct/__init__.py` | 8 | 包初始化；设置确定性 CUDA 矩阵计算需要的环境变量 | 导入本工程时 |
| `flowoct/runner.py` | 417 | 命令行、训练、验证、恢复、推理、审计、评价；固定种子、版本和运行记录 | 正式运行的管理入口 |
| `flowoct/metrics.py` | 35 | 明确定义体 MAE/PSNR/SSIM、轴向最大投影指标和汇总 | 验证/评价 |
| `run.py` | 7 | 调用新运行入口 | 启动时 |
| `a100.py` | 143 | 自动找CSV根目录、填写A100实际配置、检查/训练/恢复/测试 | A100新包入口 |
| `configs/3m.json` | 21 | 3M 路径与参数，默认关闭分组重计算 | 3M 运行时 |
| `configs/6m.json` | 21 | 6M 路径与参数 | 6M 运行时 |
| `scripts/run_experiment.sh` | 13 | 串联训练→按实际保存配置推理→评价；使用已激活环境 | 使用一键流程时 |
| `scripts/setup_server.sh` | 12 | 建立/使用独立环境并安装依赖 | 安装时，不参与每步训练 |
| `scripts/benchmark.py` | 65 | 单个真实病例、正式尺寸下测一次反向更新和显存 | 手动检查时 |
| `scripts/verify_resume.py` | 50 | 比较连续训练和中断恢复 | 手动检查时 |
| `scripts/visualize.py` | 48 | 读取预测并画切面、投影和误差 | 手动画图时 |
| `scripts/build_provenance.py` | 63 | 保存作者快照、补丁和指纹；依赖本地原始 Git 仓库 | 交付维护时，不参与服务器训练 |
| `tests/test_core.py` | 168 | 轴向、数据检查、小波、分组梯度、恢复、投影索引测试 | 手动测试时 |
| `tests/test_a100.py` | 102 | 数据目录选择、仅路径/设备变化、已有实验及恢复保护 | 部署入口验证时 |

`runner.py` 的函数与起始行号索引：

| Function | Definition line |
|---|---:|
| `write_json()` | 29 |
| `load_config()` | 34 |
| `environment()` | 53 |
| `setup()` | 63 |
| `make_model()` | 87 |
| `seed_worker()` | 97 |
| `make_loader()` | 103 |
| `rng_state()` | 109 |
| `restore_rng()` | 115 |
| `save_checkpoint()` | 123 |
| `read_checkpoint()` | 134 |
| `to_zyx()` | 143 |
| `validate()` | 148 |
| `append_jsonl()` | 157 |
| `train()` | 162 |
| `predict()` | 258 |
| `evaluate()` | 296 |
| `audit()` | 329 |
| `main()` | 356 |


## 6. 修改或重写的文件与默认参数

另外有 **4 个修改/重写的文件**：

| 文件 | 作者版本 → 当前行数 | 实际变化 |
|---|---|---|
| `flowoct/model.py` | 288 → 303 | 上述 6 块修改；没有替换生成器架构 |
| `flowoct/wavelets.py` | 652 → 644 | 调整导入；2D DWT/IDWT 矩阵跟随输入设备和 dtype；系数、矩阵构造和自动求导公式保留 |
| `flowoct/data.py` | 127 → 165 | **加载器重写**：读取已有 CSV、检查病例/重复组/文件、ZYX→YXZ、统一预处理、返回几何信息；不是只加几行路径配置 |
| `requirements.txt` | 10 → 11 | 固定实际版本，补运行所需依赖，移除核心流程未调用的感知指标等依赖 |

其中不应被“工程修改”四个字掩盖的配置/协议变化如下：

| 项目 | 作者公开入口 | 当前默认 | 是否可能影响训练/结果 |
|---|---|---|---|
| 训练主干分组 | 整批执行 | `network_chunk=0`，不做激活重计算 | 已恢复作者整批调用；可选分组代码不参与当前默认运行 |
| 验证/推理分组 | 每组最多 32 张 | `inference_chunk=32` | 已恢复作者推理批量；三步 Euler 保持 |
| 训练轮数 | 400 | 200，采用论文写明的预算 | **会改变训练预算**；不是原脚本默认参数 |
| 数据入口 | 事先分好的三个文件夹 | 你现有的固定 CSV 清单 | 具体病例名单不保证等于论文原名单 |
| 数据轴 | 加载器注释示例末轴为 640 | 服务器 ZYX 转为 YXZ | 根据作者注释作出的轴适配假设，缺少作者打包脚本作最终确认 |
| 输入的既有离线处理 | 未公开完整打包流程 | 使用已有 Z=256 公共数据 | 与原始 Z=640 直接送作者 Resize 的全部细节不能划等号 |
| 种子 | 入口未固定 | 2026 | 不能恢复作者未知的原随机序列 |
| 数据 worker | 4 | 0 | 改变数据加载和随机状态组织，可改配置 |
| CPU 线程 | 原入口未统一指定 | 4 | 影响资源使用；有限精度执行算法也可能不同 |
| 确定性计算 | 原入口未明确统一控制 | 开启确定性算法，关闭 cuDNN benchmark | 可能影响速度与浮点计算路径 |
| 验证模式 | 原验证入口未显式切换 `eval()` | 验证时 `eval()`，下一轮 `train()` | 明确模型模式；当前没有非零 dropout 或 BatchNorm |
| 权重保存 | 分别保存主干/对比头权重 | 一个完整状态文件，latest/best/每10轮快照 | 改变恢复和文件管理协议 |
| 指标 | 仓库只有部分基础函数 | 新增明确的体/投影指标与 CSV | **不是补全了论文全部 foreground/LPIPS/FID/诊断协议** |

推理批量由新 `runner.py` 传入 `cfg["inference_chunk"]`；模型方法定义中的默认 `chunk_size=32` 保留。两份配置现也使用 32。只看模型文件的差异，不能发现所有运行参数变化。

**2 个原样复制的文件**是 `flowoct/wavelet_functions.py` 和 `flowoct/roi.py`；已核对与作者快照的字节哈希一致。它们不属于我新设计的损失或小波算法。

其它新增内容也有明确用途：`reference/upstream/` 是作者原件快照，`provenance/` 是差异和指纹，`verification/` 是运行证据，README/CHANGES/VERIFICATION/START_HERE/DELIVERY 是说明与交付记录。这些资料目录不作为额外模型模块参与前向计算。

## 7. 当前已关闭显存修改

当前两个配置文件都写着：

```json
"train_slices": 0,
"network_chunk": 0,
"inference_chunk": 32
```

- `train_slices=0` 沿用作者 `D//4` 规则，正式尺寸抽 64 张；并非不抽片。
- `network_chunk=0` 一次执行完整主干，不做激活重计算。
- `inference_chunk=32` 恢复作者推理批量。训练轮数保留论文的 200，当前不采用脚本的 400。

关闭分组后，ROI 修复、数据适配和完整状态管理仍然存在，不能把该运行称为“作者原文件完全未修改”。当前整批路径在这张 3090 上重试仍显存不足，本次没有启动 200 轮正式训练。后续迁移到 A100 时保留这些默认参数，先核对环境与单步显存。恢复和推理使用各自 run 保存的 `config.json`，避免把旧分批实验静默切换到当前参数。

## 8. A100自动入口新增了什么

新包解压到已有CSV和z256的数据目录中后，代码处于其下一层 `flowoct_reproduction_a100`。执行 `python a100.py check`，启动器检查CSV根目录、进程内cuda:0与锁定依赖，自动填写实际路径，然后顺序调用原来的audit、正式尺寸benchmark和full-size smoke。GPU预检在独立进程结束后释放CUDA上下文，不额外长期占用训练显存。

`python a100.py train`仍调用原200轮训练循环；`resume/test`读取原run保存的配置，分别使用latest/best。默认不覆盖已有run或测试输出，不把分批旧run或仅一例的诊断run当作正式实验恢复。不需要native，不扫描全服务器，也不移动或转换数据；多份候选CSV会明确列出，需用 `--data-dir` 指定。

新增源码为143行入口和102行部署测试，维护脚本更新新入口指纹。训练模板的科学参数没有变化，实际生成配置只改manifest/device。这里新增的是部署与调用流程，不是模型模块。7项本地部署测试通过；尚未连接或训练验证你的A100。

## 附录 A：作者文件与当前版本的完整差异

补丁以 `reference/upstream/` 中的锁定原件与当前真实源文件生成。`+` 表示当前版本增加的行，`-` 表示作者原行被删除或替换。模型、小波、加载器、依赖文件都列出。

### `models/FlowOCT_model.py` -> `flowoct/model.py`

```diff
--- upstream/models/FlowOCT_model.py
+++ flowoct/model.py
@@ -8,11 +8,12 @@
 import torch
 import torch.nn as nn
 import torch.nn.functional as F
+from torch.utils.checkpoint import checkpoint
 from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
 from tqdm import tqdm
 
-from util.DWT_IDWT_layer import DWT_2D, IDWT_2D
-from util.auto_crop import auto_crop_octa_images
+from .wavelets import DWT_2D, IDWT_2D
+from .roi import auto_crop_octa_images
 
 
 class FlowOCT(nn.Module):
@@ -61,9 +62,8 @@
             params = list(self.netG.parameters()) + list(self.contrast_projection_head.parameters())
             self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta, 0.999))
 
-        # Load checkpoint if resuming
-        if not self.isTrain or opt.continue_train:
-            self._load_checkpoint(opt.epoch)
+        # The runner restores one complete checkpoint (network, head, optimizer,
+        # RNG and loader state).
         
         self.generation_steps = 3
 
@@ -113,7 +113,9 @@
         B, C, D, H, W = self.real_A.shape
         
         # Sample random 2D slices along depth
-        num_slices = max(1, D // 4)
+        num_slices = getattr(self.opt, 'train_slices', 0) or max(1, D // 4)
+        if not 1 <= num_slices <= D:
+            raise ValueError('train_slices must lie in [1, volume slice count]')
         indices = torch.randperm(D, device=self.device)[:num_slices]
         
         oct_slices = self.real_A[:, :, indices, :, :]   # (B, C, N, H, W)
@@ -135,7 +137,18 @@
         LL, LH, HL, HH = self.dwt_2d(x_t)
         x_t_dwt = torch.cat([LL, LH, HL, HH], dim=1)
 
-        pred_dx_dwt = self.netG(x_t_dwt, t)
+        # Keep the full sampled slice set and both global auxiliary losses.
+        # Recompute small independent network chunks during backward to fit a
+        # 24GB GPU, rather than reducing the InfoNCE negative set or projection.
+        network_chunk = getattr(self.opt, 'network_chunk', 0)
+        if network_chunk:
+            pred_dx_dwt = torch.cat([
+                checkpoint(self.netG, x_t_dwt[start:start + network_chunk],
+                           t[start:start + network_chunk], use_reentrant=False)
+                for start in range(0, x_t_dwt.shape[0], network_chunk)
+            ], dim=0)
+        else:
+            pred_dx_dwt = self.netG(x_t_dwt, t)
   
         # DWT inverse
         pred_dx = self.idwt_2d(
@@ -171,8 +184,8 @@
         )
         xm, ym, xM, yM = bbox['expanded_bbox']
         self.proj_loss = self.criterionL1(
-            fake_proj[ym:yM+1, xm:xM+1],
-            real_proj[ym:yM+1, xm:xM+1],
+            fake_proj[:, ym:yM+1, xm:xM+1],
+            real_proj[:, ym:yM+1, xm:xM+1],
         )
 
         # Total loss
@@ -183,6 +196,8 @@
             self.projweight * self.proj_loss
         )
         
+        if not torch.isfinite(self.loss):
+            raise FloatingPointError('Non-finite FlowOCT loss; update was not applied')
         self.loss.backward()
         self.optimizer_G.step()
```

### `util/DWT_IDWT_layer.py` -> `flowoct/wavelets.py`

```diff
--- upstream/util/DWT_IDWT_layer.py
+++ flowoct/wavelets.py
@@ -5,7 +5,7 @@
 import torch
 from torch.nn import Module
 
-from .DWT_IDWT_Functions import DWTFunction_1D, IDWTFunction_1D, \
+from .wavelet_functions import DWTFunction_1D, IDWTFunction_1D, \
     DWTFunction_2D_tiny, DWTFunction_2D, IDWTFunction_2D, \
     DWTFunction_3D, IDWTFunction_3D
 
@@ -296,16 +296,10 @@
         matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
         matrix_g_1 = np.transpose(matrix_g_1)
 
-        if torch.cuda.is_available():
-            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
-            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
-            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
-            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
-        else:
-            self.matrix_low_0 = torch.Tensor(matrix_h_0)
-            self.matrix_low_1 = torch.Tensor(matrix_h_1)
-            self.matrix_high_0 = torch.Tensor(matrix_g_0)
-            self.matrix_high_1 = torch.Tensor(matrix_g_1)
+        self.matrix_low_0 = torch.as_tensor(matrix_h_0, device=self._matrix_device, dtype=self._matrix_dtype)
+        self.matrix_low_1 = torch.as_tensor(matrix_h_1, device=self._matrix_device, dtype=self._matrix_dtype)
+        self.matrix_high_0 = torch.as_tensor(matrix_g_0, device=self._matrix_device, dtype=self._matrix_dtype)
+        self.matrix_high_1 = torch.as_tensor(matrix_g_1, device=self._matrix_device, dtype=self._matrix_dtype)
 
     def forward(self, input):
         """
@@ -319,6 +313,8 @@
         assert len(input.size()) == 4
         self.input_height = input.size()[-2]
         self.input_width = input.size()[-1]
+        self._matrix_device = input.device
+        self._matrix_dtype = input.dtype
         self.get_matrix()
         return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)
 
@@ -386,16 +382,10 @@
         matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
         matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
         matrix_g_1 = np.transpose(matrix_g_1)
-        if torch.cuda.is_available():
-            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
-            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
-            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
-            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
-        else:
-            self.matrix_low_0 = torch.Tensor(matrix_h_0)
-            self.matrix_low_1 = torch.Tensor(matrix_h_1)
-            self.matrix_high_0 = torch.Tensor(matrix_g_0)
-            self.matrix_high_1 = torch.Tensor(matrix_g_1)
+        self.matrix_low_0 = torch.as_tensor(matrix_h_0, device=self._matrix_device, dtype=self._matrix_dtype)
+        self.matrix_low_1 = torch.as_tensor(matrix_h_1, device=self._matrix_device, dtype=self._matrix_dtype)
+        self.matrix_high_0 = torch.as_tensor(matrix_g_0, device=self._matrix_device, dtype=self._matrix_dtype)
+        self.matrix_high_1 = torch.as_tensor(matrix_g_1, device=self._matrix_device, dtype=self._matrix_dtype)
 
     def forward(self, LL, LH, HL, HH):
         """
@@ -414,6 +404,8 @@
             HL.size()) == len(HH.size()) == 4
         self.input_height = LL.size()[-2] + HH.size()[-2]
         self.input_width = LL.size()[-1] + HH.size()[-1]
+        self._matrix_device = LL.device
+        self._matrix_dtype = LL.dtype
         self.get_matrix()
         return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)
```

### `datasets/OCTA500_dataset.py` -> `flowoct/data.py`

```diff
--- upstream/datasets/OCTA500_dataset.py
+++ flowoct/data.py
@@ -1,127 +1,165 @@
-"""Dataset class for FlowOCT: loading paired OCT and OCTA 3D volumes."""
+"""Adapt the existing fixed OCTA500_common_v1 manifests to the author loader."""
 
+import csv
+import hashlib
+import random
+import re
+import zipfile
+from collections import Counter
+from pathlib import Path
 
-import os
-import random
 import numpy as np
 import torch
 import torchio as tio
-
-def _make_dataset(directory, max_size=float("inf")):
-    """Collect file paths from directory and subdirectories."""
-    paths = []
-    for root, _, fnames in sorted(os.walk(directory)):
-        for fname in fnames:
-            if fname.endswith('.npz'):
-                paths.append(os.path.join(root, fname))
-    return paths[:min(max_size, len(paths))]
+from torch.utils.data import Dataset
 
 
-class OCTA500Dataset(torch.utils.data.Dataset):
-    """Paired OCT-to-OCTA 3D dataset.
+COUNTS = {"3M": {"train": 140, "validation": 10, "test": 50},
+          "6M": {"train": 180, "validation": 20, "test": 100}}
 
-    Each .npz file contains 'oct' and 'octa' arrays.
+
+def sha256(path):
+    digest = hashlib.sha256()
+    with Path(path).open("rb") as stream:
+        for block in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(block)
+    return digest.hexdigest()
+
+
+def read_manifest(path, fov, check_counts=True):
+    path = Path(path).resolve()
+    with path.open(encoding="utf-8-sig", newline="") as stream:
+        rows = list(csv.DictReader(stream))
+    if not rows:
+        raise ValueError("Empty manifest")
+    cases, files, patients, groups = set(), set(), {}, {}
+    for row in rows:
+        if any(not row.get(k, "").strip() for k in ("case_id", "patient_id", "path", "split")):
+            raise ValueError("Manifest requires case_id, patient_id, path and split")
+        cid, pid, split = row["case_id"], row["patient_id"], row["split"]
+        if not re.fullmatch(r"[A-Za-z0-9_-]+", cid) or split not in COUNTS[fov]:
+            raise ValueError(f"Invalid case or split: {cid}, {split}")
+        if row.get("fov", fov) != fov:
+            raise ValueError(f"Mixed FOV manifest: {cid}")
+        source = (path.parent / row["path"]).resolve()
+        if cid in cases or source in files:
+            raise ValueError(f"Duplicate case/file: {cid}")
+        if not source.is_file() or source.suffix.lower() != ".npz":
+            raise FileNotFoundError(source)
+        if pid in patients and patients[pid] != split:
+            raise ValueError(f"Patient crosses splits: {pid}")
+        group = row.get("leakage_group", "").strip()
+        if group and group in groups and groups[group] != split:
+            raise ValueError(f"Duplicate image group crosses splits: {group}")
+        cases.add(cid)
+        files.add(source)
+        patients[pid] = split
+        if group:
+            groups[group] = split
+        row["resolved_path"] = str(source)
+    counts = dict(Counter(row["split"] for row in rows))
+    if check_counts and counts != COUNTS[fov]:
+        raise ValueError(f"{fov}: expected {COUNTS[fov]}, found {counts}")
+    return rows
+
+
+def inspect_header(row, axis_order, fov):
+    """Read NPZ array headers without inflating every large volume into RAM."""
+    shapes = []
+    with zipfile.ZipFile(row["resolved_path"]) as archive:
+        for key in ("oct", "octa"):
+            with archive.open(key + ".npy") as stream:
+                version = np.lib.format.read_magic(stream)
+                if version == (1, 0):
+                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
+                elif version == (2, 0):
+                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
+                else:
+                    raise ValueError(f"Unsupported NPY header version {version}")
+            if len(shape) != 3 or not np.issubdtype(dtype, np.number):
+                raise ValueError(f"Invalid {key} header for {row['case_id']}")
+            shapes.append(shape)
+    if shapes[0] != shapes[1]:
+        raise ValueError(f"Unaligned shapes: {row['case_id']}")
+    expected = (256, 304, 304) if fov == "3M" else (256, 400, 400)
+    zyx_shape = tuple(shapes[0][axis_order.index(axis)] for axis in "ZYX")
+    if zyx_shape != expected:
+        raise ValueError(f"Expected common z256 data {expected}, found {zyx_shape}")
+    with np.load(row["resolved_path"], allow_pickle=False) as data:
+        for key, value in (("axis_order", axis_order), ("case_id", row["case_id"]),
+                           ("patient_id", row["patient_id"]), ("intensity_scale", 255)):
+            if key in data and data[key].item() != value:
+                raise ValueError(f"NPZ {key} conflicts with manifest/config: {row['case_id']}")
+    return {"case_id": row["case_id"], "split": row["split"],
+            "shape": list(shapes[0]), "zyx_shape": list(zyx_shape)}
+
+
+class PairedVolumeDataset(Dataset):
+    """Original resize/crop/flip/minmax, after an explicit ZYX -> YXZ adapter.
+
+    Y is the B-scan index, X the in-scan lateral coordinate, Z axial depth.
+    YXZ follows the upstream loader's [400,400,640] comment; the upstream
+    packing script was not released, so this is a documented assumption.
     """
 
-    def __init__(self, opt, phase):
-        self.opt = opt
-        self.phase = phase
+    def __init__(self, rows, split, cfg, limit=0):
+        self.rows = [row for row in rows if row["split"] == split]
+        if limit:
+            self.rows = self.rows[:limit]
+        if not self.rows:
+            raise ValueError(f"No {split} cases")
+        self.split, self.cfg = split, cfg
+        self.resize = tio.Resize((cfg["load_size"], cfg["load_size"], cfg["crop_size"]))
 
-        data_dir = os.path.join(opt.dataroot, phase)
-        self.paths = sorted(_make_dataset(data_dir))
-        print(f"[Dataset] {phase}: {len(self.paths)} volumes")
+    def __len__(self):
+        return len(self.rows)
 
-        self.input_nc = opt.input_nc
-        self.output_nc = opt.output_nc
+    def __getitem__(self, index):
+        row = self.rows[index]
+        arrays = []
+        with np.load(row["resolved_path"], allow_pickle=False) as data:
+            if "axis_order" in data and data["axis_order"].item() != self.cfg["axis_order"]:
+                raise ValueError(f"Axis metadata mismatch: {row['case_id']}")
+            for key in ("oct", "octa"):
+                array = np.asarray(data[key], dtype=np.float32)
+                if array.ndim != 3 or not np.isfinite(array).all():
+                    raise ValueError(f"Non-finite or non-3D {key}: {row['case_id']}")
+                if array.min() < 0 or array.max() > 255:
+                    raise ValueError(f"Expected raw [0,255] units for {key}: {row['case_id']}")
+                arrays.append(array)
+        if arrays[0].shape != arrays[1].shape:
+            raise ValueError(f"OCT/OCTA shapes differ: {row['case_id']}")
+        raw_shape = arrays[0].shape
+        permutation = [self.cfg["axis_order"].index(axis) for axis in "YXZ"]
+        volumes = [np.asarray(self.resize(np.transpose(a, permutation).copy()[None]),
+                              dtype=np.float32) for a in arrays]
+        crop = self.cfg["crop_size"]
+        if self.split == "train":
+            starts = [random.randint(0, max(0, n - crop)) for n in volumes[0].shape[1:]]
+            flip = random.random() > 0.5
+        else:
+            starts = [(n - crop) // 2 for n in volumes[0].shape[1:]]
+            flip = False
+        region = (slice(None),) + tuple(slice(s, s + crop) for s in starts)
+        tensors, ranges = [], []
+        for volume in volumes:
+            volume = volume[region]
+            if flip:
+                volume = np.flip(volume, axis=-1)
+            low, high = float(volume.min()), float(volume.max())
+            if high <= low:
+                raise ValueError(f"Constant crop would produce NaN: {row['case_id']}")
+            tensors.append(torch.from_numpy(np.ascontiguousarray((volume - low) / (high - low))))
+            ranges.append([low, high])
+        return {"A": tensors[0], "B": tensors[1], "case_id": row["case_id"],
+                "patient_id": row["patient_id"], "disease": row.get("disease", "unknown"),
+                "geometry": {"stored_axis_order": self.cfg["axis_order"],
+                             "source_shape": list(raw_shape), "model_axis_order": "YXZ",
+                             "resized_shape": [self.cfg["load_size"], self.cfg["load_size"], crop],
+                             "crop_start_yxz": starts, "crop_size": crop, "flipped_z": flip,
+                             "oct_minmax": ranges[0], "octa_minmax": ranges[1]}}
 
 
-    def _get_crop_params(self, shape):
-        """Random crop (train) or center crop (val/test)."""
-        _, d, h, w = shape  # (C, D, H, W)
-
-        if self.phase == 'train':
-            d_start = random.randint(0, max(0, d - self.opt.crop_size))
-            h_start = random.randint(0, max(0, h - self.opt.crop_size))
-            w_start = random.randint(0, max(0, w - self.opt.crop_size))
-            flip = random.random() > 0.5
-        else:
-            d_start = (d - self.opt.crop_size) // 2
-            h_start = (h - self.opt.crop_size) // 2
-            w_start = (w - self.opt.crop_size) // 2
-            flip = False
-
-        return {
-            'crop': (d_start, d_start + self.opt.crop_size,
-                     h_start, h_start + self.opt.crop_size,
-                     w_start, w_start + self.opt.crop_size),
-            'flip': flip,
-        }
-        
-    def __getitem__(self, index):        
-        path = self.paths[index]
-
-        data = np.load(path) 
-
-        A = np.expand_dims(data['oct'], axis=0) # [1,400,400,640]
-        B = np.expand_dims(data['octa'], axis=0)
-
-        # Resize
-        A = self._resize(A)
-        B = self._resize(B)
-
-        # Crop & flip
-        params = self._get_crop_params(A.shape)
-        c = params['crop']
-        A = A[:, c[0]:c[1], c[2]:c[3], c[4]:c[5]]
-        B = B[:, c[0]:c[1], c[2]:c[3], c[4]:c[5]]
-
-        if params['flip']:
-            A = np.flip(A, axis=-1).copy()
-            B = np.flip(B, axis=-1).copy()
-        
-        # min-max normalize to [0, 1]
-        A = (A - A.min()) / (A.max() - A.min())
-        B = (B - B.min()) / (B.max() - B.min())
-
-        return {
-            'A': torch.from_numpy(A),
-            'B': torch.from_numpy(B),
-            'A_paths': path,
-            'B_paths': path,
-        }
-
-    def _resize(self, x):
-        """Resize volume to (load_size, load_size, crop_size)."""
-        transform = tio.Resize(
-            (self.opt.load_size, self.opt.load_size, self.opt.crop_size)
-        )
-        return transform(x)
-
-    def __len__(self):
-        return len(self.paths)
-
-
-
-
-if __name__ == '__main__':
-    import argparse
-
-    parser = argparse.ArgumentParser()
-    parser.add_argument('--dataroot', type=str, required=True)
-    parser.add_argument('--phase', type=str, default='train')
-    parser.add_argument('--input_nc', type=int, default=1)
-    parser.add_argument('--output_nc', type=int, default=1)
-    parser.add_argument('--load_size', type=int, default=400)
-    parser.add_argument('--crop_size', type=int, default=256)
-    opt = parser.parse_args()
-
-    dataset = OCTA500Dataset(opt, opt.phase)
-    print(f"Dataset size: {len(dataset)}")
-
-    sample = dataset[0]
-    print(f"A shape: {sample['A'].shape}, dtype: {sample['A'].dtype}")
-    print(f"B shape: {sample['B'].shape}, dtype: {sample['B'].dtype}")
-    print(f"A min/max: {sample['A'].min():.4f} / {sample['A'].max():.4f}")
-    print(f"B min/max: {sample['B'].min():.4f} / {sample['B'].max():.4f}")
-
-#python OCTA500_dataset.py --dataroot /root/autodl-tmp/cwdm-main-oct/datasets/OCT_volume_6mm_denoise --phase train+def collate_volumes(samples):
+    return {"A": torch.stack([s["A"] for s in samples]),
+            "B": torch.stack([s["B"] for s in samples]), "metadata": samples}
```

### `requirements.txt` -> `requirements.txt`

```diff
--- upstream/requirements.txt
+++ requirements.txt
@@ -1,10 +1,11 @@
-torch>=2.2.0,<3.0.0
-torchvision>=0.17.0,<1.0.0
-torchio>=0.20.0,<1.0.0
-monai>=1.4.0,<2.0.0
-numpy>=1.26.0,<2.0.0
-scikit-image>=0.25.0,<1.0.0
-tqdm>=4.66.0,<5.0.0
-opencv-python>=4.11.0,<5.0.0
-lpips>=0.1.4
-clean-fid>=0.1.35+# Install the CUDA wheel explicitly first; see README.
+torch==2.5.1
+numpy==1.26.4
+monai==1.4.0
+einops==0.8.1
+torchio==0.20.4
+scikit-image==0.25.2
+PyWavelets==1.8.0
+SimpleITK==2.5.6
+tqdm==4.67.1
+matplotlib==3.10.1
```


## 附录 B：全部新增源文件全文，以及重写后的加载器全文

以下直接从当前项目读取，包含上表 15 个新增文件及重写的 `flowoct/data.py`。原样复制的两个作者文件不重复粘贴。

### `flowoct/__init__.py`

```python
"""FlowOCT public-code reproduction, pinned to upstream 2309126."""

import os

# Required by deterministic CUDA matmul; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "1.0.0"
```

### `flowoct/metrics.py`

```python
"""Explicit [0,1] volume and axial max-projection metrics."""

import numpy as np
from skimage.metrics import structural_similarity


def image_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or min(target.shape) < 7:
        raise ValueError("Metrics require matching arrays with every axis >=7")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite metric input")
    if min(prediction.min(), target.min()) < 0 or max(prediction.max(), target.max()) > 1:
        raise ValueError("Metrics require normalized [0,1] arrays")
    difference = prediction.astype(np.float64) - target
    mse = float(np.mean(difference ** 2))
    # Cap exact-match PSNR at the float64 epsilon floor so JSON remains finite.
    return {"mae": float(np.mean(np.abs(difference))),
            "psnr": float(-10 * np.log10(max(mse, np.finfo(np.float64).eps))),
            "ssim": float(structural_similarity(target, prediction, data_range=1.0, win_size=7))}


def volume_metrics(prediction_zyx, target_zyx):
    result = {"volume_" + k: v for k, v in image_metrics(prediction_zyx, target_zyx).items()}
    result.update({"axial_mip_" + k: v for k, v in image_metrics(
        prediction_zyx.max(axis=0), target_zyx.max(axis=0)).items()})
    return result


def summarize(rows):
    keys = [key for key in rows[0] if key.startswith(("volume_", "axial_mip_"))]
    return {"cases": len(rows), "aggregation": "equal weight per scan; population standard deviation",
            "metrics": {key: {"mean": float(np.mean([row[key] for row in rows])),
                              "std": float(np.std([row[key] for row in rows]))} for key in keys}}
```

### `flowoct/runner.py`

```python
"""Auditable single-GPU train/validate/checkpoint/predict/evaluate workflow."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (PairedVolumeDataset, collate_volumes, inspect_header,
                   read_manifest, sha256)
from .metrics import summarize, volume_metrics


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def load_config(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        cfg = json.load(stream)
    if sorted(cfg["axis_order"]) != ["X", "Y", "Z"]:
        raise ValueError("axis_order must be a permutation of ZYX")
    if cfg["batch_size"] != 1:
        raise ValueError("Use batch_size=1: upstream projection otherwise mixes patients")
    if cfg["load_size"] < cfg["crop_size"] or cfg["crop_size"] < 32 or cfg["crop_size"] % 16:
        raise ValueError("Require load_size >= crop_size, crop_size >=32 and a multiple of 16")
    for key in ("epochs", "inference_chunk", "val_every", "save_every", "threads"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["workers"] < 0 or cfg["network_chunk"] < 0 or not 0 <= cfg["train_slices"] <= cfg["crop_size"]:
        raise ValueError("Invalid worker or training slice count")
    if cfg["lr"] <= 0 or not 0 <= cfg["beta"] < 1:
        raise ValueError("Invalid Adam settings")
    return cfg


def environment(device):
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "monai", "torchio", "scikit-image", "PyWavelets", "SimpleITK", "einops")}
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                              for path in sorted((ROOT / "flowoct").glob("*.py"))},
            "device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def setup(cfg):
    torch.set_num_threads(cfg["threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu for diagnostics")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(cfg["seed"])
    # Epoch-boundary replay is tested on the same software/hardware. This does
    # not promise identical floating point results across different GPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    versions = environment(device)["packages"]
    expected = {"torch": "2.5.1", "numpy": "1.26.4", "monai": "1.4.0", "torchio": "0.20.4"}
    for package, version in expected.items():
        if versions[package].split("+")[0] != version:
            raise RuntimeError(f"Baseline requires {package}=={version}; got {versions[package]}")
    return device


def make_model(cfg, training):
    from .model import FlowOCT
    device = torch.device(cfg["device"])
    opt = SimpleNamespace(gpu_ids=[device.index or 0] if device.type == "cuda" else [],
                          isTrain=training, checkpoints_dir=".", name="unused",
                          lr=cfg["lr"], beta=cfg["beta"], train_slices=cfg["train_slices"],
                          network_chunk=cfg["network_chunk"])
    return FlowOCT(opt)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, cfg, generator, training=False):
    return DataLoader(dataset, batch_size=1, shuffle=training, num_workers=cfg["workers"],
                      generator=generator, worker_init_fn=seed_worker, collate_fn=collate_volumes,
                      persistent_workers=False, pin_memory=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path, model, cfg, epoch, global_step, best_psnr, manifest_hash, generators):
    payload = {"format": "flowoct-reproduction-v1", "model": model.state_dict(),
               "optimizer": model.optimizer_G.state_dict(), "config": cfg,
               "epoch": epoch, "global_step": global_step, "best_psnr": best_psnr,
               "manifest_sha256": manifest_hash, "rng": rng_state(),
               "loader_rng": {key: value.get_state() for key, value in generators.items()}}
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    # Includes Python/NumPy optimizer and RNG objects. Load only your own
    # checkpoints; this project does not download or load third-party weights.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "flowoct-reproduction-v1":
        raise ValueError("Not a full reproduction checkpoint; upstream two-file weights differ")
    return payload


def to_zyx(tensor):
    return tensor.detach().float().cpu().numpy()[0, 0].transpose(2, 0, 1).copy()


@torch.no_grad()
def validate(model, loader, cfg):
    model.eval()
    scores = []
    for batch in loader:
        prediction = model.validation(batch["A"], None, cfg["inference_chunk"])
        scores.append(volume_metrics(to_zyx(prediction), to_zyx(batch["B"])))
    return summarize(scores)


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def train(cfg, output, resume=None, limits=None):
    cfg = dict(cfg)
    cfg["diagnostic_limits"] = dict(limits) if limits else None
    device = setup(cfg)
    rows = read_manifest(cfg["manifest"], cfg["fov"])
    manifest_hash = sha256(cfg["manifest"])
    output = Path(output).resolve()
    checkpoint = read_checkpoint(resume) if resume else None
    if checkpoint:
        if output != Path(resume).resolve().parent:
            raise ValueError("Resume into the checkpoint's existing run directory")
        allowed_changes = {"epochs", "device"}
        if cfg["device"] != checkpoint["config"]["device"]:
            print("Device changed: exact numerical replay is only verified on the same GPU.", flush=True)
        for key, value in cfg.items():
            if key not in allowed_changes and checkpoint["config"].get(key) != value:
                raise ValueError(f"Resume config changed: {key}")
        if manifest_hash != checkpoint["manifest_sha256"]:
            raise ValueError("Manifest changed since checkpoint")
        # Latest is the authoritative epoch-boundary state. Older snapshots
        # need a new run with an explicit protocol, rather than stale future logs.
        if Path(resume).resolve() != output / "latest.pt":
            raise ValueError("Resume from latest.pt; use best.pt for evaluation")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "environment.json", environment(device))
        shutil.copyfile(cfg["manifest"], output / "manifest.csv")
    write_json(output / "config.json", cfg)
    train_dataset = PairedVolumeDataset(rows, "train", cfg, (limits or {}).get("train", 0))
    val_dataset = PairedVolumeDataset(rows, "validation", cfg, (limits or {}).get("validation", 0))
    write_json(output / "training_scope.json", {
        "limited_diagnostic": bool(limits),
        "train_cases": [row["case_id"] for row in train_dataset.rows],
        "validation_cases": [row["case_id"] for row in val_dataset.rows]})
    generators = {key: torch.Generator().manual_seed(cfg["seed"] + offset)
                  for offset, key in enumerate(("train", "validation"))}
    loaders = {"train": make_loader(train_dataset, cfg, generators["train"], True),
               "validation": make_loader(val_dataset, cfg, generators["validation"])}
    model = make_model(cfg, True)
    start_epoch, global_step, best_psnr = 1, 0, float("-inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        model.optimizer_G.load_state_dict(checkpoint["optimizer"])
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best_psnr = checkpoint["best_psnr"]
        for key, state in checkpoint["loader_rng"].items():
            generators[key].set_state(state.cpu())
        restore_rng(checkpoint["rng"])
    if start_epoch > cfg["epochs"]:
        raise ValueError("No remaining epochs; increase --epochs to continue")
    print(json.dumps({"run": str(output), "device": str(device), "start_epoch": start_epoch,
                      "train_cases": len(train_dataset), "validation_cases": len(val_dataset),
                      "sampled_slices": cfg["train_slices"] or cfg["crop_size"] // 4,
                      "generator_parameters": sum(p.numel() for p in model.netG.parameters())}), flush=True)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        tic = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses = []
        for step, batch in enumerate(loaders["train"], 1):
            model.set_input(batch)
            model.forward()
            global_step += 1
            loss = {"total": model.loss.item(), **model.get_current_loss()}
            losses.append(loss)
            append_jsonl(output / "steps.jsonl", {"epoch": epoch, "step": global_step,
                                                    "case_id": batch["metadata"][0]["case_id"], **loss})
            if step == 1 or step % cfg["log_every"] == 0:
                print(f"Epoch {epoch}/{cfg['epochs']} scan {step}/{len(train_dataset)} loss={loss['total']:.6f}", flush=True)
        row = {"epoch": epoch, "global_step": global_step,
               "loss": {key: float(np.mean([item[key] for item in losses])) for key in losses[0]},
               "train_seconds": time.perf_counter() - tic}
        improved = False
        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            tic = time.perf_counter()
            row["validation"] = validate(model, loaders["validation"], cfg)
            row["validation_seconds"] = time.perf_counter() - tic
            psnr = row["validation"]["metrics"]["volume_psnr"]["mean"]
            improved = psnr > best_psnr
            best_psnr = max(best_psnr, psnr)
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        row["best_validation_psnr"] = best_psnr if np.isfinite(best_psnr) else None
        # No scheduler, AMP or scaler is used, matching the public FP32 loop.
        save_checkpoint(output / "latest.pt", model, cfg, epoch, global_step, best_psnr,
                        manifest_hash, generators)
        if improved:
            shutil.copyfile(output / "latest.pt", output / "best.pt")
        if epoch % cfg["save_every"] == 0:
            shutil.copyfile(output / "latest.pt", output / f"epoch_{epoch:04d}.pt")
        append_jsonl(output / "epochs.jsonl", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return output


@torch.no_grad()
def predict(cfg, checkpoint_path, output, split="test", limit=0):
    device = setup(cfg)
    checkpoint = read_checkpoint(checkpoint_path)
    if checkpoint["manifest_sha256"] != sha256(cfg["manifest"]):
        raise ValueError("Prediction manifest differs from the training manifest")
    for key in ("axis_order", "crop_size", "load_size", "fov"):
        if cfg[key] != checkpoint["config"][key]:
            raise ValueError(f"Prediction preprocessing differs from training: {key}")
    rows = read_manifest(cfg["manifest"], cfg["fov"])
    dataset = PairedVolumeDataset(rows, split, cfg, limit)
    loader = make_loader(dataset, cfg, torch.Generator().manual_seed(cfg["seed"]))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = make_model(cfg, False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    records = []
    for batch in loader:
        meta = batch["metadata"][0]
        prediction = model.validation(batch["A"], None, cfg["inference_chunk"])
        arrays = {"oct": to_zyx(batch["A"]), "octa": to_zyx(prediction),
                  "octa_gt": to_zyx(batch["B"])}
        np.savez_compressed(output / f"{meta['case_id']}.npz", **arrays,
                            axis_order=np.asarray("ZYX"), case_id=np.asarray(meta["case_id"]))
        record = {key: meta[key] for key in ("case_id", "patient_id", "disease", "geometry")}
        record.update(volume_metrics(arrays["octa"], arrays["octa_gt"]))
        records.append(record)
        print(f"Saved {meta['case_id']} {arrays['octa'].shape}", flush=True)
    write_json(output / "prediction.json", {"checkpoint": str(Path(checkpoint_path).resolve()),
               "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
               "manifest_sha256": checkpoint["manifest_sha256"], "split": split,
               "limited_diagnostic": bool(limit or checkpoint["config"].get("diagnostic_limits")),
               "config": cfg, "environment": environment(device),
               "coverage": "center crop after resize; not restored full FOV or native axial resolution",
               "records": records, "summary": summarize(records)})
    return output


def evaluate(predictions, output):
    predictions, output = Path(predictions).resolve(), Path(output).resolve()
    info = json.loads((predictions / "prediction.json").read_text(encoding="utf-8"))
    expected = {record["case_id"] for record in info["records"]}
    actual = {path.stem for path in predictions.glob("*.npz")}
    if not expected or actual != expected:
        raise ValueError("Prediction files do not match the recorded case set")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for record in info["records"]:
        with np.load(predictions / f"{record['case_id']}.npz", allow_pickle=False) as data:
            if data["axis_order"].item() != "ZYX":
                raise ValueError("Expected saved ZYX arrays")
            score = volume_metrics(data["octa"], data["octa_gt"])
        rows.append({"case_id": record["case_id"], "patient_id": record["patient_id"],
                     "disease": record["disease"], **score})
    with (output / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["protocol"] = {"data_range": 1, "ssim_win_size": 7,
                           "ssim": "3D volume / 2D axial maximum projection; no anatomical layers",
                           "exact_match_psnr_cap_db": float(-10 * np.log10(np.finfo(np.float64).eps)),
                           "coverage": info["coverage"], "limited_diagnostic": info["limited_diagnostic"],
                           "split": info["split"], "checkpoint_sha256": info["checkpoint_sha256"]}
    summary["by_disease"] = {disease: summarize([row for row in rows if row["disease"] == disease])
                             for disease in sorted({row["disease"] for row in rows})}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def audit(cfg, output, deep=False):
    rows = read_manifest(cfg["manifest"], cfg["fov"])
    records = []
    for row in rows:
        record = inspect_header(row, cfg["axis_order"], cfg["fov"])
        if deep:
            with np.load(row["resolved_path"], allow_pickle=False) as data:
                record["ranges"] = {}
                for key in ("oct", "octa"):
                    a = data[key]
                    low, high = float(a.min()), float(a.max())
                    if not np.isfinite(a).all() or not 0 <= low < high <= 255:
                        raise ValueError(f"Invalid {key} values: {row['case_id']}")
                    record["ranges"][key] = [low, high]
            record["sha256"] = sha256(row["resolved_path"])
        records.append(record)
    result = {"fov": cfg["fov"], "counts": dict(Counter(row["split"] for row in rows)),
              "manifest_sha256": sha256(cfg["manifest"]), "audit": "all values and file hashes" if deep else "all paths, headers, metadata and split groups",
              "records": records}
    path = Path(output).resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "train", "predict", "smoke"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--manifest", type=Path)
        sub.add_argument("--device")
        sub.add_argument("--output", required=True, type=Path)
        if command == "audit":
            sub.add_argument("--deep", action="store_true")
        if command == "smoke":
            sub.add_argument("--full-size", action="store_true", help="Keep formal resize/crop settings; still one train/validation scan")
        if command in ("train", "predict"):
            sub.add_argument("--inference-chunk", type=int)
        if command == "train":
            sub.add_argument("--resume", type=Path)
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--train-slices", type=int)
            sub.add_argument("--network-chunk", type=int)
        if command == "predict":
            sub.add_argument("--checkpoint", required=True, type=Path)
            sub.add_argument("--split", choices=("validation", "test"), default="test")
            sub.add_argument("--limit", type=int, default=0, help="Diagnostic only; 0 = complete split")
    sub = commands.add_parser("evaluate")
    sub.add_argument("--predictions", required=True, type=Path)
    sub.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.predictions, args.output)
        return
    cfg = load_config(args.config)
    for key in ("manifest", "device", "epochs", "train_slices", "network_chunk", "inference_chunk"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    cfg["manifest"] = str(Path(cfg["manifest"]).resolve())
    if cfg["inference_chunk"] < 1:
        raise ValueError("inference_chunk must be positive")
    # Revalidate command-line overrides through the same checks.
    if args.command == "train":
        if cfg["epochs"] < 1 or cfg["network_chunk"] < 0 or not 0 <= cfg["train_slices"] <= cfg["crop_size"] or cfg["inference_chunk"] < 1:
            raise ValueError("Invalid training overrides")
        train(cfg, args.output, args.resume)
    elif args.command == "predict":
        if args.limit < 0:
            raise ValueError("limit must be nonnegative")
        predict(cfg, args.checkpoint, args.output, args.split, args.limit)
    elif args.command == "audit":
        audit(cfg, args.output, args.deep)
    else:
        # Uses actual server data, but only one train and validation scan at 32^3.
        # The isolated output is explicitly marked diagnostic throughout.
        cfg.update(epochs=1, train_slices=0, workers=0, save_every=1, val_every=1)
        if not args.full_size:
            cfg.update(load_size=32, crop_size=32)
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        run = train(cfg, output / "train", limits={"train": 1, "validation": 1})
        predictions = predict(cfg, run / "best.pt", output / "predictions", "validation", 1)
        evaluate(predictions, output / "evaluation")
        print("Real-data diagnostic completed; not a paper performance reproduction.", flush=True)
```

### `run.py`

```python
"""Run from this directory: python run.py --help."""

from flowoct.runner import main


if __name__ == "__main__":
    main()
```

### `flowoct/data.py`

```python
"""Adapt the existing fixed OCTA500_common_v1 manifests to the author loader."""

import csv
import hashlib
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset


COUNTS = {"3M": {"train": 140, "validation": 10, "test": 50},
          "6M": {"train": 180, "validation": 20, "test": 100}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path, fov, check_counts=True):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest")
    cases, files, patients, groups = set(), set(), {}, {}
    for row in rows:
        if any(not row.get(k, "").strip() for k in ("case_id", "patient_id", "path", "split")):
            raise ValueError("Manifest requires case_id, patient_id, path and split")
        cid, pid, split = row["case_id"], row["patient_id"], row["split"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", cid) or split not in COUNTS[fov]:
            raise ValueError(f"Invalid case or split: {cid}, {split}")
        if row.get("fov", fov) != fov:
            raise ValueError(f"Mixed FOV manifest: {cid}")
        source = (path.parent / row["path"]).resolve()
        if cid in cases or source in files:
            raise ValueError(f"Duplicate case/file: {cid}")
        if not source.is_file() or source.suffix.lower() != ".npz":
            raise FileNotFoundError(source)
        if pid in patients and patients[pid] != split:
            raise ValueError(f"Patient crosses splits: {pid}")
        group = row.get("leakage_group", "").strip()
        if group and group in groups and groups[group] != split:
            raise ValueError(f"Duplicate image group crosses splits: {group}")
        cases.add(cid)
        files.add(source)
        patients[pid] = split
        if group:
            groups[group] = split
        row["resolved_path"] = str(source)
    counts = dict(Counter(row["split"] for row in rows))
    if check_counts and counts != COUNTS[fov]:
        raise ValueError(f"{fov}: expected {COUNTS[fov]}, found {counts}")
    return rows


def inspect_header(row, axis_order, fov):
    """Read NPZ array headers without inflating every large volume into RAM."""
    shapes = []
    with zipfile.ZipFile(row["resolved_path"]) as archive:
        for key in ("oct", "octa"):
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unsupported NPY header version {version}")
            if len(shape) != 3 or not np.issubdtype(dtype, np.number):
                raise ValueError(f"Invalid {key} header for {row['case_id']}")
            shapes.append(shape)
    if shapes[0] != shapes[1]:
        raise ValueError(f"Unaligned shapes: {row['case_id']}")
    expected = (256, 304, 304) if fov == "3M" else (256, 400, 400)
    zyx_shape = tuple(shapes[0][axis_order.index(axis)] for axis in "ZYX")
    if zyx_shape != expected:
        raise ValueError(f"Expected common z256 data {expected}, found {zyx_shape}")
    with np.load(row["resolved_path"], allow_pickle=False) as data:
        for key, value in (("axis_order", axis_order), ("case_id", row["case_id"]),
                           ("patient_id", row["patient_id"]), ("intensity_scale", 255)):
            if key in data and data[key].item() != value:
                raise ValueError(f"NPZ {key} conflicts with manifest/config: {row['case_id']}")
    return {"case_id": row["case_id"], "split": row["split"],
            "shape": list(shapes[0]), "zyx_shape": list(zyx_shape)}


class PairedVolumeDataset(Dataset):
    """Original resize/crop/flip/minmax, after an explicit ZYX -> YXZ adapter.

    Y is the B-scan index, X the in-scan lateral coordinate, Z axial depth.
    YXZ follows the upstream loader's [400,400,640] comment; the upstream
    packing script was not released, so this is a documented assumption.
    """

    def __init__(self, rows, split, cfg, limit=0):
        self.rows = [row for row in rows if row["split"] == split]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No {split} cases")
        self.split, self.cfg = split, cfg
        self.resize = tio.Resize((cfg["load_size"], cfg["load_size"], cfg["crop_size"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        arrays = []
        with np.load(row["resolved_path"], allow_pickle=False) as data:
            if "axis_order" in data and data["axis_order"].item() != self.cfg["axis_order"]:
                raise ValueError(f"Axis metadata mismatch: {row['case_id']}")
            for key in ("oct", "octa"):
                array = np.asarray(data[key], dtype=np.float32)
                if array.ndim != 3 or not np.isfinite(array).all():
                    raise ValueError(f"Non-finite or non-3D {key}: {row['case_id']}")
                if array.min() < 0 or array.max() > 255:
                    raise ValueError(f"Expected raw [0,255] units for {key}: {row['case_id']}")
                arrays.append(array)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError(f"OCT/OCTA shapes differ: {row['case_id']}")
        raw_shape = arrays[0].shape
        permutation = [self.cfg["axis_order"].index(axis) for axis in "YXZ"]
        volumes = [np.asarray(self.resize(np.transpose(a, permutation).copy()[None]),
                              dtype=np.float32) for a in arrays]
        crop = self.cfg["crop_size"]
        if self.split == "train":
            starts = [random.randint(0, max(0, n - crop)) for n in volumes[0].shape[1:]]
            flip = random.random() > 0.5
        else:
            starts = [(n - crop) // 2 for n in volumes[0].shape[1:]]
            flip = False
        region = (slice(None),) + tuple(slice(s, s + crop) for s in starts)
        tensors, ranges = [], []
        for volume in volumes:
            volume = volume[region]
            if flip:
                volume = np.flip(volume, axis=-1)
            low, high = float(volume.min()), float(volume.max())
            if high <= low:
                raise ValueError(f"Constant crop would produce NaN: {row['case_id']}")
            tensors.append(torch.from_numpy(np.ascontiguousarray((volume - low) / (high - low))))
            ranges.append([low, high])
        return {"A": tensors[0], "B": tensors[1], "case_id": row["case_id"],
                "patient_id": row["patient_id"], "disease": row.get("disease", "unknown"),
                "geometry": {"stored_axis_order": self.cfg["axis_order"],
                             "source_shape": list(raw_shape), "model_axis_order": "YXZ",
                             "resized_shape": [self.cfg["load_size"], self.cfg["load_size"], crop],
                             "crop_start_yxz": starts, "crop_size": crop, "flipped_z": flip,
                             "oct_minmax": ranges[0], "octa_minmax": ranges[1]}}


def collate_volumes(samples):
    return {"A": torch.stack([s["A"] for s in samples]),
            "B": torch.stack([s["B"] for s in samples]), "metadata": samples}
```

### `configs/3m.json`

```json
{
  "manifest": "/docsystem/niuke/lvyuehang/OCTA500_common_v1/split_3M_z256.csv",
  "fov": "3M",
  "axis_order": "ZYX",
  "load_size": 400,
  "crop_size": 256,
  "batch_size": 1,
  "train_slices": 0,
  "network_chunk": 0,
  "inference_chunk": 32,
  "epochs": 200,
  "lr": 0.0002,
  "beta": 0.5,
  "seed": 2026,
  "workers": 0,
  "device": "cuda:1",
  "threads": 4,
  "val_every": 1,
  "save_every": 10,
  "log_every": 20
}
```

### `configs/6m.json`

```json
{
  "manifest": "/docsystem/niuke/lvyuehang/OCTA500_common_v1/split_6M_z256.csv",
  "fov": "6M",
  "axis_order": "ZYX",
  "load_size": 400,
  "crop_size": 256,
  "batch_size": 1,
  "train_slices": 0,
  "network_chunk": 0,
  "inference_chunk": 32,
  "epochs": 200,
  "lr": 0.0002,
  "beta": 0.5,
  "seed": 2026,
  "workers": 0,
  "device": "cuda:1",
  "threads": 4,
  "val_every": 1,
  "save_every": 10,
  "log_every": 20
}
```

### `scripts/run_experiment.sh`

```bash
#!/usr/bin/env bash
# Uses the configured author execution; activate your environment first.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
fov="${1:?Usage: bash scripts/run_experiment.sh 3m|6m [run_name]}"
case "$fov" in 3m|6m) ;; *) printf 'Choose 3m or 6m\n' >&2; exit 2 ;; esac
run_name="${2:-flowoct_${fov}_seed2026}"
python_bin="${FLOWOCT_PYTHON:-python}"
"$python_bin" -u run.py train --config "configs/$fov.json" --output "runs/$run_name"
# Inference reuses the configuration saved by this training run.
"$python_bin" -u run.py predict --config "runs/$run_name/config.json" --checkpoint "runs/$run_name/best.pt" --output "results/$run_name"
"$python_bin" -u run.py evaluate --predictions "results/$run_name" --output "evaluations/$run_name"
```

### `scripts/setup_server.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_python="${FLOWOCT_BASE_PYTHON:-/docsystem/niuke/lvyuehang/conda_envs/lyh_fm3d/bin/python}"
env_dir="${FLOWOCT_ENV:-/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct}"
if [[ ! -x "$env_dir/bin/python" ]]; then
  "$base_python" -m venv --system-site-packages "$env_dir"
fi
"$env_dir/bin/python" -m pip install -r "$root/requirements.txt"
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" -m pip freeze > "$root/verification/server_environment_freeze.txt"
printf 'FlowOCT Python: %s\n' "$env_dir/bin/python"
```

### `scripts/benchmark.py`

```python
"""One real-data update at the configured training resolution; no long run."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flowoct.data import PairedVolumeDataset, collate_volumes, read_manifest
from flowoct.runner import environment, load_config, make_model, setup, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--network-chunk", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.network_chunk is not None:
        if args.network_chunk < 0:
            raise ValueError("network_chunk must be nonnegative")
        cfg["network_chunk"] = args.network_chunk
    device = setup(cfg)
    dataset = PairedVolumeDataset(read_manifest(cfg["manifest"], cfg["fov"]), "train", cfg, 1)
    batch = collate_volumes([dataset[0]])
    model = make_model(cfg, True)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    tic = time.perf_counter()
    model.set_input(batch)
    try:
        model.forward()
    except torch.cuda.OutOfMemoryError as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {"status": "CUDA_OUT_OF_MEMORY", "config": cfg,
                   "environment": environment(device), "error": str(error),
                   "seconds_until_failure": time.perf_counter() - tic})
        raise
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {"status": "PASS", "config": cfg, "environment": environment(device), "case_id": batch["metadata"][0]["case_id"],
              "input_shape": list(batch["A"].shape),
              "sampled_slices": cfg["train_slices"] or cfg["crop_size"] // 4,
              "network_chunk": cfg["network_chunk"], "seconds": time.perf_counter() - tic,
              "loss": model.loss.item(), **model.get_current_loss(),
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
              "generator_parameters": sum(p.numel() for p in model.netG.parameters())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
```

### `scripts/verify_resume.py`

```python
"""Compare two uninterrupted epochs with one epoch plus a complete resume."""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flowoct.runner import load_config, read_checkpoint, train, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    cfg.update(load_size=32, crop_size=32, epochs=2, workers=0,
               train_slices=0, save_every=1, val_every=1)
    limits = {"train": 1, "validation": 1}
    uninterrupted = train(cfg, args.output / "uninterrupted", limits=limits)
    gc.collect()
    interrupted = train({**cfg, "epochs": 1}, args.output / "resumed", limits=limits)
    gc.collect()
    train(cfg, interrupted, resume=interrupted / "latest.pt", limits=limits)
    first, second = [read_checkpoint(path / "latest.pt") for path in (uninterrupted, interrupted)]
    unequal = [name for name in first["model"] if not torch.equal(first["model"][name], second["model"][name])]
    optimizer_equal = True
    for index, state in first["optimizer"]["state"].items():
        for key, value in state.items():
            other = second["optimizer"]["state"][index][key]
            optimizer_equal &= torch.equal(value, other) if torch.is_tensor(value) else value == other
    result = {"status": "PASS" if not unequal and optimizer_equal else "FAIL",
              "config": cfg, "diagnostic_only": True, "epochs": 2,
              "unequal_model_tensors": unequal, "optimizer_equal": bool(optimizer_equal),
              "global_step_equal": first["global_step"] == second["global_step"] == 2,
              "best_validation_psnr_equal": first["best_psnr"] == second["best_psnr"],
              "comparison": "same GPU, real train/validation scans, 32^3; epoch-boundary resume"}
    write_json(args.output / "comparison.json", result)
    print(result, flush=True)
    if result["status"] != "PASS" or not result["global_step_equal"] or not result["best_validation_psnr_equal"]:
        raise AssertionError("Epoch-boundary replay differs")


if __name__ == "__main__":
    main()
```

### `scripts/visualize.py`

```python
"""Display model-space predictions as physical ZYX slices and axial MIP."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.input, allow_pickle=False) as data:
        if data["axis_order"].item() != "ZYX":
            raise ValueError("Expected axis_order=ZYX")
        oct_volume, target, prediction = [data[key] for key in ("oct", "octa_gt", "octa")]
    volumes = [oct_volume, target, prediction, np.abs(prediction - target)]
    names = ["OCT", "Real OCTA", "Generated OCTA", "Absolute error"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for column, (volume, name) in enumerate(zip(volumes, names)):
        # Select anatomical planes using saved axis metadata, not numeric lengths.
        planes = [volume[:, volume.shape[1] // 2, :],
                  volume[:, :, volume.shape[2] // 2], volume.max(axis=0)]
        if column == 3:
            planes[2] = np.abs(prediction.max(axis=0) - target.max(axis=0))
        for row, plane in enumerate(planes):
            axes[row, column].imshow(plane, cmap="magma" if column == 3 else "gray", vmin=0, vmax=1)
            label = ["B-scan (ZX)", "Orthogonal slice (ZY)", "Axial max projection (YX)"][row]
            axes[row, column].set_title(f"{name}\n{label}", fontsize=10)
            axes[row, column].axis("off")
    info_path = args.input.parent / "prediction.json"
    diagnostic = info_path.is_file() and json.loads(info_path.read_text(encoding="utf-8")).get("limited_diagnostic")
    prefix = "DIAGNOSTIC | " if diagnostic else ""
    fig.suptitle(prefix + args.input.stem + " | resized center crop; not original full FOV", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

### `scripts/build_provenance.py`

```python
"""Maintainer utility: build exact reference copies and adaptation patches."""

import difflib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    upstream = root.parent / "upstream"
    pinned = "2309126956d2b80a95f2d307d85a50799cc658a6"
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if head != pinned or status:
        raise RuntimeError("Upstream must be the pinned, unmodified author checkout")
    files = subprocess.check_output(["git", "-C", str(upstream), "ls-files"], text=True).splitlines()
    original_hashes = {}
    for relative in files:
        source = upstream / relative
        destination = root / "reference/upstream" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        original_hashes[relative] = digest(source)
    mappings = {"models/FlowOCT_model.py": "flowoct/model.py",
                "util/DWT_IDWT_layer.py": "flowoct/wavelets.py",
                "util/DWT_IDWT_Functions.py": "flowoct/wavelet_functions.py",
                "util/auto_crop.py": "flowoct/roi.py",
                "datasets/OCTA500_dataset.py": "flowoct/data.py",
                "requirements.txt": "requirements.txt"}
    for original, adapted in mappings.items():
        patch = "".join(difflib.unified_diff(
            (upstream / original).read_text(encoding="utf-8").splitlines(keepends=True),
            (root / adapted).read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile="a/upstream/" + original, tofile="b/" + adapted))
        (root / "provenance" / (Path(adapted).stem + ".patch")).write_text(patch, encoding="utf-8")
    current = {str(path.relative_to(root)).replace("\\", "/"): digest(path)
               for base in ("flowoct", "configs", "scripts", "tests")
               for path in sorted((root / base).rglob("*"))
               if path.is_file() and "__pycache__" not in path.parts}
    for name in ("run.py", "a100.py", "requirements.txt"):
        current[name] = digest(root / name)
    report = {"upstream_url": "https://github.com/cnu-medilab/FlowOCT", "upstream_commit": pinned,
              "upstream_clean": True, "upstream_files_sha256": original_hashes,
              "adapted_files_sha256": current,
              "copied_without_changes": ["flowoct/wavelet_functions.py", "flowoct/roi.py"],
              "new_workflow": ["flowoct/runner.py", "flowoct/metrics.py", "run.py", "a100.py", "configs/", "scripts/", "tests/"]}
    (root / "provenance/source_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    dataset = root.parent / "data/OCTA500_common_v1"
    shutil.copyfile(dataset / "protocol.json", root / "provenance/dataset_protocol.json")
    hashes = {name: digest(dataset / name) for name in ("split_3M_z256.csv", "split_6M_z256.csv")}
    (root / "provenance/split_manifest_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print("Reference, source fingerprints and patches written; upstream unchanged.")


if __name__ == "__main__":
    main()
```

### `tests/test_core.py`

```python
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from flowoct.data import PairedVolumeDataset, read_manifest
from flowoct.metrics import volume_metrics
from flowoct.runner import (ROOT, load_config, make_model, read_checkpoint,
                            restore_rng, save_checkpoint, setup)
from flowoct.wavelets import DWT_2D, IDWT_2D


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        z, y, x = np.indices((32, 32, 32), dtype=np.float32)
        self.array = (4 * z + 2 * y + x) / 217 * 255
        self.rows = []
        for index, split in enumerate(("train", "validation", "test")):
            cid = str(index + 1)
            path = self.root / (cid + ".npz")
            np.savez(path, oct=self.array, octa=255 - self.array, axis_order="ZYX")
            self.rows.append({"case_id": cid, "patient_id": cid, "path": path.name,
                              "resolved_path": str(path), "split": split, "fov": "3M",
                              "leakage_group": cid})
        self.cfg = load_config(ROOT / "configs/3m.json")
        self.cfg.update(load_size=32, crop_size=32)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self):
        path = self.root / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def test_manifest_rejects_patient_and_content_leakage(self):
        read_manifest(self.manifest(), "3M", check_counts=False)
        self.rows[2]["patient_id"] = "1"
        with self.assertRaisesRegex(ValueError, "Patient crosses"):
            read_manifest(self.manifest(), "3M", check_counts=False)
        self.rows[2]["patient_id"] = "3"
        self.rows[2]["leakage_group"] = "1"
        with self.assertRaisesRegex(ValueError, "image group"):
            read_manifest(self.manifest(), "3M", check_counts=False)

    def test_orientation_and_paired_minmax(self):
        sample = PairedVolumeDataset(self.rows, "validation", self.cfg)[0]
        expected = self.array.transpose(1, 2, 0) / 255
        np.testing.assert_allclose(sample["A"][0], expected, atol=2e-7)
        np.testing.assert_allclose(sample["A"] + sample["B"], 1, atol=3e-7)
        self.assertEqual(sample["geometry"]["model_axis_order"], "YXZ")
        self.assertFalse(sample["geometry"]["flipped_z"])

    def test_constant_volume_fails_before_normalization(self):
        np.savez(self.rows[1]["resolved_path"], oct=np.ones((32,) * 3), octa=np.ones((32,) * 3))
        with self.assertRaisesRegex(ValueError, "Constant crop"):
            PairedVolumeDataset(self.rows, "validation", self.cfg)[0]


class NumericalTests(unittest.TestCase):
    def test_haar_reconstruction_gradients_and_subband_signs(self):
        # These signs/order match the author's matrix implementation.
        x = torch.tensor([[[[1., 2.], [3., 4.]]]], dtype=torch.float64, requires_grad=True)
        bands = DWT_2D("haar")(x)
        np.testing.assert_allclose([band.item() for band in bands], [5., -1., -2., 0.], atol=1e-14)
        restored = IDWT_2D("haar")(*bands)
        torch.testing.assert_close(restored, x)
        restored.square().sum().backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())

    def test_metrics_are_finite_and_projection_uses_z(self):
        rng = np.random.default_rng(8)
        target = rng.random((8, 9, 10), dtype=np.float32)
        scores = volume_metrics(target, target)
        self.assertEqual(scores["volume_mae"], 0)
        self.assertAlmostEqual(scores["volume_ssim"], 1)
        self.assertAlmostEqual(scores["axial_mip_ssim"], 1)
        self.assertTrue(all(np.isfinite(list(scores.values()))))

    def config(self):
        cfg = load_config(ROOT / "configs/3m.json")
        cfg.update(device="cpu", threads=2, crop_size=32, load_size=32, network_chunk=0)
        setup(cfg)
        return cfg

    @staticmethod
    def update(model, batch):
        model.train()
        model.set_input(batch)
        model.forward()
        return {"total": model.loss.item(), **model.get_current_loss()}

    def test_chunk_recomputation_keeps_objective_and_gradients(self):
        cfg = self.config()
        batch = {"A": torch.rand(1, 1, 8, 32, 32), "B": torch.rand(1, 1, 8, 32, 32)}
        regular = make_model(cfg, True)
        # A warmup avoids testing only the initially zero output layer.
        self.update(regular, batch)
        chunked = make_model({**cfg, "network_chunk": 1}, True)
        chunked.load_state_dict(regular.state_dict())
        # Deep-copy optimizer state so the two updates cannot share moments.
        import copy
        chunked.optimizer_G.load_state_dict(copy.deepcopy(regular.optimizer_G.state_dict()))
        torch.manual_seed(59)
        loss_a = self.update(regular, batch)
        torch.manual_seed(59)
        loss_b = self.update(chunked, batch)
        for key in loss_a:
            self.assertAlmostEqual(loss_a[key], loss_b[key], delta=3e-5)
        # Batch-size-dependent FP32 reductions can leave ~1e-9 residual bias
        # gradients around GroupNorm. Adam can amplify those via its epsilon;
        # the invariant is the full objective/gradient, not bitwise weights.
        for key, parameter in regular.named_parameters():
            other = dict(chunked.named_parameters())[key]
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=5e-4)
            self.assertTrue(torch.isfinite(other).all())

    def test_complete_checkpoint_replays_next_update(self):
        cfg = self.config()
        model = make_model(cfg, True)
        batch = {"A": torch.rand(1, 1, 8, 32, 32), "B": torch.rand(1, 1, 8, 32, 32)}
        self.update(model, batch)
        generators = {"train": torch.Generator().manual_seed(5),
                      "validation": torch.Generator().manual_seed(6)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, model, cfg, 1, 1, 0.0, "test-manifest", generators)
            expected_random = (random.random(), np.random.rand(), torch.randperm(8, generator=generators["train"]))
            expected_loss = self.update(model, batch)
            payload = read_checkpoint(path)
            restored = make_model(cfg, True)
            restored.load_state_dict(payload["model"])
            restored.optimizer_G.load_state_dict(payload["optimizer"])
            restore_rng(payload["rng"])
            generators["train"].set_state(payload["loader_rng"]["train"])
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            self.assertTrue(torch.equal(torch.randperm(8, generator=generators["train"]), expected_random[2]))
            actual_loss = self.update(restored, batch)
            self.assertEqual(expected_loss, actual_loss)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_projection_roi_with_nonzero_y_is_finite(self):
        cfg = self.config()
        model = make_model(cfg, True)
        target = torch.zeros(1, 1, 8, 32, 32)
        # Content at row 16 has a strictly positive padded ROI origin. The
        # upstream [y,x] indexing slices the C=1 axis empty on this example.
        target[:, :, :, 16:20, 10:20] = 0.75
        losses = self.update(model, {"A": torch.zeros_like(target), "B": target})
        self.assertTrue(all(np.isfinite(list(losses.values()))))
        self.assertGreater(losses["proj_loss"], 0)


if __name__ == "__main__":
    unittest.main()
```


### `a100.py`

```python
"""A100 entry point: discover existing data and use the unchanged training loop."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def find_data(fov, explicit=None):
    """Locate CSV roots only; do not traverse large volume directories."""
    filename = f"split_{fov.upper()}_z256.csv"
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        bases = (Path.cwd(), ROOT, ROOT.parent, ROOT.parent.parent, Path.home())
        candidates = []
        for base in bases:
            candidates.extend((base, base / "OCTA500_v1", base / "OCTA500_common_v1",
                               base / "data/OCTA500_v1", base / "data/OCTA500_common_v1"))
    found = sorted({path.resolve() for path in candidates
                    if (path / filename).is_file()})
    if len(found) != 1:
        detail = "\n".join(str(path) for path in found) or "No matching CSV found."
        raise ValueError(
            f"Need exactly one data directory containing {filename}.\n{detail}\n"
            "Extract this project inside your data directory, or pass "
            "--data-dir /absolute/path/to/the/CSV/directory. "
            "A directory containing only native data is not supported.")
    data = found[0]
    if not (data / "z256" / fov.upper()).is_dir():
        raise FileNotFoundError(data / "z256" / fov.upper())
    return data


def make_config(fov, data, device):
    cfg = json.loads((ROOT / "configs" / f"{fov}.json").read_text(encoding="utf-8-sig"))
    expected = {"epochs": 200, "network_chunk": 0, "inference_chunk": 32,
                "train_slices": 0, "batch_size": 1, "crop_size": 256}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Baseline template changed; expected 200 epochs, 64 full slices and FP32.")
    cfg.update(manifest=str(data / f"split_{fov.upper()}_z256.csv"), device=device)
    return cfg


def write_config(path, cfg):
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def invoke(*arguments):
    command = [sys.executable, "-u", *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def check_gpu(cfg):
    # Exit the probe before training so its CUDA context does not occupy VRAM.
    source = """import json, sys
try:
    from flowoct.runner import setup
    import torch
except ImportError:
    raise SystemExit('Activate the FlowOCT environment and install requirements.txt first.')
cfg = json.loads(sys.argv[1])
device = setup(cfg)
free, total = torch.cuda.mem_get_info(device)
print(json.dumps({'device': str(device), 'gpu': torch.cuda.get_device_name(device),
                  'free_gib': round(free / 2**30, 2), 'total_gib': round(total / 2**30, 2),
                  'network_chunk': 0, 'sampled_slices': 64, 'epochs': 200}), flush=True)
"""
    subprocess.run([sys.executable, "-c", source, json.dumps(cfg)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train", "resume", "test"))
    parser.add_argument("--fov", choices=("3m", "6m"), default="3m")
    parser.add_argument("--data-dir", type=Path, help="CSV directory; normally found automatically")
    parser.add_argument("--device", default="cuda:0", help="Index within the GPUs allocated to you")
    parser.add_argument("--run-name", help="Default: flowoct_3m_200e or flowoct_6m_200e")
    args = parser.parse_args()
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("Use an allocated CUDA device, for example cuda:0")
    name = args.run_name or f"flowoct_{args.fov}_200e"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        parser.error("run-name must be a simple directory name")
    run = ROOT / "runs" / name

    if args.command in ("resume", "test"):
        if args.data_dir is not None:
            parser.error("Resume/test reuse the saved run configuration; omit --data-dir")
        saved = run / "config.json"
        cfg = json.loads(saved.read_text(encoding="utf-8"))
        if cfg["fov"] != args.fov.upper():
            raise ValueError("The saved run belongs to a different FOV")
        if cfg["network_chunk"] != 0 or cfg.get("diagnostic_limits"):
            raise ValueError("Use a formal unchunked run, not a chunked or limited diagnostic run")
        if args.command == "resume":
            invoke(ROOT / "run.py", "train", "--config", saved, "--device", args.device,
                   "--output", run, "--resume", run / "latest.pt")
        else:
            predictions = ROOT / "results" / name
            evaluation = ROOT / "evaluations" / name
            if predictions.exists() or evaluation.exists():
                raise FileExistsError("Test outputs already exist; they will not be overwritten")
            invoke(ROOT / "run.py", "predict", "--config", saved, "--device", args.device,
                   "--checkpoint", run / "best.pt", "--output", predictions)
            invoke(ROOT / "run.py", "evaluate", "--predictions", predictions, "--output", evaluation)
        return

    if args.command == "train" and run.exists():
        raise FileExistsError(f"Run already exists: {run}. Use resume or choose --run-name.")
    data = find_data(args.fov, args.data_dir)
    cfg = make_config(args.fov, data, args.device)
    print("Using existing data:", data, flush=True)
    check_gpu(cfg)
    if args.command == "check":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")
        output = ROOT / "verification" / f"a100_{args.fov}_{stamp}"
        output.mkdir(parents=True, exist_ok=False)
        config = output / "config.json"
        write_config(config, cfg)
        invoke(ROOT / "run.py", "audit", "--config", config, "--output", output / "audit.json")
        invoke(ROOT / "scripts/benchmark.py", "--config", config, "--output", output / "benchmark.json")
        invoke(ROOT / "run.py", "smoke", "--config", config, "--full-size", "--output", output / "fullsize")
        print("Checks passed. Diagnostic records:", output, flush=True)
        print(f"Start the separate 200-epoch run with: python a100.py train --fov {args.fov}", flush=True)
    else:
        config = ROOT / "configs" / f"{args.fov}_a100.json"
        write_config(config, cfg)
        invoke(ROOT / "run.py", "train", "--config", config, "--output", run)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"A100 launcher stopped: {error}")
```

### `tests/test_a100.py`

```python
"""Deployment checks: data selection and baseline preservation, without training."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import a100


class A100DeploymentTests(unittest.TestCase):
    def setUp(self):
        workspace = a100.ROOT.parent.resolve()
        self.temporary = tempfile.TemporaryDirectory(prefix="a100_test_", dir=workspace)
        self.root = Path(self.temporary.name).resolve()
        # Verify the cleanup target before TemporaryDirectory removes it.
        self.assertTrue(self.root.is_relative_to(workspace))
        self.assertNotEqual(self.root, workspace)

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, parent, name="OCTA500_v1"):
        data = parent / name
        (data / "z256/3M").mkdir(parents=True)
        (data / "split_3M_z256.csv").write_text("case_id,path,split\n", encoding="utf-8")
        return data

    def test_extracting_inside_renamed_data_directory_needs_no_native(self):
        data = self.dataset(self.root)
        project = data / "flowoct_reproduction"
        project.mkdir()
        with patch.object(a100, "ROOT", project), patch.object(Path, "cwd", return_value=project), \
                patch.object(Path, "home", return_value=self.root):
            self.assertEqual(a100.find_data("3m"), data)
        self.assertFalse((data / "native").exists())

    def test_ambiguous_datasets_require_an_explicit_choice(self):
        first = self.dataset(self.root)
        self.dataset(self.root, "OCTA500_common_v1")
        project = self.root / "flowoct_reproduction"
        project.mkdir()
        with patch.object(a100, "ROOT", project), patch.object(Path, "cwd", return_value=project), \
                patch.object(Path, "home", return_value=self.root):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                a100.find_data("3m")
            self.assertEqual(a100.find_data("3m", first), first)

    def test_config_changes_only_manifest_and_device(self):
        data = self.dataset(self.root)
        original = json.loads((a100.ROOT / "configs/3m.json").read_text(encoding="utf-8"))
        adapted = a100.make_config("3m", data, "cuda:0")
        changed = {key for key in original if original[key] != adapted[key]}
        self.assertEqual(changed, {"manifest", "device"})
        self.assertEqual((adapted["network_chunk"], adapted["epochs"], adapted["train_slices"]), (0, 200, 0))

    def test_empty_upload_directory_is_rejected(self):
        empty = self.root / "OCTA500_v1"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No matching CSV"):
            a100.find_data("3m", empty)

    def test_resume_uses_saved_config_without_discovering_other_data(self):
        project = self.root / "flowoct_reproduction"
        run = project / "runs/flowoct_3m_200e"
        run.mkdir(parents=True)
        saved = run / "config.json"
        saved.write_text(json.dumps({"fov": "3M", "network_chunk": 0,
                                     "diagnostic_limits": None}), encoding="utf-8")
        before = saved.read_bytes()
        with patch.object(a100, "ROOT", project), patch("sys.argv", ["a100.py", "resume"]), \
                patch.object(a100, "find_data", side_effect=AssertionError("Must use saved data")), \
                patch.object(a100, "invoke") as invoke:
            a100.main()
        arguments = invoke.call_args.args
        self.assertIn(saved, arguments)
        self.assertIn(run / "latest.pt", arguments)
        self.assertEqual(saved.read_bytes(), before)

    def test_existing_run_is_not_overwritten(self):
        project = self.root / "flowoct_reproduction"
        (project / "runs/flowoct_3m_200e").mkdir(parents=True)
        with patch.object(a100, "ROOT", project), patch("sys.argv", ["a100.py", "train"]):
            with self.assertRaisesRegex(FileExistsError, "Run already exists"):
                a100.main()

    def test_diagnostic_or_chunked_run_cannot_be_resumed_as_formal(self):
        project = self.root / "flowoct_reproduction"
        run = project / "runs/flowoct_3m_200e"
        run.mkdir(parents=True)
        for chunk, limits in ((4, None), (0, {"train": 1, "validation": 1})):
            with self.subTest(chunk=chunk, limits=limits):
                (run / "config.json").write_text(json.dumps({"fov": "3M", "network_chunk": chunk,
                                                            "diagnostic_limits": limits}), encoding="utf-8")
                with patch.object(a100, "ROOT", project), patch("sys.argv", ["a100.py", "resume"]):
                    with self.assertRaisesRegex(ValueError, "formal unchunked"):
                        a100.main()


if __name__ == "__main__":
    unittest.main()
```

## 附录 C：这次审阅的源码指纹

这些指纹来自当前 `provenance/source_manifest.json`，包括新增A100入口和测试。模型、小波、数据加载、损失及训练循环的源码指纹保持不变；打包时逐项核对压缩包内文件。

```json
{
  "flowoct/__init__.py": "c5d9125502b98e89d98e94b99a0675b085dd010e9ae7e194d5199651125b8741",
  "flowoct/data.py": "cf503eeac9ae0ac2441b72a07ab7b1c7de4a6382f833ca59354a1b4d93d75415",
  "flowoct/metrics.py": "926e86aec2f33c283ac7f667ed7b9b1ff5ef710490a86f5f53f69c4214048efe",
  "flowoct/model.py": "9c4e1350dd2926120f7f00e2b569313c08369e6d893a45ae1d1a4f8a860dae1e",
  "flowoct/roi.py": "1c729065c592c28d1a86d328dbae7ebd883d2f7153c910d0db13f5a149dd30e0",
  "flowoct/runner.py": "11c55c27261c48e7cd37dfca8531ad5f3394fe1cd34c9102537d2cc23ec39ee6",
  "flowoct/wavelet_functions.py": "33623b3e9b8d59df0227a840f88db4789095d3449e820e6454a31499ed3d156f",
  "flowoct/wavelets.py": "02625d63ff997de3af1633f0fa511021d74212dabbc984162218e0bf8f594768",
  "configs/3m.json": "92391ed2db33b329132ddf8a1b0a0c61a4abca94de31722ee541165d90b15938",
  "configs/6m.json": "0cb9ca2c085f441741aaf988ca0e756d9a3108b18e9f39d85b94a6114d1c6bb2",
  "scripts/benchmark.py": "a479023d9c219e4763e37455276a0584ccfad965ca7be98025fa7dfe781014ef",
  "scripts/build_provenance.py": "1e6ccd56d64e7f823678839157d5f70c2477c7f4da43b51c471dbd3e1fc5797d",
  "scripts/run_experiment.sh": "774c4b929bfa0a896d85cede02a59322d6193cd3d63f994e5c5be480bb86f867",
  "scripts/setup_server.sh": "8c7a0e4e5d2f2c3fdc54471d3a8fda981841c3e358e2a281366df673ea4f60c7",
  "scripts/verify_resume.py": "d5b9b033b0207db9915a4ea7c6645f3d777f08034247677c3c82e1b74a31b2a5",
  "scripts/visualize.py": "5a414cde4349bf97fb94c9c69d174e2f928c0e91e8683d00b3fa88118687a7fd",
  "tests/test_a100.py": "f6023939250f497c5db8257886feb67c098d8cad36ab5dbe29a7b51597d29a93",
  "tests/test_core.py": "9108ab10416dca7d646a6db3f48a0f9b0678e14a0fcc9967bb3c3dd50e4b27f6",
  "run.py": "13f63572b7d4c05b8e0d4ba1cfd1134f04ba4b2eb7b0a09bfefb08634ea1fe73",
  "a100.py": "d5b18627fcf4ead55fd5952c01ec02cdbdb5b0bf3e505c227e7c30445bb541ee",
  "requirements.txt": "9fc5bd825019a6510f68783d8e85de08f5395cacfcf1aa226dd083920c7ef18d"
}
```
