# FlowOCT：测试200轮最佳模型，然后续训到400轮

本流程面向你现有的A100目录和lyh_flowoct环境。新增的唯一运行脚本是FlowOCT_test200_then_train400.sh；它调用已有run.py，不修改模型、损失、数据预处理、FP32、64张整批方式或三步推理。

2026-10-24修正：所有Python步骤现在直接使用 `/workspace/.conda/envs/lyh_flowoct/bin/python`，不依赖终端当前是否激活该环境。启动时先检查训练依赖；终端显示base也可执行。脚本提示改为英文，避免当前终端把中文显示为下划线。

## 上传与启动

将本地文件：

D:/data/code/keyan_fuxian/FlowOCT/FlowOCT_test200_then_train400.sh

上传到服务器现有的flowoct_work/uploads文件夹。在训练用的GPU终端中操作。

需要新建tmux时，先执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate lyh_tmux
tmux new -s flowoct400
```

进入tmux后执行：

```bash
cd /workspace/flowoct_work/flowoct_reproduction_a100
bash /workspace/flowoct_work/uploads/FlowOCT_test200_then_train400.sh
```

按Ctrl+B，松手，再按小写d离开会话；任务继续运行。重新进入：

```bash
conda activate lyh_tmux
tmux attach -t flowoct400
```

脚本按顺序执行：检查训练环境与200轮模型 → best测试全部50例 → 评价与打包 → 复制完整状态到新目录 → 从201轮继续训练到400轮。任何步骤失败都会停止后续步骤。已有同名预测、评价、400轮目录或结果包时也会停止，不覆盖。

如果上次因缺少torchio而在导入阶段退出，只留下了测试日志，修正版会在环境检查通过后将旧日志改名为 `.previous_时间戳_进程号` 并保留，然后重新测试。它不会自动删除或改名已有预测目录。此次报错发生在模型与数据加载之前，没有完成任何测试病例，也没有启动400轮续训。

测试使用原best.pt，按本次日志应为193轮，并会与epochs.jsonl核对。续训使用原latest.pt，即第200轮结束状态，恢复Adam及随机状态，不从第193轮回退重练。

## 输出位置

| 内容 | 位置 |
| --- | --- |
| 原200轮实验，保持原样 | /workspace/flowoct_work/flowoct_reproduction_a100/runs/flowoct_3m_200e |
| 200轮预算内最佳模型的50例预测 | /workspace/flowoct_work/flowoct_reproduction_a100/results/flowoct_3m_best_after200 |
| 测试汇总与逐例指标 | /workspace/flowoct_work/flowoct_reproduction_a100/evaluations/flowoct_3m_best_after200 |
| 新400轮实验 | /workspace/flowoct_work/flowoct_reproduction_a100/runs/flowoct_3m_400e_from200 |
| 本次测试日志 | /workspace/flowoct_work/logs/flowoct_3m_best_after200_test.log |
| 201–400轮训练日志 | /workspace/flowoct_work/logs/flowoct_3m_201_to_400_train.log |

400轮是你指定的延长训练实验；原论文预算下的200轮结果独立保留。新目录最初继承200轮历史和原best，之后仍按验证集平均体积PSNR选模。

进入续训后，另一个终端可查看：

```bash
tail -n 30 -f /workspace/flowoct_work/logs/flowoct_3m_201_to_400_train.log
```

Ctrl+C退出tail查看，不停止训练。

## 测试后下载给我什么

脚本第3步结束后，以下文件会出现在uploads目录；可以在后续训练运行时下载：

- /workspace/flowoct_work/uploads/flowoct_3m_best_after200_review.tar
- /workspace/flowoct_work/uploads/flowoct_3m_best_after200_review.tar.sha256

保存到D:/data/FSdata。无需自行解包；告诉我下载完成即可。tar可能为数GB，包含全部50例三维数据；NPZ本身已压缩，所以此处采用tar打包。

包内包含：50例NPZ（预测OCTA、真实OCTA、输入OCT）、prediction.json、summary.json、per_case.csv、原训练配置/病例清单/环境/轮次记录、此次best.pt、原训练日志与本次测试日志。

这些内容足以核对实际测试模型、病例完整性、体积指标、疾病分组，并重新计算FULL平均投影，与此前98轮同病例做配对比较。当前run.py自动生成的axial_mip指标仍是最大投影；FULL平均投影将在收到三维结果后单独计算。

## 如果400轮训练中断

若已启动400轮训练，中断后在没有另一个同实验训练进程运行时，进入同一项目目录和环境，使用新目录的latest恢复。无需重跑整套测试脚本：

```bash
/workspace/.conda/envs/lyh_flowoct/bin/python -u run.py train \
  --config runs/flowoct_3m_400e_from200/config.json \
  --device cuda:0 \
  --output runs/flowoct_3m_400e_from200 \
  --resume runs/flowoct_3m_400e_from200/latest.pt \
  --epochs 400 \
  2>&1 | tee -a /workspace/flowoct_work/logs/flowoct_3m_201_to_400_train.log
```

## 验证范围

已在本地检查Bash语法，并使用模拟CLI及模拟checkpoint检查：顺序执行、50例结果打包、原200轮文件不变、新目录从200轮恢复并指定400轮、预测失败停止、已有输出停止、未完成200轮停止、错误当前Python环境下仍使用指定解释器、失败日志保留后重试、训练依赖缺失时在创建测试输出之前停止、解释器不存在时停止。模拟检查没有执行真实模型推理或GPU训练。

A100上的实际测试与续训由上述启动命令执行。

