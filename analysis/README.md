# FlowOCT 复现结果总览

[返回首页](../README.md)

各目录保留对应实验的报告、数字记录及结果图；有二维投影数组时也一并收录。

| 实验或分析 | 内容 |
| --- | --- |
| [flowoct_3m_100e_20261022](flowoct_3m_100e_20261022/) | 100 轮训练趋势 |
| [flowoct_3m_200e_20261024](flowoct_3m_200e_20261024/) | 200 轮训练日志解析与验证曲线 |
| [flowoct_3m_best_test_20261022](flowoct_3m_best_test_20261022/) | 第 98 轮：50 例测试与 FULL 投影 |
| [flowoct_3m_best193_test_20261024](flowoct_3m_best193_test_20261024/) | 第 193 轮：50 例测试、逐例指标、配对图和二维数组 |
| [flowoct_trainval_results_20261025](flowoct_trainval_results_20261025/) | 70/140/313 轮：固定训练 10 例与验证 10 例 |
| [flowoct_singlecase_results_20261026](flowoct_singlecase_results_20261026/) | 单病例 400 轮拟合及训练路径检查 |
| [flowoct_fixed_plane_control_20261027](flowoct_fixed_plane_control_20261027/) | 固定单切片对照：400/800/2000 步 |
| [flowoct_blur_audit_20261025](flowoct_blur_audit_20261025/) | 细节模糊与原始像素检查 |
| [flowoct_paper_audit_20261025](flowoct_paper_audit_20261025/) | 论文及公开实现核查、损失区域与方向检查 |
| [flowoct_validation_projection_plan_20261025](flowoct_validation_projection_plan_20261025/) | 验证投影方案与诊断参数 |
| [flowoct_comparison_20261028](flowoct_comparison_20261028/) | 代码逐行对照、三列比较表、实际运行指纹与指标核对 |

## 推荐阅读顺序

1. 第 193 轮 50 例测试报告，结合第 98 轮配对比较。
2. 70/140/313 轮训练与验证诊断，以及单病例和固定切片对照。
3. 最新三列对照表和实际运行指纹，确认指标对应的代码。

FULL、最大投影、三维体积使用不同评价对象；完整定义见各实验报告。
