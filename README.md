# Stochastic Hopf early-warning experiments — seed 2026

这是论文 **Inverse modelling of stochastic Hopf dynamics for early warning of stepping-cycle anomalies** 的实验复现材料包。它把主评价、多事件窗口、零噪声干预、训练消融、beta/frequency/sigma 参数回放及其代码和汇总结果放在同一目录，并按随机流分开标记。

## 使用哪一组数字

- **文章主评价与零噪声比较**：看 `results/primary_seed2026/noise_by_horizon.csv`。未来 2 秒目标下，学习噪声为 **84/91、160 次报警、0.680 秒、Brier 0.23675**；关闭噪声为 **83/91、302 次、1.295 秒、Brier 0.28809**。两者来自 seed-2026 主评价流，batch size 32。
- **训练消融和参数敏感性参照**：看 `results/component_replay_seed2026/evaluation/summary.csv` 与 `results/parameter_sensitivity_seed2026/`。完整模型回放是 **83/91、166 次报警、0.685 秒、Brier 0.23655**，使用 seed-2026 的独立 batch-16 回放。
- 根目录 `results/canonical_comparison.csv` 将这些结果放在一张表里，但保留 `evaluation_stream` 列；不能把两条随机路径流当成一次运行。
- 之前 seed-20261027 的结果被放进 `results/legacy_seed20261027/`，明确标为历史，不进入当前结果比较。

## 统一实验协议

21 条记录、7 名参与者；按时间顺序划分为 52% 拟合、13% 验证、35% 测试。神经网络输入为 1T（1.13 秒）双通道力历史；Hopf 方程向前模拟 5 秒，每个起点 128 条路径，积分步长 0.01 秒、参数步长和滚动起点间隔 0.05 秒。主要事件目标为未来 2 秒内出现新的周期异常，并用同一组路径计算 1、2、3、5 秒风险。报警要求风险严格大于 0.80 且连续两个起点满足。周期标签按相邻有效峰间隔大于 1.5T（D=1.695 秒），不再额外合并相邻事件。详细参数在 `protocol/canonical_protocol.json`。

## 实验内容

1. `results/primary_seed2026/`：同一批主模型路径的 1/2/3/5 秒结果、零扩散对照、Brier 独立核查、每起点风险/标签/报警、路径首次事件时间及随机状态。
2. `results/baselines_seed2026/`：两秒主评价基线模型、事件/报警汇总与评分审计。
3. `results/component_replay_seed2026/`：无事件损失、未来常系数和直接风险 MLP 的同协议比较。
4. `results/parameter_sensitivity_seed2026/`：冻结 beta、频率和 sigma 曲线的配对回放。固定 sigma 仍有随机扩散；零噪声结果在主评价目录。
5. `scripts/`：主模型、评价、消融和敏感性分析脚本；`SHA256SUMS.txt` 用于校验包内文件。
6. `results/decoder_limit_cycle_audit_20261030/`：逐记录检查 21 个主模型解码矩阵的满秩性、条件数，并比较潜在圆极限环及其原始双通道受力空间椭圆像。检查结论是矩阵均可逆，但条件数中位数约 69、最高约 435，反演对压缩方向误差敏感；可逆仿射映射保持确定性稳定性类型和 Floquet 乘子不变。图中另区分了冻结参数的确定性环与保存的 128 路集合均值。中文说明见该目录 `REPORT_中文.md`。

## 复现与数据

结果表、风险、标签、报警、RNG 状态和首次事件时间已随包提供。为了让 GitHub 仓库保持轻量，未复制原始力数据、模型权重和服务器上约 4 GB 的逐路径力数组。原始数据来源链接和模型/代码校验信息见论文及结果目录。重新模拟时，需要先取得源数据和对应 checkpoint，再按脚本参数传入 source/reference 目录；当前包中的汇总分析可直接从已保存 CSV 重算。

主运行入口见 `scripts/primary_seed2026/run_unified.py`，控制实验入口见 `scripts/component_replay/`，参数敏感性入口见 `scripts/parameter_sensitivity/`。实际服务器环境为 PyTorch 2.6.0+cu124、CUDA 12.4；逐批 RNG 状态只保证在相同代码、设备、批大小和数值设置下复现。

解码矩阵审计入口见 `scripts/decoder_limit_cycle_audit/analyze_affine_cycles.py`。复算需要向 `--checkpoint-root` 提供每条记录的 `model.pt` 和 `predictions.npz`；渲染图需要本地安装 NumPy、Matplotlib。为控制包体积，检查点和单条随机路径仍未打包，现有审计汇总、映射后的均值轨迹和图已包含在结果目录。

## 校验和 GitHub

运行 `python validate_package.py` 可检查共享协议与关键汇总值；运行 `python verify_checksums.py` 可验证文件完整性；运行 `python generate_checksums.py` 可重建 SHA256 清单。清单逐一覆盖包内文件（不含清单自身）。尚未添加开源许可证；公开仓库前需由作者选择许可，并确认公开模型权重和数据副本的分发条件。

The author-supplied manuscript PDF is cited but not redistributed in this package. Raw force data, trained model checkpoints, and full decoded path arrays are also omitted.

Checksums cover the published files. Git metadata, Python caches, local environments, and input files in the ignored data/checkpoint directories are excluded. Text files use LF line endings so the checksums also apply to a fresh Git clone or downloaded source archive.

