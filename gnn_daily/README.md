# 日频 HCGNN 期货投资复现

> 状态更新（2026-07-23）：`training_handoff(1).zip` 的 38 维训练属于已完成的历史
> 基准，结果保留但不再代表下一轮正式口径。下一轮使用 `training_handoff(2).zip`：
> 57 个登记概念因子中 47 个可用，展开为 92 个模型输入列，数据截止 2023-12-31。
> GNN 与旧团队协议的差异见 `model_protocol/GNN_PROTOCOL_AMENDMENT.md`，提交格式见
> `GNN_SUBMISSION_REPORT_TEMPLATE.md`。新数据尚未正式重训。

## 历史协议版 GNN（已训练基准）

- 数据：交接包中的 38 个因子、30 个商品期货品种，截止 2023-12-31。
- 主标签：`future_return_5d`。
- 输入：过去 20 个交易日的因子历史，形成 30 节点动态图。
- 滚动折：协议定义的 2018–2023 六个验证年；2024/2025 暂不使用。
- `protocol_gnn`：只学习五日收益，作为协议口径的主 GNN。
- `hcgnn_continual`：在相同因子输入上按 CPD→GAP→MA→五日收益持续学习，作为论文方法扩展和消融对照。
- 特征统计、目标统计均只使用当前折训练期；训练标签必须在验证期开始前结束。

运行：

```powershell
python prepare_handoff.py --config config_handoff.yaml
python train_handoff_gnn.py --config config_handoff.yaml --fold fold_2018 --variant protocol_gnn
python train_handoff_gnn.py --config config_handoff.yaml --fold fold_2018 --variant hcgnn_continual
```

这是对 Hu et al., *Graph Portfolio: High-Frequency Factor Predictor via Heterogeneous Continual GNNs* 的日频适配实现。代码读取仓库中的 Tushare 中国期货日行情，构造论文列出的 49 个品种的主力连续序列，生成无未来泄漏的滚动时间折，并训练四任务 HCGNN。

## 日频适配

- 输入：过去 60 个交易日的结算价序列，各品种仅使用当前 fold 训练期统计量做 Z-score。
- 图：每个批次根据节点表示通过 self-attention 动态学习品种间邻接矩阵。
- 时空编码：两个带残差连接的谱域时空块；空间部分执行归一化图拉普拉斯的 GFT、可学习谱滤波和 IGFT，时间部分执行 DFT、Conv1D、GLU 和 IDFT。
- 连续任务顺序：CPD 分类 → 20 日 GAP 回归 → 40 日 MA 回归 → 下一日收益预测。
- 抗遗忘：完成每个任务后，以两种带噪视图的 InfoNCE 梯度估计参数重要度，后续任务使用论文式二次惩罚保护重要参数。
- 投资：按下一日预测收益做多最高的 K 个品种、做空最低的 K 个品种，计入换手成本。

CPD 完整执行论文公式 (4)–(6)：对未来 60 日窗口使用 `ruptures.Dynp(model="l2")` 最小化各分段平方损失，再取第一个未来变点及其后续分段；若该分段相对变点价格的最大涨幅超过 5%，标签为 1（做多），否则为 0（做空）。论文没有公开断点数、最小分段长度和日频阈值，默认显式设为 3 个断点、最短 5 日、阈值 5%，均可在 `config.yaml` 中修改。最终价格任务预测下一交易日收盘价；投资阶段再根据预测价相对当前收盘价的变化生成多空信号。

## 数据切分

默认使用 expanding-window rolling validation：

| Fold | 训练 | 验证 | 测试 |
|---|---|---|---|
| 0 | 2015–2020 | 2021 | 2022 |
| 1 | 2015–2021 | 2022 | 2023 |
| 2 | 2015–2022 | 2023 | 2024 |
| 3 | 2015–2023 | 2024 | 2025 |

最长辅助任务会查看未来 60 个交易日。因此每个 split 尾部会 purge 60 个交易日；标准化参数只由该 fold 的训练样本估计。每折还要求品种在训练、验证、测试三个区间的收盘价覆盖率均不低于 90%，从而保证固定节点图不会因新上市品种产生大面积缺失。

## 运行

在本目录执行：

```powershell
python prepare_data.py --config config.yaml
python train.py --config config.yaml --fold 0
```

正式配置按论文设置最大 100 epochs，并根据验证损失早停。依次训练全部滚动折可运行：

```powershell
powershell -ExecutionPolicy Bypass -File run_all_folds.ps1
```

先快速验证完整管线：

```powershell
python train.py --config config.yaml --fold 0 --smoke
```

切分结果位于 `artifacts_cpd_dynp/prepared/rolling_folds.csv`，每折的节点与边界记录在相邻 JSON 中。收益目标训练输出到 `artifacts_return_cpd_dynp`。旧版简化 CPD 的 `artifacts` 和 `artifacts_return` 会保留，不会被新版实验覆盖。

## 重要说明

- `A.DCE` 这类代码是主力连续序列。代码通过 `fut_basic_all.csv` 的真实品种/交易所映射选择节点，因而不会把 `AL.DCE`（豆一长期连续序列）误识别为沪铝。
- CPD 标签依赖 `ruptures` 的精确动态规划；`cpd_jump: 1` 不跳过候选断点。标签生成按品种并行，并在四个相同节点面板的 rolling folds 间缓存复用。
- 论文的 0.1% 成本在日频上很高，配置中保留该默认值，建议同时报告 1–5bp 的敏感性分析。
- 若用于正式研究，建议进一步加入成交量门槛、涨跌停不可交易约束、保证金占用、主力换月成交成本和 walk-forward 超参数选择。

## 正式提交图表与打包

十项正式训练全部完成后运行：

```bash
python build_submission_bundle.py \
  --artifacts artifacts_handoff_gnn_v2 \
  --config config_handoff_v2.yaml \
  --bundle-dir submission_hcgnn_v2
```

脚本会生成 12 类 PNG/PDF 图、所需指标/净值/特征 CSV、预测、持仓、成本敏感性、训练历史、实验配置、Markdown 报告和 `submission_hcgnn_v2.zip`。默认要求两个版本和 2019—2023 十个输出全部存在；`--allow-incomplete` 只能用于排版预览，不能用于正式提交。
