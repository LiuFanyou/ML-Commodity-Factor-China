# 2025 模型回测数据包

给已在 `training_handoff`（2015—2024 开发集）上训练好的模型做 **2025 年回测**。

## 必用文件

| 文件 | 用途 |
|---|---|
| `ml_features_2025.csv.gz` | 2025 年模型输入特征（7290 行 = 交易日×品种） |
| `labels_2025_evaluator_only.csv.gz` | 2025 年标签与可成交标记（主标签 `future_return_5d`） |
| `ml_feature_registry.csv` | 合法特征名清单（须使用全部 `feature_name`） |
| `evaluation_contract_2025.json` | 2025 评价合同（切分、标签、执行口径） |

## 辅助文件

- `feature_dictionary.md`：特征含义
- `training_contract.json` / `PROTOCOL_AMENDMENT.md` / `UNIFIED_EXPERIMENT_PROTOCOL_SNAPSHOT.md`：与开发期同一套协议
- `example_backtest_2025.py`：读入、对齐、按统一协议做多空回测的最小示例
- `example_train_reference.py`：开发期训练接口参考（本包不含 2024 及以前训练数据）
- `factor_screening_2025_reference.md`：单因子 2025 参考报告（非 ML 必用）

## 使用步骤

1. 用开发期交接包训练模型；特征列 = `ml_feature_registry.csv` 的全部 `feature_name`。
2. 读取本包 `ml_features_2025.csv.gz`，按相同列生成每日每品种预测分数 `score`。
3. 与 `labels_2025_evaluator_only.csv.gz` 按 `trade_date, product` 内连接。
4. 过滤：`future_return_5d` 非空，且入场/退出可执行标记允许（见标签文件中的 execution / lock 字段）。
5. 按统一执行协议回测：
   - 信号：T 日收盘（特征日）
   - 调仓：每 5 个交易日
   - 组合：预测分最高 20% 做多、最低 20% 做空，组内等权，多空名义各 0.5
   - 收益：用 `future_return_5d`
   - 成本：默认单边 5bps；建议同时报告 0/3/5/10bps

## 不要做的事

- 不要用 2025 重新筛选特征或重调超参后再把同一套 2025 当最终成绩。
- 不要增删 `ml_feature_registry.csv` 以外的输入列。
- 本包不含原始 `data_final_raw`；只需模型推理与回测时不必再要原始行情。

## 快速检查

```bash
python example_backtest_2025.py
```

若未提供预测文件，脚本会用横截面随机分数跑通接口（仅验证打包完整性，不是策略结果）。
将模型预测存成 `predictions_2025.csv`（列：`trade_date,product,score`）后再运行，即可得到 2025 回测摘要。
