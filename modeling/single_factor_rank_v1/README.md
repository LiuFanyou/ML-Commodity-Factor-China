# Single Factor Rank V1

对每个注册字段单独做横截面排序信号，并按统一协议做扩展窗口样本外多空评价。不做多因子加权或回归。

口径对齐 `training_handoff/training_contract.json` 与统一实验协议；**候选字段默认与 Ridge 一致：使用 `ml_feature_registry.csv` 全部 `feature_name`（当前 92 个）**。

## Handoff 接口

目录默认：`training_handoff/`

| 文件 | 作用 |
|------|------|
| `training_contract.json` | 训练契约 |
| `fold_definitions.csv` | 扩展窗口折定义（必须使用） |
| `ml_feature_registry.csv` | 默认使用全部 `feature_name`（与 Ridge 一致，约 92 列） |
| `ml_features_development.csv.gz` | 特征矩阵 |
| `labels_development.csv.gz` | `future_return_5d`、`label_end_date_5d` 等 |

主键：`trade_date + product`。

## 运行

```bash
# Windows
set PYTHONPATH=src
py -m factor_modeling.single_factor_rank --config modeling/single_factor_rank_v1/config.json

# 或使用本目录入口脚本
py modeling/single_factor_rank_v1/run.py

# 冒烟：只跑前 3 个候选字段
py -m factor_modeling.single_factor_rank --config modeling/single_factor_rank_v1/config.json --limit 3
```

## 模型逻辑摘要

1. 标签：`future_return_5d`（同一真实合约开盘→开盘）；`target_cs_rank_5d` 仅作可选质量指标。
2. 分数：`signed_signal = factor * direction`，再按日做 `cs_rank_centered` 映射到 `[-1, 1]`。
3. 方向：优先用因子目录 `expected_direction`（经 `factor_no` 映射）；若为 0，仅在该折训练窗内用日均 RankIC 符号冻结方向。
4. 策略：T 收盘信号、T+1 开盘成交；最高/最低 20% 多空；持有/调仓 5 日。
5. 成本：主口径单边 5bp；敏感性 0/3/5/10bp。
6. 验证：`fold_definitions.csv` 扩展窗口（2019–2024）；净化 `label_end_date_5d < valid_start`。
7. **主输出是每个因子各自的指标**（不是全体中位数）；默认**不**再输出 TopK 短名单。
8. 展示资产：运行 `py modeling/single_factor_rank_v1/make_visuals.py` 生成 `outputs/visuals/` 图与 `outputs/tables/` 有序表；总说明见 `outputs/visuals/VISUALS_AND_TABLES.md`。

## 与旧版差异

| 项 | 旧口径 | 现口径 |
|---|---|---|
| 标签 | `future_return_1d` | `future_return_5d` |
| 切分 | 滚动 3 年（含 2018） | 扩展窗口 2019–2024 |
| 策略 | 每日换仓 | 5 日持有/调仓 |
| 成本 | 1/3/5 bp | 0/3/5/10 bp |
| 注册表 | `source_factor_id` | 兼容 `factor_no` |
| 结果汇总 | 容易误读为中位数 | 逐因子指标 + 对照短表（非强制 Top3） |

## 主要输出

- `单因子模型_团队规范提交报告.md`（**按团队提交流程模板填写**，含差异声明与 OOS 指标）
- `单因子排序模型报告.md`（模型说明 + 结果解读）
- `SINGLE_FACTOR_RESULTS_REPORT.md`（英文文件名的结果分析版）
- `single_factor_rank_report.md`（自动明细宽表）
- `outputs/visuals/`（结果图；说明见 `图表说明.md` / `VISUALS_AND_TABLES.md`）
- `outputs/tables/`（有序表与短表，含 `SHORTLISTS.md`；说明见 `visuals/图表说明.md`）
- `outputs/figures/`（旧目录，可忽略；新图在 `visuals/`）
- `outputs/leaderboard.csv` / `factor_strategy_metrics.csv` / `strategy_daily_returns.csv` / `run_summary.json`
- `outputs/per_factor_metrics.json`

## 源码位置

- `src/factor_modeling/single_factor_rank.py`
- 共享回测：`src/factor_modeling/common.py`
- Handoff 加载：`src/factor_modeling/handoff.py`
- 图表短表：`modeling/single_factor_rank_v1/make_visuals.py`
