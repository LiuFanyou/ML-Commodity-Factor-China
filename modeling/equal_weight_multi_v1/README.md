# Equal Weight Multi V1（全体等权）

在事前固定的候选因子宇宙上，对全部字段做方向对齐与横截面可比化后等权合成单一 `prediction`，再按统一协议做扩展窗口样本外评价与多空回测。

本版是**无因子选择**的多因子地板基准：不估计回归系数、不做 Top-K、不枚举子集、不按回测表现挑因子。

口径对齐 `training_handoff/` 与统一实验协议（与 Ridge / 单因子排序同口径）。

## 与其他模型的差异

| 模型 | 本质 |
|------|------|
| 单因子排序 | 逐字段单独评价，输出排行榜 |
| Ridge | 在训练窗学习权重（回归系数），92 字段联合 |
| **本版全体等权** | 注册表全部字段等权平均，无挑选、无回归 |

## Handoff 接口

目录默认：`training_handoff/`

| 文件 | 作用 |
|------|------|
| `training_contract.json` | 训练契约 |
| `fold_definitions.csv` | 扩展窗口折定义（必须使用） |
| `ml_feature_registry.csv` | 默认使用**全部** `feature_name`（与 Ridge 一致，约 92 列） |
| `ml_features_development.csv.gz` | 特征矩阵 |
| `labels_development.csv.gz` | `future_return_5d`、`label_end_date_5d` 等 |

## 运行

```bash
set PYTHONPATH=src
py -m factor_modeling.equal_weight_multi --config modeling/equal_weight_multi_v1/config.json

# 或
py modeling/equal_weight_multi_v1/run.py

# 冒烟：只跑部分测试年
py modeling/equal_weight_multi_v1/run.py --years 2019,2020
```

## 模型逻辑摘要

1. 标签：`future_return_5d`（开盘→开盘）；`target_cs_rank_5d` 仅作质量对照。
2. 宇宙：`use_all_registered_features=true`，使用 handoff 注册表**全部 92** 字段（含 predictor / quality_flag / risk / conditional）。
3. 方向：目录 `expected_direction`；若为 0，仅在该折训练窗用日均 RankIC 符号定向并冻结。
4. 合成：逐因子定向 → `cs_rank_centered` → 对非缺失因子等权平均（`missing_policy=mean_available`）。
5. 策略：T 收盘信号、T+1 开盘；最高/最低 20% 多空；持有/调仓 5 日。
6. 成本：主口径单边 5bp；敏感性 0/3/5/10bp。
7. 验证：`fold_definitions.csv` 扩展窗口 2019–2024；净化 `label_end_date_5d < valid_start`。

## 主要输出

- `等权多因子模型_团队规范提交报告.md`（**按团队提交流程模板填写**，含差异声明与 OOS 指标）
- `equal_weight_multi_report.md`
- `outputs/oos_predictions.csv.gz`
- `outputs/strategy_daily_returns.csv`
- `outputs/fold_metrics.csv` / `yearly_metrics.csv`
- `outputs/universe_factors.csv`
- `outputs/fold_directions.csv`
- `outputs/daily_prediction_ic.csv`
- `outputs/handoff_summary.json`
- `outputs/run_summary.json`

## 源码位置

- `src/factor_modeling/equal_weight_multi.py`
- 共享回测：`src/factor_modeling/common.py`
- Handoff：`src/factor_modeling/handoff.py`
