# Linear Ridge V1

Ridge（L2 正则）线性回归模型。**只消费 training-handoff 矩阵接口**，不在本模块内加工因子或标签。

口径对齐 `training_handoff/training_contract.json` 与统一实验协议。

## Handoff 接口

目录默认：`training_handoff/`

| 文件 | 作用 |
|------|------|
| `training_contract.json` | 训练契约；模型 config 中的文件名优先 |
| `fold_definitions.csv` | 扩展窗口折定义（必须使用） |
| `ml_feature_registry.csv` | 必含 `feature_name`；使用全部注册字段 |
| `ml_features_development.csv.gz` | 特征矩阵：`trade_date, product, sector` + 全部注册字段 |
| `labels_development.csv.gz` | 标签：`future_return_5d`、`label_end_date_5d` 等 |

主键：`trade_date + product`（唯一）。

加载与校验实现：`src/factor_modeling/handoff.py`。

## 模型协议

- 主标签：`future_return_5d`（同一真实合约开盘→开盘）
- 训练目标：同日横截面中心化排名 `target_cs_rank_5d`
- 外层：读取 `fold_definitions.csv` 扩展窗口（2015 起点 → 2019–2024 验证）
- 净化：`label_end_date_5d < valid_start`
- 内层：训练窗最后一年选 `alpha`
- 策略：T 收盘信号、T+1 开盘成交；最高/最低 20% 多空；持有/调仓 5 日
- 成本：主口径单边 5bp；敏感性 0/3/5/10bp

## 运行

```bash
set PYTHONPATH=src
python -m factor_modeling.linear_ridge --config modeling/linear_ridge_v1/config.json
```

或：

```bash
python modeling/linear_ridge_v1/run.py
```

## 更换 handoff 因子包

1. 准备新目录，放入同 schema 文件（含 `fold_definitions.csv`）。
2. 修改 `config.json` 的 `handoff_directory`。
3. 重新运行上述命令。

## 主要输出

- `线性回归模型_团队规范提交报告.md`（**按团队提交流程模板填写**，含差异声明与 OOS 指标）
- `linear_ridge_report.md`
- `outputs/figures/`（结果图 + `FIGURES.md` 说明；由 `make_visuals.py` 生成）
- `outputs/handoff_summary.json`
- `outputs/oos_predictions.csv.gz`
- `outputs/strategy_daily_returns.csv`
- `outputs/fold_metrics.csv`
- `outputs/fold_definitions_used.csv`
- `outputs/alpha_search.csv`
- `outputs/coefficients_by_fold.csv`
- `outputs/coefficient_summary.csv`
- `outputs/daily_prediction_ic.csv`
- `outputs/run_summary.json`
