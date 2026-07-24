# Equal Weight Multi V2（筛弱 / 去重等权）

在 **predictor 先验宇宙**上，每折仅用训练窗：按日均 RankIC 剔弱，再按横截面相关贪心去重，对入选因子等权合成 `prediction`，并按与 v1 相同的扩展窗口协议做样本外评价。

相对 [`equal_weight_multi_v1`](../equal_weight_multi_v1/)（92 字段全体等权地板基准）：

| 项 | v1 | v2 |
|----|----|----|
| 先验宇宙 | 注册表全部 92 字段 | 仅 `usage_type=predictor` |
| 因子选择 | 无 | 折内训练窗 RankIC + 相关去重 |
| 合成 | 等权 | 等权 |
| OOS 偷看 | 否 | 否 |

## 筛选规则（写死）

1. 先验：`candidate_usage_types = ["predictor"]`
2. 方向：与 v1 相同（catalog；`0` 则训练窗 RankIC 符号冻结）
3. 剔弱：方向对齐后训练窗日均 RankIC ≥ `min_train_rank_ic`（默认 0.01）
4. 去重：按训练窗 RankIC 从高到低贪心纳入；与已入选因子的训练窗日均 `|corr|` ≥ `max_pairwise_corr`（默认 0.7）则跳过
5. 合成：定向 → `cs_rank_centered` → `mean_available` 等权

入选名单**允许折间变化**；审计表见 `outputs/fold_selected_factors.csv`。

## 运行

```bash
set PYTHONPATH=src
py -m factor_modeling.equal_weight_multi --config modeling/equal_weight_multi_v2/config.json

# 或
py modeling/equal_weight_multi_v2/run.py

# 冒烟
py modeling/equal_weight_multi_v2/run.py --years 2019,2020
```

## 主要输出

- `等权多因子模型_团队规范提交报告.md`（**按团队提交流程模板填写**，含差异声明、OOS 指标与 v1 对照）
- `equal_weight_multi_report.md`（自动技术报告，含与 v1 对照表）
- `outputs/fold_selected_factors.csv`
- `outputs/oos_predictions.csv.gz`
- `outputs/strategy_daily_returns.csv`
- `outputs/fold_metrics.csv` / `yearly_metrics.csv`
- `outputs/universe_factors.csv`（先验名单）
- `outputs/run_summary.json`

## 源码

- [`src/factor_modeling/equal_weight_multi.py`](../../src/factor_modeling/equal_weight_multi.py)
  - `select_fold_universe()` / `mean_daily_abs_corr_matrix()`
