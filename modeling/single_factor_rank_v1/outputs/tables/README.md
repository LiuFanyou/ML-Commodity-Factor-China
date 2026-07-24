# 单因子排序有序结果表

- 字段数：92
- 样本外口径：future_return_5d，扩展窗口 2019–2024，5 日持有/调仓
- 对照短表说明见 `SHORTLISTS.md`

## 文件清单

- `modeling\single_factor_rank_v1\outputs\tables\01_all_by_rank_ic.csv`
- `modeling\single_factor_rank_v1\outputs\tables\02_predictor_by_rank_ic.csv`
- `modeling\single_factor_rank_v1\outputs\tables\03_all_by_gross_sharpe.csv`
- `modeling\single_factor_rank_v1\outputs\tables\04_predictor_by_net5bps_sharpe.csv`
- `modeling\single_factor_rank_v1\outputs\tables\05_by_usage_conditional_rank_ic.csv`
- `modeling\single_factor_rank_v1\outputs\tables\05_by_usage_predictor_rank_ic.csv`
- `modeling\single_factor_rank_v1\outputs\tables\05_by_usage_quality_flag_rank_ic.csv`
- `modeling\single_factor_rank_v1\outputs\tables\05_by_usage_risk_rank_ic.csv`
- `modeling\single_factor_rank_v1\outputs\tables\06_shortlist_top10_by_rank_ic.csv`
- `modeling\single_factor_rank_v1\outputs\tables\07_shortlist_top10_by_net5bps_sharpe.csv`
- `modeling\single_factor_rank_v1\outputs\tables\08_shortlist_top10_by_stability.csv`
- `modeling\single_factor_rank_v1\outputs\tables\09_shortlist_overlap.csv`
- `modeling\single_factor_rank_v1\outputs\tables\SHORTLISTS.md`

## Predictor 按 RankIC 前 15（摘录）

| rank | feature | RankIC | hit | gross Sharpe | net5 Sharpe |
| --- | --- | --- | --- | --- | --- |
| 10 | `carry_annualized__csz` | 0.0291 | 0.547 | 0.279 | -0.272 |
| 11 | `basis_fresh__csz` | 0.0285 | 0.557 | 0.463 | -0.168 |
| 12 | `warehouse_tightness__csz` | 0.0230 | 0.540 | 0.831 | 0.053 |
| 13 | `information_divergence__csz` | 0.0227 | 0.539 | 0.284 | -0.387 |
| 14 | `inventory_price_divergence__csz` | 0.0214 | 0.546 | 0.313 | -0.551 |
| 15 | `term_structure_slope__csz` | 0.0204 | 0.546 | 0.107 | -0.388 |
| 16 | `nonlinear_tightness__csz` | 0.0197 | 0.524 | 0.539 | -0.150 |
| 17 | `tightness_composite__csz` | 0.0194 | 0.525 | 0.523 | -0.166 |
| 18 | `information_consistency__csz` | 0.0190 | 0.537 | 0.322 | -0.367 |
| 19 | `short_reversal_5__csz` | 0.0182 | 0.526 | 0.438 | -0.033 |
| 20 | `inventory_acceleration__csz` | 0.0178 | 0.544 | 0.195 | -0.568 |
| 21 | `term_structure_acceleration__csz` | 0.0159 | 0.534 | 0.128 | -0.476 |
| 29 | `inventory_change_5__csz` | 0.0130 | 0.528 | 0.377 | -0.488 |
| 30 | `term_structure_momentum__csz` | 0.0128 | 0.540 | 0.012 | -0.558 |
| 31 | `fundamental_underreaction__csz` | 0.0127 | 0.518 | -0.111 | -0.802 |
