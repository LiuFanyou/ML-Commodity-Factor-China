# Predictor 对照短表（Top10 × 多指标）

- 宇宙：`usage_type=predictor` 且排除 `__missing`
- 每张短表取前 10 名
- 全量 92 字段仍以 `01_`–`05_` 表为准；本文件只服务正文/汇报可读性

## 怎么读

- **A∩B**：预测力与 5bp 成本后都靠前 → 优先讨论
- **只在 A**：RankIC 好但成本后一般 → 关注换手/持有期
- **只在 B**：成本后靠前但 RankIC 未必最高 → 谨慎解读
- **C**：在 RankIC>0 子集里按「正 IC 年数 / 胜率」看稳定性

## A. Top10 by RankIC（主短表）

规则：rank_ic_mean ↓, hit ↓, gross Sharpe ↓

| # | overall_rank | feature | RankIC | hit | gross Sharpe | net5 Sharpe | +IC years |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | `carry_annualized__csz` | 0.0291 | 0.547 | 0.279 | -0.272 | 5/6 |
| 2 | 11 | `basis_fresh__csz` | 0.0285 | 0.557 | 0.463 | -0.168 | 5/6 |
| 3 | 12 | `warehouse_tightness__csz` | 0.0230 | 0.540 | 0.831 | 0.053 | 5/6 |
| 4 | 13 | `information_divergence__csz` | 0.0227 | 0.539 | 0.284 | -0.387 | 5/6 |
| 5 | 14 | `inventory_price_divergence__csz` | 0.0214 | 0.546 | 0.313 | -0.551 | 5/6 |
| 6 | 15 | `term_structure_slope__csz` | 0.0204 | 0.546 | 0.107 | -0.388 | 4/6 |
| 7 | 16 | `nonlinear_tightness__csz` | 0.0197 | 0.524 | 0.539 | -0.150 | 4/6 |
| 8 | 17 | `tightness_composite__csz` | 0.0194 | 0.525 | 0.523 | -0.166 | 4/6 |
| 9 | 18 | `information_consistency__csz` | 0.0190 | 0.537 | 0.322 | -0.367 | 4/6 |
| 10 | 19 | `short_reversal_5__csz` | 0.0182 | 0.526 | 0.438 | -0.033 | 5/6 |

## B. Top10 by net 5bp Sharpe（成本对照）

规则：net_5bps_sharpe ↓, rank_ic_mean ↓

| # | overall_rank | feature | RankIC | hit | gross Sharpe | net5 Sharpe | +IC years |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12 | `warehouse_tightness__csz` | 0.0230 | 0.540 | 0.831 | 0.053 | 5/6 |
| 2 | 19 | `short_reversal_5__csz` | 0.0182 | 0.526 | 0.438 | -0.033 | 5/6 |
| 3 | 16 | `nonlinear_tightness__csz` | 0.0197 | 0.524 | 0.539 | -0.150 | 4/6 |
| 4 | 17 | `tightness_composite__csz` | 0.0194 | 0.525 | 0.523 | -0.166 | 4/6 |
| 5 | 11 | `basis_fresh__csz` | 0.0285 | 0.557 | 0.463 | -0.168 | 5/6 |
| 6 | 36 | `sector_relative_value_60__csz` | 0.0109 | 0.524 | 0.317 | -0.220 | 5/6 |
| 7 | 10 | `carry_annualized__csz` | 0.0291 | 0.547 | 0.279 | -0.272 | 5/6 |
| 8 | 18 | `information_consistency__csz` | 0.0190 | 0.537 | 0.322 | -0.367 | 4/6 |
| 9 | 13 | `information_divergence__csz` | 0.0227 | 0.539 | 0.284 | -0.387 | 5/6 |
| 10 | 15 | `term_structure_slope__csz` | 0.0204 | 0.546 | 0.107 | -0.388 | 4/6 |

## C. Top10 by stability（可选）

规则：universe: RankIC>0; sort by years_rank_ic_positive ↓, hit ↓, RankIC ↓

| # | overall_rank | feature | RankIC | hit | gross Sharpe | net5 Sharpe | +IC years |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 11 | `basis_fresh__csz` | 0.0285 | 0.557 | 0.463 | -0.168 | 5/6 |
| 2 | 10 | `carry_annualized__csz` | 0.0291 | 0.547 | 0.279 | -0.272 | 5/6 |
| 3 | 14 | `inventory_price_divergence__csz` | 0.0214 | 0.546 | 0.313 | -0.551 | 5/6 |
| 4 | 20 | `inventory_acceleration__csz` | 0.0178 | 0.544 | 0.195 | -0.568 | 5/6 |
| 5 | 12 | `warehouse_tightness__csz` | 0.0230 | 0.540 | 0.831 | 0.053 | 5/6 |
| 6 | 30 | `term_structure_momentum__csz` | 0.0128 | 0.540 | 0.012 | -0.558 | 5/6 |
| 7 | 13 | `information_divergence__csz` | 0.0227 | 0.539 | 0.284 | -0.387 | 5/6 |
| 8 | 21 | `term_structure_acceleration__csz` | 0.0159 | 0.534 | 0.128 | -0.476 | 5/6 |
| 9 | 19 | `short_reversal_5__csz` | 0.0182 | 0.526 | 0.438 | -0.033 | 5/6 |
| 10 | 36 | `sector_relative_value_60__csz` | 0.0109 | 0.524 | 0.317 | -0.220 | 5/6 |

## 交集与错位（A vs B）

- A∩B（9）：`basis_fresh__csz`, `carry_annualized__csz`, `information_consistency__csz`, `information_divergence__csz`, `nonlinear_tightness__csz`, `short_reversal_5__csz`, `term_structure_slope__csz`, `tightness_composite__csz`, `warehouse_tightness__csz`
- 只在 A（1）：`inventory_price_divergence__csz`
- 只在 B（1）：`sector_relative_value_60__csz`

## 文件

- `modeling\single_factor_rank_v1\outputs\tables\06_shortlist_top10_by_rank_ic.csv`
- `modeling\single_factor_rank_v1\outputs\tables\07_shortlist_top10_by_net5bps_sharpe.csv`
- `modeling\single_factor_rank_v1\outputs\tables\08_shortlist_top10_by_stability.csv`
- `modeling\single_factor_rank_v1\outputs\tables\09_shortlist_overlap.csv`
