# 等权多因子模型测试报告（筛弱/去重等权 v2）

> 本报告由 handoff 矩阵接口自动生成；模型为**先验 predictor + 折内训练窗剔弱/去相关后的等权合成**。
> 筛选仅使用该折训练窗；**不**用样本外表现挑因子，不做子集枚举、不估计回归系数。
> 2025 不参与方向选择或主评价。

## Handoff 数据接口

| 项目 | 值 |
| --- | --- |
| handoff目录 | D:\软件2\AI-15-demo-main\--main2\--main - 副本 (3)\training_handoff |
| 特征文件 | ml_features_development.csv.gz |
| 注册表 | ml_feature_registry.csv |
| 标签文件 | labels_development.csv.gz |
| 折定义 | fold_definitions.csv |
| 注册特征数 | 92 |
| 特征行数 | 70822 |
| 特征日期 | 2015-01-05 — 2024-12-31 |
| 品种数 | 30 |

## 模型与标签选择

- 模型：`equal_weight_multi_v2` / `combination_scheme = all_equal` + `selection_mode = train_ic_corr`。
- 先验宇宙：仅 `usage_type=predictor`；排除 quality_flag / risk / conditional。
- 折内剔弱：训练窗方向对齐后日均 RankIC ≥ `0.01`。
- 折内去重：按训练窗 RankIC 从高到低贪心纳入；与已入选因子日均 |相关| ≥ `0.7` 则跳过。
- 合成：定向 → `cs_rank_centered` → 对入选且非缺失因子等权平均。
- 分数变换：`cs_rank_centered`（同日横截面中心化排名，映射到[-1,1]）。
- 缺失策略：`mean_available`（对可用因子等权平均）。
- 原始收益标签：`future_return_5d`（同一真实合约开盘→开盘，持有 5 个交易日）。
- 对照目标：`target_cs_rank_5d`（仅作可选对照，不参与任何选择）。
- IC 与策略收益一律相对 `future_return_5d` 计算。
- 外层验证：读取 `fold_definitions.csv` 的扩展窗口（2015 起点不变，2019–2024 逐年验证）；训练净化规则为 `label_end_date_5d < valid_start`。
- 方向：读取因子目录 `expected_direction`；若为 0，仅在该折训练窗内用日均 RankIC 符号选向并冻结。

## 策略定义

- 每个调仓信号日对**合成分** `prediction` 排序，最高 20% 做多、最低 20% 做空。
- 多头权重合计 0.5，空头权重合计 0.5；净敞口 0，总敞口 1；组内等权。
- T 收盘形成信号，T+1 开盘成交；持有 5 日，每 5 日调仓。
- 主成本口径单边 5 bp；敏感性报告单边 0/3/5/10 bp。
- 调仓日有效品种数 < 10 则跳过组合；IC 要求当日有效样本 ≥ 5 且两侧有变异。

## 候选宇宙

- usage_type 过滤：['predictor']
- 先验宇宙大小 N = **36**（predictor-only）；折均入选 **12.7**。
- 开发样本截止：2024-12-31

| usage_type | 字段数 |
| --- | --- |
| predictor | 36 |

目录期望方向分布（+1 / -1 / 0）：

| catalog_direction | 字段数 |
| --- | --- |
| 0 | 5 |
| 1 | 31 |

完整先验宇宙名单：

| feature_name | source_factor | usage_type | catalog_direction |
| --- | --- | --- | --- |
| tsmom_60__csz | tsmom_60 | predictor | 1 |
| csmom_60__csz | csmom_60 | predictor | 1 |
| trend_efficiency_60__csz | trend_efficiency_60 | predictor | 0 |
| multi_horizon_trend__csz | multi_horizon_trend | predictor | 1 |
| carry_annualized__csz | carry_annualized | predictor | 1 |
| carry_change_20__csz | carry_change_20 | predictor | 1 |
| basis_fresh__csz | basis_fresh | predictor | 1 |
| basis_shock_20__csz | basis_shock_20 | predictor | 1 |
| basis_momentum_20__csz | basis_momentum_20 | predictor | 1 |
| curve_curvature__csz | curve_curvature | predictor | 0 |
| value_252__csz | value_252 | predictor | 1 |
| short_reversal_5__csz | short_reversal_5 | predictor | 1 |
| inventory_scarcity__csz | inventory_scarcity | predictor | 1 |
| inventory_change_5__csz | inventory_change_5 | predictor | 1 |
| inventory_surprise__csz | inventory_surprise | predictor | 1 |
| warehouse_tightness__csz | warehouse_tightness | predictor | 1 |
| tightness_composite__csz | tightness_composite | predictor | 1 |
| price_oi_interaction_20__csz | price_oi_interaction_20 | predictor | 1 |
| seasonality_3y__csz | seasonality_3y | predictor | 1 |
| sector_relative_value_60__csz | sector_relative_value_60 | predictor | 1 |
| fundamental_composite_news__csz | fundamental_composite_news | predictor | 1 |
| fundamental_underreaction__csz | fundamental_underreaction | predictor | 1 |
| information_consistency__csz | information_consistency | predictor | 1 |
| information_divergence__csz | information_divergence | predictor | 0 |
| warehouse_surprise__csz | warehouse_surprise | predictor | 1 |
| momentum_quality_60__csz | momentum_quality_60 | predictor | 1 |
| vol_managed_momentum__csz | vol_managed_momentum | predictor | 1 |
| term_structure_slope__csz | term_structure_slope | predictor | 1 |
| term_structure_momentum__csz | term_structure_momentum | predictor | 1 |
| term_structure_acceleration__csz | term_structure_acceleration | predictor | 1 |
| basis_surprise__csz | basis_surprise | predictor | 1 |
| inventory_acceleration__csz | inventory_acceleration | predictor | 1 |
| inventory_price_divergence__csz | inventory_price_divergence | predictor | 1 |
| volume_price_divergence_20__csz | volume_price_divergence_20 | predictor | 0 |
| momentum_carry_interaction__csz | momentum_carry_interaction | predictor | 0 |
| nonlinear_tightness__csz | nonlinear_tightness | predictor | 1 |

## 折内训练窗筛选摘要

- `selection_mode = train_ic_corr`：仅用该折训练窗；**未**使用样本外 IC/Sharpe 挑因子。
- 剔弱阈值 `min_train_rank_ic = 0.01`。
- 去重阈值 `max_pairwise_corr = 0.7`（训练窗日均 |横截面相关|）。

各折入选数量与名单：

| fold_id | test_year | n_selected | selected_features |
| --- | --- | --- | --- |
| fold_2019 | 2019 | 15 | basis_fresh__csz, carry_annualized__csz, information_divergence__csz, inventory_price_divergence__csz, inventory_scarcity__csz, inventory_surprise__csz, momentum_carry_interaction__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, vol_managed_momentum__csz, volume_price_divergence_20__csz, warehouse_surprise__csz, warehouse_tightness__csz |
| fold_2020 | 2020 | 12 | basis_fresh__csz, basis_shock_20__csz, information_divergence__csz, inventory_scarcity__csz, inventory_surprise__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, vol_managed_momentum__csz, volume_price_divergence_20__csz, warehouse_tightness__csz |
| fold_2021 | 2021 | 13 | basis_fresh__csz, information_divergence__csz, inventory_price_divergence__csz, inventory_scarcity__csz, inventory_surprise__csz, nonlinear_tightness__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, vol_managed_momentum__csz, volume_price_divergence_20__csz, warehouse_surprise__csz |
| fold_2022 | 2022 | 12 | basis_fresh__csz, information_divergence__csz, inventory_price_divergence__csz, inventory_scarcity__csz, inventory_surprise__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, vol_managed_momentum__csz, volume_price_divergence_20__csz, warehouse_tightness__csz |
| fold_2023 | 2023 | 13 | basis_fresh__csz, fundamental_underreaction__csz, information_divergence__csz, inventory_acceleration__csz, inventory_change_5__csz, inventory_price_divergence__csz, short_reversal_5__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, tightness_composite__csz, value_252__csz, vol_managed_momentum__csz |
| fold_2024 | 2024 | 11 | basis_fresh__csz, information_divergence__csz, inventory_acceleration__csz, inventory_change_5__csz, inventory_price_divergence__csz, nonlinear_tightness__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, warehouse_surprise__csz |

筛选原因计数（跨折汇总）：

| selection_reason | 记录数 |
| --- | --- |
| dropped_weak_train_rank_ic | 115 |
| selected | 76 |
| dropped_high_corr | 25 |

## 方向规则与分折方向解析摘要

各因子方向来源汇总：

| feature_name | catalog_direction | resolved_directions | direction_source |
| --- | --- | --- | --- |
| basis_fresh__csz | 1 | 1 | catalog |
| basis_momentum_20__csz | 1 | 1 | catalog |
| basis_shock_20__csz | 1 | 1 | catalog |
| basis_surprise__csz | 1 | 1 | catalog |
| carry_annualized__csz | 1 | 1 | catalog |
| carry_change_20__csz | 1 | 1 | catalog |
| csmom_60__csz | 1 | 1 | catalog |
| curve_curvature__csz | 0 | -1,1 | train_rank_ic |
| fundamental_composite_news__csz | 1 | 1 | catalog |
| fundamental_underreaction__csz | 1 | 1 | catalog |
| information_consistency__csz | 1 | 1 | catalog |
| information_divergence__csz | 0 | 1 | train_rank_ic |
| inventory_acceleration__csz | 1 | 1 | catalog |
| inventory_change_5__csz | 1 | 1 | catalog |
| inventory_price_divergence__csz | 1 | 1 | catalog |
| inventory_scarcity__csz | 1 | 1 | catalog |
| inventory_surprise__csz | 1 | 1 | catalog |
| momentum_carry_interaction__csz | 0 | -1,1 | train_rank_ic |
| momentum_quality_60__csz | 1 | 1 | catalog |
| multi_horizon_trend__csz | 1 | 1 | catalog |
| nonlinear_tightness__csz | 1 | 1 | catalog |
| price_oi_interaction_20__csz | 1 | 1 | catalog |
| seasonality_3y__csz | 1 | 1 | catalog |
| sector_relative_value_60__csz | 1 | 1 | catalog |
| short_reversal_5__csz | 1 | 1 | catalog |
| term_structure_acceleration__csz | 1 | 1 | catalog |
| term_structure_momentum__csz | 1 | 1 | catalog |
| term_structure_slope__csz | 1 | 1 | catalog |
| tightness_composite__csz | 1 | 1 | catalog |
| trend_efficiency_60__csz | 0 | -1 | train_rank_ic |
| tsmom_60__csz | 1 | 1 | catalog |
| value_252__csz | 1 | 1 | catalog |
| vol_managed_momentum__csz | 1 | 1 | catalog |
| volume_price_divergence_20__csz | 0 | 1 | train_rank_ic |
| warehouse_surprise__csz | 1 | 1 | catalog |
| warehouse_tightness__csz | 1 | 1 | catalog |

训练窗定号明细（仅 `expected_direction=0` 的因子）：

| fold_id | test_year | feature_name | resolved_direction | train_rank_ic |
| --- | --- | --- | --- | --- |
| fold_2019 | 2019 | trend_efficiency_60__csz | -1 | 0.0078 |
| fold_2019 | 2019 | curve_curvature__csz | -1 | 0.0049 |
| fold_2019 | 2019 | information_divergence__csz | 1 | 0.0406 |
| fold_2019 | 2019 | volume_price_divergence_20__csz | 1 | 0.0200 |
| fold_2019 | 2019 | momentum_carry_interaction__csz | -1 | 0.0109 |
| fold_2020 | 2020 | trend_efficiency_60__csz | -1 | 0.0012 |
| fold_2020 | 2020 | curve_curvature__csz | 1 | 0.0042 |
| fold_2020 | 2020 | information_divergence__csz | 1 | 0.0364 |
| fold_2020 | 2020 | volume_price_divergence_20__csz | 1 | 0.0141 |
| fold_2020 | 2020 | momentum_carry_interaction__csz | -1 | 0.0074 |
| fold_2021 | 2021 | trend_efficiency_60__csz | -1 | 0.0061 |
| fold_2021 | 2021 | curve_curvature__csz | -1 | 0.0019 |
| fold_2021 | 2021 | information_divergence__csz | 1 | 0.0428 |
| fold_2021 | 2021 | volume_price_divergence_20__csz | 1 | 0.0148 |
| fold_2021 | 2021 | momentum_carry_interaction__csz | -1 | 0.0072 |
| fold_2022 | 2022 | trend_efficiency_60__csz | -1 | 0.0080 |
| fold_2022 | 2022 | curve_curvature__csz | -1 | 0.0062 |
| fold_2022 | 2022 | information_divergence__csz | 1 | 0.0409 |
| fold_2022 | 2022 | volume_price_divergence_20__csz | 1 | 0.0102 |
| fold_2022 | 2022 | momentum_carry_interaction__csz | 1 | 0.0011 |
| fold_2023 | 2023 | trend_efficiency_60__csz | -1 | 0.0049 |
| fold_2023 | 2023 | curve_curvature__csz | -1 | 0.0055 |
| fold_2023 | 2023 | information_divergence__csz | 1 | 0.0389 |
| fold_2023 | 2023 | volume_price_divergence_20__csz | 1 | 0.0044 |
| fold_2023 | 2023 | momentum_carry_interaction__csz | 1 | 0.0057 |
| fold_2024 | 2024 | trend_efficiency_60__csz | -1 | 0.0010 |
| fold_2024 | 2024 | curve_curvature__csz | -1 | 0.0014 |
| fold_2024 | 2024 | information_divergence__csz | 1 | 0.0345 |
| fold_2024 | 2024 | volume_price_divergence_20__csz | 1 | 0.0033 |
| fold_2024 | 2024 | momentum_carry_interaction__csz | 1 | 0.0060 |

## 总体样本外预测结果

| 指标 | 结果 |
| --- | --- |
| 模型版本 | equal_weight_multi_v2 |
| 组合方案 | all_equal |
| 筛选模式 | train_ic_corr |
| 样本外年份 | 2019—2024 |
| 先验宇宙大小 N | 36 |
| 折均入选数 | 12.7 |
| 缺失策略 | mean_available |
| score_transform | cs_rank_centered |
| 日均 Pearson IC | 0.0255 |
| 日均 RankIC | 0.0293 |
| RankIC 胜率 | 0.5703 |
| RankICIR | 0.1300 |
| 相对 target_cs_rank_5d 的 R² | -0.0975 |
| 相对 target_cs_rank_5d 的 MAE | 0.5307 |

## 与 v1（全体等权）对照

同协议、同 handoff；v2 仅改变先验宇宙与折内训练窗筛选。

| 版本 | 宇宙/入选 | RankIC | RankIC胜率 | 毛Sharpe | 净3bp Sharpe | 净5bp Sharpe |
| --- | --- | --- | --- | --- | --- | --- |
| v1 全体92等权 | 92 | 0.0163 | 0.5324 | 0.3642 | 0.0158 | -0.2164 |
| v2 筛弱/去重等权 | 36先验 / 折均12.7 | 0.0293 | 0.5703 | 0.6565 | 0.2412 | -0.0357 |

## 策略总体结果（毛 / 0/3/5/10 bp）

| 口径 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 总收益 | 胜率 |
| --- | --- | --- | --- | --- | --- | --- |
| 毛收益 | 0.0478 | 0.0728 | 0.6565 | -0.1100 | 0.3001 | 0.5427 |
| 净 0bp | 0.0478 | 0.0728 | 0.6565 | -0.1100 | 0.3001 | 0.5427 |
| 净 3bp | 0.0176 | 0.0728 | 0.2412 | -0.1243 | 0.0906 | 0.5222 |
| 净 5bp | -0.0026 | 0.0728 | -0.0357 | -0.1527 | -0.0300 | 0.5085 |
| 净 10bp | -0.0530 | 0.0728 | -0.7278 | -0.3390 | -0.2765 | 0.4642 |

## 分年度扩展窗口结果

分年度 RankIC、毛 Sharpe、净 3bp / 5bp Sharpe：

| fold_id | test_year | prior_universe_size | universe_size | rank_ic_mean | rank_ic_hit_rate | gross_sharpe | net_3bps_sharpe | net_5bps_sharpe | gross_annual_return | strategy_days | n_factors_used_mean | prediction_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_2019 | 2019 | 36 | 15 | 0.0129 | 0.4918 | 0.4231 | 0.0405 | -0.2145 | 0.0334 | 49 | 15.0000 | 1.0000 |
| fold_2020 | 2020 | 36 | 12 | 0.1036 | 0.6872 | 1.8732 | 1.5172 | 1.2799 | 0.1591 | 49 | 12.0000 | 1.0000 |
| fold_2021 | 2021 | 36 | 13 | 0.0242 | 0.5761 | -0.1947 | -0.6179 | -0.9000 | -0.0139 | 49 | 13.0000 | 1.0000 |
| fold_2022 | 2022 | 36 | 12 | 0.0041 | 0.5248 | 0.8032 | 0.3990 | 0.1296 | 0.0601 | 49 | 12.0000 | 1.0000 |
| fold_2023 | 2023 | 36 | 13 | 0.0073 | 0.5413 | 0.5938 | 0.1477 | -0.1498 | 0.0402 | 49 | 13.0000 | 1.0000 |
| fold_2024 | 2024 | 36 | 11 | 0.0236 | 0.6017 | 0.1234 | -0.4121 | -0.7690 | 0.0070 | 48 | 11.0000 | 1.0000 |

## 覆盖率与 n_factors_used 摘要

| 指标 | 结果 |
| --- | --- |
| 策略调仓日数 | 293 |
| 预测交易日数 | 1450 |
| 策略覆盖率 | 0.2021 |
| 预测行覆盖率 | 1.0000 |
| n_factors_used 均值 | 12.6649 |
| n_factors_used P25 | 12.0000 |
| n_factors_used P50 | 13.0000 |
| n_factors_used P75 | 13.0000 |

## 使用边界

1. 点时：因子在 T 日收盘后形成，信号仅可用于 T+1 开盘交易；切分、方向选择、筛选与评价均未使用测试段。
2. 本模型是筛弱/去重后的等权多因子，**不是**单因子排行榜，也**不是** Ridge / OLS 加权结果。
3. 入选名单允许折间变化；阈值写死在配置中，未对阈值做网格搜索。
4. 成本仅为单边 0/3/5/10 基点敏感性，不得解读为已核验的真实手续费或冲击成本。
5. 2025 未参与主评价；若另做补充样本外，须单独配置与目录。
6. 未核验合约乘数、涨跌停、成交容量与真实交易成本，结果仅属研究回测。
7. v1（92 全体等权）仍是无选择地板基准；本版用于检验去弱/去重是否提升稳健性。
