# 等权多因子模型测试报告（全体等权）

> 本报告由 handoff 矩阵接口自动生成；模型为**全体等权多因子合成**，不估计回归系数、不做 Top-K、不枚举子集。
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

- 模型：方案 B / `combination_scheme = all_equal` —— 主宇宙内全部字段等权合成单一 `prediction`。
- **未做**因子挑选、子集枚举、IC/Sharpe 加权或回归拟合。
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

- usage_type 过滤：None
- 宇宙大小 N = **92**（「全体」= `ml_feature_registry.csv` 全部注册字段，与 Ridge / handoff 一致）。
- 开发样本截止：2024-12-31

| usage_type | 字段数 |
| --- | --- |
| conditional | 5 |
| predictor | 36 |
| quality_flag | 45 |
| risk | 6 |

目录期望方向分布（+1 / -1 / 0）：

| catalog_direction | 字段数 |
| --- | --- |
| -1 | 6 |
| 0 | 23 |
| 1 | 63 |

完整宇宙名单：

| feature_name | source_factor | usage_type | catalog_direction |
| --- | --- | --- | --- |
| tsmom_60__csz | tsmom_60 | predictor | 1 |
| tsmom_60__missing | tsmom_60 | quality_flag | 1 |
| csmom_60__csz | csmom_60 | predictor | 1 |
| csmom_60__missing | csmom_60 | quality_flag | 1 |
| trend_efficiency_60__csz | trend_efficiency_60 | predictor | 0 |
| trend_efficiency_60__missing | trend_efficiency_60 | quality_flag | 0 |
| multi_horizon_trend__csz | multi_horizon_trend | predictor | 1 |
| multi_horizon_trend__missing | multi_horizon_trend | quality_flag | 1 |
| carry_annualized__csz | carry_annualized | predictor | 1 |
| carry_annualized__missing | carry_annualized | quality_flag | 1 |
| carry_change_20__csz | carry_change_20 | predictor | 1 |
| carry_change_20__missing | carry_change_20 | quality_flag | 1 |
| basis_fresh__csz | basis_fresh | predictor | 1 |
| basis_fresh__missing | basis_fresh | quality_flag | 1 |
| basis_shock_20__csz | basis_shock_20 | predictor | 1 |
| basis_shock_20__missing | basis_shock_20 | quality_flag | 1 |
| basis_momentum_20__csz | basis_momentum_20 | predictor | 1 |
| basis_momentum_20__missing | basis_momentum_20 | quality_flag | 1 |
| curve_curvature__csz | curve_curvature | predictor | 0 |
| curve_curvature__missing | curve_curvature | quality_flag | 0 |
| value_252__csz | value_252 | predictor | 1 |
| value_252__missing | value_252 | quality_flag | 1 |
| short_reversal_5__csz | short_reversal_5 | predictor | 1 |
| inventory_scarcity__csz | inventory_scarcity | predictor | 1 |
| inventory_scarcity__missing | inventory_scarcity | quality_flag | 1 |
| inventory_change_5__csz | inventory_change_5 | predictor | 1 |
| inventory_change_5__missing | inventory_change_5 | quality_flag | 1 |
| inventory_surprise__csz | inventory_surprise | predictor | 1 |
| inventory_surprise__missing | inventory_surprise | quality_flag | 1 |
| warehouse_tightness__csz | warehouse_tightness | predictor | 1 |
| warehouse_tightness__missing | warehouse_tightness | quality_flag | 1 |
| tightness_composite__csz | tightness_composite | predictor | 1 |
| tightness_composite__missing | tightness_composite | quality_flag | 1 |
| oi_growth_20__csz | oi_growth_20 | conditional | 0 |
| oi_growth_20__missing | oi_growth_20 | quality_flag | 0 |
| price_oi_interaction_20__csz | price_oi_interaction_20 | predictor | 1 |
| price_oi_interaction_20__missing | price_oi_interaction_20 | quality_flag | 1 |
| volume_shock_20__csz | volume_shock_20 | conditional | 0 |
| volume_shock_20__missing | volume_shock_20 | quality_flag | 0 |
| amihud_20__csz | amihud_20 | risk | -1 |
| amihud_20__missing | amihud_20 | quality_flag | -1 |
| turnover_oi__csz | turnover_oi | conditional | 0 |
| realized_volatility_20__csz | realized_volatility_20 | risk | 0 |
| realized_volatility_20__missing | realized_volatility_20 | quality_flag | 0 |
| low_volatility_20__csz | low_volatility_20 | risk | 1 |
| low_volatility_20__missing | low_volatility_20 | quality_flag | 1 |
| realized_skewness_60__csz | realized_skewness_60 | risk | -1 |
| realized_skewness_60__missing | realized_skewness_60 | quality_flag | -1 |
| seasonality_3y__csz | seasonality_3y | predictor | 1 |
| seasonality_3y__missing | seasonality_3y | quality_flag | 1 |
| sector_relative_value_60__csz | sector_relative_value_60 | predictor | 1 |
| sector_relative_value_60__missing | sector_relative_value_60 | quality_flag | 1 |
| factor_momentum__tsz | factor_momentum | conditional | 0 |
| factor_momentum__missing | factor_momentum | quality_flag | 0 |
| fundamental_composite_news__csz | fundamental_composite_news | predictor | 1 |
| fundamental_composite_news__missing | fundamental_composite_news | quality_flag | 1 |
| fundamental_underreaction__csz | fundamental_underreaction | predictor | 1 |
| fundamental_underreaction__missing | fundamental_underreaction | quality_flag | 1 |
| information_consistency__csz | information_consistency | predictor | 1 |
| information_consistency__missing | information_consistency | quality_flag | 1 |
| information_divergence__csz | information_divergence | predictor | 0 |
| information_divergence__missing | information_divergence | quality_flag | 0 |
| warehouse_surprise__csz | warehouse_surprise | predictor | 1 |
| warehouse_surprise__missing | warehouse_surprise | quality_flag | 1 |
| momentum_quality_60__csz | momentum_quality_60 | predictor | 1 |
| momentum_quality_60__missing | momentum_quality_60 | quality_flag | 1 |
| vol_managed_momentum__csz | vol_managed_momentum | predictor | 1 |
| vol_managed_momentum__missing | vol_managed_momentum | quality_flag | 1 |
| idiosyncratic_volatility_60__csz | idiosyncratic_volatility_60 | risk | 0 |
| idiosyncratic_volatility_60__missing | idiosyncratic_volatility_60 | quality_flag | 0 |
| term_structure_slope__csz | term_structure_slope | predictor | 1 |
| term_structure_slope__missing | term_structure_slope | quality_flag | 1 |
| term_structure_momentum__csz | term_structure_momentum | predictor | 1 |
| term_structure_momentum__missing | term_structure_momentum | quality_flag | 1 |
| term_structure_acceleration__csz | term_structure_acceleration | predictor | 1 |
| term_structure_acceleration__missing | term_structure_acceleration | quality_flag | 1 |
| basis_surprise__csz | basis_surprise | predictor | 1 |
| basis_surprise__missing | basis_surprise | quality_flag | 1 |
| inventory_acceleration__csz | inventory_acceleration | predictor | 1 |
| inventory_acceleration__missing | inventory_acceleration | quality_flag | 1 |
| inventory_price_divergence__csz | inventory_price_divergence | predictor | 1 |
| inventory_price_divergence__missing | inventory_price_divergence | quality_flag | 1 |
| oi_surprise_60__csz | oi_surprise_60 | conditional | 0 |
| oi_surprise_60__missing | oi_surprise_60 | quality_flag | 0 |
| volume_price_divergence_20__csz | volume_price_divergence_20 | predictor | 0 |
| volume_price_divergence_20__missing | volume_price_divergence_20 | quality_flag | 0 |
| momentum_carry_interaction__csz | momentum_carry_interaction | predictor | 0 |
| momentum_carry_interaction__missing | momentum_carry_interaction | quality_flag | 0 |
| nonlinear_tightness__csz | nonlinear_tightness | predictor | 1 |
| nonlinear_tightness__missing | nonlinear_tightness | quality_flag | 1 |
| correlation_risk_20_120__csz | correlation_risk_20_120 | risk | -1 |
| correlation_risk_20_120__missing | correlation_risk_20_120 | quality_flag | -1 |

## 方向规则与分折方向解析摘要

各因子方向来源汇总：

| feature_name | catalog_direction | resolved_directions | direction_source |
| --- | --- | --- | --- |
| amihud_20__csz | -1 | -1 | catalog |
| amihud_20__missing | -1 | -1 | catalog |
| basis_fresh__csz | 1 | 1 | catalog |
| basis_fresh__missing | 1 | 1 | catalog |
| basis_momentum_20__csz | 1 | 1 | catalog |
| basis_momentum_20__missing | 1 | 1 | catalog |
| basis_shock_20__csz | 1 | 1 | catalog |
| basis_shock_20__missing | 1 | 1 | catalog |
| basis_surprise__csz | 1 | 1 | catalog |
| basis_surprise__missing | 1 | 1 | catalog |
| carry_annualized__csz | 1 | 1 | catalog |
| carry_annualized__missing | 1 | 1 | catalog |
| carry_change_20__csz | 1 | 1 | catalog |
| carry_change_20__missing | 1 | 1 | catalog |
| correlation_risk_20_120__csz | -1 | -1 | catalog |
| correlation_risk_20_120__missing | -1 | -1 | catalog |
| csmom_60__csz | 1 | 1 | catalog |
| csmom_60__missing | 1 | 1 | catalog |
| curve_curvature__csz | 0 | -1,1 | train_rank_ic |
| curve_curvature__missing | 0 | -1 | train_rank_ic |
| factor_momentum__missing | 0 | 1 | train_rank_ic |
| factor_momentum__tsz | 0 | 1 | train_rank_ic |
| fundamental_composite_news__csz | 1 | 1 | catalog |
| fundamental_composite_news__missing | 1 | 1 | catalog |
| fundamental_underreaction__csz | 1 | 1 | catalog |
| fundamental_underreaction__missing | 1 | 1 | catalog |
| idiosyncratic_volatility_60__csz | 0 | 1 | train_rank_ic |
| idiosyncratic_volatility_60__missing | 0 | 1 | train_rank_ic |
| information_consistency__csz | 1 | 1 | catalog |
| information_consistency__missing | 1 | 1 | catalog |
| information_divergence__csz | 0 | 1 | train_rank_ic |
| information_divergence__missing | 0 | 1 | train_rank_ic |
| inventory_acceleration__csz | 1 | 1 | catalog |
| inventory_acceleration__missing | 1 | 1 | catalog |
| inventory_change_5__csz | 1 | 1 | catalog |
| inventory_change_5__missing | 1 | 1 | catalog |
| inventory_price_divergence__csz | 1 | 1 | catalog |
| inventory_price_divergence__missing | 1 | 1 | catalog |
| inventory_scarcity__csz | 1 | 1 | catalog |
| inventory_scarcity__missing | 1 | 1 | catalog |
| inventory_surprise__csz | 1 | 1 | catalog |
| inventory_surprise__missing | 1 | 1 | catalog |
| low_volatility_20__csz | 1 | 1 | catalog |
| low_volatility_20__missing | 1 | 1 | catalog |
| momentum_carry_interaction__csz | 0 | -1,1 | train_rank_ic |
| momentum_carry_interaction__missing | 0 | -1 | train_rank_ic |
| momentum_quality_60__csz | 1 | 1 | catalog |
| momentum_quality_60__missing | 1 | 1 | catalog |
| multi_horizon_trend__csz | 1 | 1 | catalog |
| multi_horizon_trend__missing | 1 | 1 | catalog |
| nonlinear_tightness__csz | 1 | 1 | catalog |
| nonlinear_tightness__missing | 1 | 1 | catalog |
| oi_growth_20__csz | 0 | -1,1 | train_rank_ic |
| oi_growth_20__missing | 0 | 1 | train_rank_ic |
| oi_surprise_60__csz | 0 | -1 | train_rank_ic |
| oi_surprise_60__missing | 0 | 1 | train_rank_ic |
| price_oi_interaction_20__csz | 1 | 1 | catalog |
| price_oi_interaction_20__missing | 1 | 1 | catalog |
| realized_skewness_60__csz | -1 | -1 | catalog |
| realized_skewness_60__missing | -1 | -1 | catalog |
| realized_volatility_20__csz | 0 | 1 | train_rank_ic |
| realized_volatility_20__missing | 0 | -1,1 | train_rank_ic |
| seasonality_3y__csz | 1 | 1 | catalog |
| seasonality_3y__missing | 1 | 1 | catalog |
| sector_relative_value_60__csz | 1 | 1 | catalog |
| sector_relative_value_60__missing | 1 | 1 | catalog |
| short_reversal_5__csz | 1 | 1 | catalog |
| term_structure_acceleration__csz | 1 | 1 | catalog |
| term_structure_acceleration__missing | 1 | 1 | catalog |
| term_structure_momentum__csz | 1 | 1 | catalog |
| term_structure_momentum__missing | 1 | 1 | catalog |
| term_structure_slope__csz | 1 | 1 | catalog |
| term_structure_slope__missing | 1 | 1 | catalog |
| tightness_composite__csz | 1 | 1 | catalog |
| tightness_composite__missing | 1 | 1 | catalog |
| trend_efficiency_60__csz | 0 | -1 | train_rank_ic |
| trend_efficiency_60__missing | 0 | -1 | train_rank_ic |
| tsmom_60__csz | 1 | 1 | catalog |
| tsmom_60__missing | 1 | 1 | catalog |
| turnover_oi__csz | 0 | -1,1 | train_rank_ic |
| value_252__csz | 1 | 1 | catalog |
| value_252__missing | 1 | 1 | catalog |
| vol_managed_momentum__csz | 1 | 1 | catalog |
| vol_managed_momentum__missing | 1 | 1 | catalog |
| volume_price_divergence_20__csz | 0 | 1 | train_rank_ic |
| volume_price_divergence_20__missing | 0 | 1 | train_rank_ic |
| volume_shock_20__csz | 0 | -1 | train_rank_ic |
| volume_shock_20__missing | 0 | -1 | train_rank_ic |
| warehouse_surprise__csz | 1 | 1 | catalog |
| warehouse_surprise__missing | 1 | 1 | catalog |
| warehouse_tightness__csz | 1 | 1 | catalog |
| warehouse_tightness__missing | 1 | 1 | catalog |

训练窗定号明细（仅 `expected_direction=0` 的因子）：

| fold_id | test_year | feature_name | resolved_direction | train_rank_ic |
| --- | --- | --- | --- | --- |
| fold_2019 | 2019 | trend_efficiency_60__csz | -1 | -0.0078 |
| fold_2019 | 2019 | trend_efficiency_60__missing | -1 | -0.0059 |
| fold_2019 | 2019 | curve_curvature__csz | -1 | -0.0049 |
| fold_2019 | 2019 | curve_curvature__missing | -1 | -0.0076 |
| fold_2019 | 2019 | oi_growth_20__csz | -1 | -0.0154 |
| fold_2019 | 2019 | oi_growth_20__missing | 1 | 0.0270 |
| fold_2019 | 2019 | volume_shock_20__csz | -1 | -0.0275 |
| fold_2019 | 2019 | volume_shock_20__missing | -1 | -0.0545 |
| fold_2019 | 2019 | turnover_oi__csz | 1 | 0.0047 |
| fold_2019 | 2019 | realized_volatility_20__csz | 1 | 0.0209 |
| fold_2019 | 2019 | realized_volatility_20__missing | 1 | 0.0070 |
| fold_2019 | 2019 | factor_momentum__tsz | 1 | — |
| fold_2019 | 2019 | factor_momentum__missing | 1 | — |
| fold_2019 | 2019 | information_divergence__csz | 1 | 0.0406 |
| fold_2019 | 2019 | information_divergence__missing | 1 | 0.0311 |
| fold_2019 | 2019 | idiosyncratic_volatility_60__csz | 1 | 0.0171 |
| fold_2019 | 2019 | idiosyncratic_volatility_60__missing | 1 | 0.0072 |
| fold_2019 | 2019 | oi_surprise_60__csz | -1 | -0.0178 |
| fold_2019 | 2019 | oi_surprise_60__missing | 1 | 0.0256 |
| fold_2019 | 2019 | volume_price_divergence_20__csz | 1 | 0.0200 |
| fold_2019 | 2019 | volume_price_divergence_20__missing | 1 | 0.0152 |
| fold_2019 | 2019 | momentum_carry_interaction__csz | -1 | -0.0109 |
| fold_2019 | 2019 | momentum_carry_interaction__missing | -1 | -0.0069 |
| fold_2020 | 2020 | trend_efficiency_60__csz | -1 | -0.0012 |
| fold_2020 | 2020 | trend_efficiency_60__missing | -1 | -0.0096 |
| fold_2020 | 2020 | curve_curvature__csz | 1 | 0.0042 |
| fold_2020 | 2020 | curve_curvature__missing | -1 | -0.0134 |
| fold_2020 | 2020 | oi_growth_20__csz | -1 | -0.0078 |
| fold_2020 | 2020 | oi_growth_20__missing | 1 | 0.0019 |
| fold_2020 | 2020 | volume_shock_20__csz | -1 | -0.0266 |
| fold_2020 | 2020 | volume_shock_20__missing | -1 | -0.0647 |
| fold_2020 | 2020 | turnover_oi__csz | 1 | 0.0075 |
| fold_2020 | 2020 | realized_volatility_20__csz | 1 | 0.0215 |
| fold_2020 | 2020 | realized_volatility_20__missing | -1 | -0.0016 |
| fold_2020 | 2020 | factor_momentum__tsz | 1 | — |
| fold_2020 | 2020 | factor_momentum__missing | 1 | — |
| fold_2020 | 2020 | information_divergence__csz | 1 | 0.0364 |
| fold_2020 | 2020 | information_divergence__missing | 1 | 0.0359 |
| fold_2020 | 2020 | idiosyncratic_volatility_60__csz | 1 | 0.0223 |
| fold_2020 | 2020 | idiosyncratic_volatility_60__missing | 1 | 0.0001 |
| fold_2020 | 2020 | oi_surprise_60__csz | -1 | -0.0172 |
| fold_2020 | 2020 | oi_surprise_60__missing | 1 | 0.0024 |
| fold_2020 | 2020 | volume_price_divergence_20__csz | 1 | 0.0141 |
| fold_2020 | 2020 | volume_price_divergence_20__missing | 1 | 0.0076 |
| fold_2020 | 2020 | momentum_carry_interaction__csz | -1 | -0.0074 |
| fold_2020 | 2020 | momentum_carry_interaction__missing | -1 | -0.0080 |
| fold_2021 | 2021 | trend_efficiency_60__csz | -1 | -0.0061 |
| fold_2021 | 2021 | trend_efficiency_60__missing | -1 | -0.0059 |
| fold_2021 | 2021 | curve_curvature__csz | -1 | -0.0019 |
| fold_2021 | 2021 | curve_curvature__missing | -1 | -0.0100 |
| fold_2021 | 2021 | oi_growth_20__csz | -1 | -0.0049 |
| fold_2021 | 2021 | oi_growth_20__missing | 1 | 0.0031 |
| fold_2021 | 2021 | volume_shock_20__csz | -1 | -0.0245 |
| fold_2021 | 2021 | volume_shock_20__missing | -1 | -0.0647 |
| fold_2021 | 2021 | turnover_oi__csz | -1 | -0.0006 |
| fold_2021 | 2021 | realized_volatility_20__csz | 1 | 0.0195 |
| fold_2021 | 2021 | realized_volatility_20__missing | -1 | -0.0010 |
| fold_2021 | 2021 | factor_momentum__tsz | 1 | — |
| fold_2021 | 2021 | factor_momentum__missing | 1 | — |
| fold_2021 | 2021 | information_divergence__csz | 1 | 0.0428 |
| fold_2021 | 2021 | information_divergence__missing | 1 | 0.0261 |
| fold_2021 | 2021 | idiosyncratic_volatility_60__csz | 1 | 0.0256 |
| fold_2021 | 2021 | idiosyncratic_volatility_60__missing | 1 | 0.0045 |
| fold_2021 | 2021 | oi_surprise_60__csz | -1 | -0.0126 |
| fold_2021 | 2021 | oi_surprise_60__missing | 1 | 0.0053 |
| fold_2021 | 2021 | volume_price_divergence_20__csz | 1 | 0.0148 |
| fold_2021 | 2021 | volume_price_divergence_20__missing | 1 | 0.0078 |
| fold_2021 | 2021 | momentum_carry_interaction__csz | -1 | -0.0072 |
| fold_2021 | 2021 | momentum_carry_interaction__missing | -1 | -0.0059 |
| fold_2022 | 2022 | trend_efficiency_60__csz | -1 | -0.0080 |
| fold_2022 | 2022 | trend_efficiency_60__missing | -1 | -0.0059 |
| fold_2022 | 2022 | curve_curvature__csz | -1 | -0.0062 |
| fold_2022 | 2022 | curve_curvature__missing | -1 | -0.0091 |
| fold_2022 | 2022 | oi_growth_20__csz | 1 | 0.0029 |
| fold_2022 | 2022 | oi_growth_20__missing | 1 | 0.0031 |
| fold_2022 | 2022 | volume_shock_20__csz | -1 | -0.0175 |
| fold_2022 | 2022 | volume_shock_20__missing | -1 | -0.0647 |
| fold_2022 | 2022 | turnover_oi__csz | 1 | 0.0024 |
| fold_2022 | 2022 | realized_volatility_20__csz | 1 | 0.0206 |
| fold_2022 | 2022 | realized_volatility_20__missing | -1 | -0.0010 |
| fold_2022 | 2022 | factor_momentum__tsz | 1 | — |
| fold_2022 | 2022 | factor_momentum__missing | 1 | — |
| fold_2022 | 2022 | information_divergence__csz | 1 | 0.0409 |
| fold_2022 | 2022 | information_divergence__missing | 1 | 0.0251 |
| fold_2022 | 2022 | idiosyncratic_volatility_60__csz | 1 | 0.0273 |
| fold_2022 | 2022 | idiosyncratic_volatility_60__missing | 1 | 0.0045 |
| fold_2022 | 2022 | oi_surprise_60__csz | -1 | -0.0115 |
| fold_2022 | 2022 | oi_surprise_60__missing | 1 | 0.0053 |
| fold_2022 | 2022 | volume_price_divergence_20__csz | 1 | 0.0102 |
| fold_2022 | 2022 | volume_price_divergence_20__missing | 1 | 0.0078 |
| fold_2022 | 2022 | momentum_carry_interaction__csz | 1 | 0.0011 |
| fold_2022 | 2022 | momentum_carry_interaction__missing | -1 | -0.0059 |
| fold_2023 | 2023 | trend_efficiency_60__csz | -1 | -0.0049 |
| fold_2023 | 2023 | trend_efficiency_60__missing | -1 | -0.0059 |
| fold_2023 | 2023 | curve_curvature__csz | -1 | -0.0055 |
| fold_2023 | 2023 | curve_curvature__missing | -1 | -0.0061 |
| fold_2023 | 2023 | oi_growth_20__csz | 1 | 0.0002 |
| fold_2023 | 2023 | oi_growth_20__missing | 1 | 0.0031 |
| fold_2023 | 2023 | volume_shock_20__csz | -1 | -0.0119 |
| fold_2023 | 2023 | volume_shock_20__missing | -1 | -0.0647 |
| fold_2023 | 2023 | turnover_oi__csz | 1 | 0.0045 |
| fold_2023 | 2023 | realized_volatility_20__csz | 1 | 0.0246 |
| fold_2023 | 2023 | realized_volatility_20__missing | -1 | -0.0010 |
| fold_2023 | 2023 | factor_momentum__tsz | 1 | — |
| fold_2023 | 2023 | factor_momentum__missing | 1 | — |
| fold_2023 | 2023 | information_divergence__csz | 1 | 0.0389 |
| fold_2023 | 2023 | information_divergence__missing | 1 | 0.0291 |
| fold_2023 | 2023 | idiosyncratic_volatility_60__csz | 1 | 0.0273 |
| fold_2023 | 2023 | idiosyncratic_volatility_60__missing | 1 | 0.0045 |
| fold_2023 | 2023 | oi_surprise_60__csz | -1 | -0.0095 |
| fold_2023 | 2023 | oi_surprise_60__missing | 1 | 0.0053 |
| fold_2023 | 2023 | volume_price_divergence_20__csz | 1 | 0.0044 |
| fold_2023 | 2023 | volume_price_divergence_20__missing | 1 | 0.0090 |
| fold_2023 | 2023 | momentum_carry_interaction__csz | 1 | 0.0057 |
| fold_2023 | 2023 | momentum_carry_interaction__missing | -1 | -0.0059 |
| fold_2024 | 2024 | trend_efficiency_60__csz | -1 | -0.0010 |
| fold_2024 | 2024 | trend_efficiency_60__missing | -1 | -0.0059 |
| fold_2024 | 2024 | curve_curvature__csz | -1 | -0.0014 |
| fold_2024 | 2024 | curve_curvature__missing | -1 | -0.0049 |
| fold_2024 | 2024 | oi_growth_20__csz | 1 | 0.0001 |
| fold_2024 | 2024 | oi_growth_20__missing | 1 | 0.0031 |
| fold_2024 | 2024 | volume_shock_20__csz | -1 | -0.0096 |
| fold_2024 | 2024 | volume_shock_20__missing | -1 | -0.0647 |
| fold_2024 | 2024 | turnover_oi__csz | 1 | 0.0023 |
| fold_2024 | 2024 | realized_volatility_20__csz | 1 | 0.0249 |
| fold_2024 | 2024 | realized_volatility_20__missing | -1 | -0.0010 |
| fold_2024 | 2024 | factor_momentum__tsz | 1 | — |
| fold_2024 | 2024 | factor_momentum__missing | 1 | — |
| fold_2024 | 2024 | information_divergence__csz | 1 | 0.0345 |
| fold_2024 | 2024 | information_divergence__missing | 1 | 0.0315 |
| fold_2024 | 2024 | idiosyncratic_volatility_60__csz | 1 | 0.0219 |
| fold_2024 | 2024 | idiosyncratic_volatility_60__missing | 1 | 0.0045 |
| fold_2024 | 2024 | oi_surprise_60__csz | -1 | -0.0080 |
| fold_2024 | 2024 | oi_surprise_60__missing | 1 | 0.0053 |
| fold_2024 | 2024 | volume_price_divergence_20__csz | 1 | 0.0033 |
| fold_2024 | 2024 | volume_price_divergence_20__missing | 1 | 0.0090 |
| fold_2024 | 2024 | momentum_carry_interaction__csz | 1 | 0.0060 |
| fold_2024 | 2024 | momentum_carry_interaction__missing | -1 | -0.0059 |

## 总体样本外预测结果

| 指标 | 结果 |
| --- | --- |
| 组合方案 | all_equal |
| 样本外年份 | 2019—2024 |
| 宇宙大小 N | 92 |
| 缺失策略 | mean_available |
| score_transform | cs_rank_centered |
| 日均 Pearson IC | 0.0186 |
| 日均 RankIC | 0.0163 |
| RankIC 胜率 | 0.5324 |
| RankICIR | 0.0678 |
| 相对 target_cs_rank_5d 的 R² | -0.0153 |
| 相对 target_cs_rank_5d 的 MAE | 0.5187 |

## 策略总体结果（毛 / 0/3/5/10 bp）

| 口径 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 总收益 | 胜率 |
| --- | --- | --- | --- | --- | --- | --- |
| 毛收益 | 0.0316 | 0.0868 | 0.3642 | -0.1321 | 0.1757 | 0.5495 |
| 净 0bp | 0.0316 | 0.0868 | 0.3642 | -0.1321 | 0.1757 | 0.5495 |
| 净 3bp | 0.0014 | 0.0868 | 0.0158 | -0.1886 | -0.0138 | 0.5358 |
| 净 5bp | -0.0188 | 0.0868 | -0.2164 | -0.2415 | -0.1229 | 0.5188 |
| 净 10bp | -0.0692 | 0.0868 | -0.7969 | -0.4024 | -0.3459 | 0.4881 |

## 分年度扩展窗口结果

分年度 RankIC、毛 Sharpe、净 3bp / 5bp Sharpe：

| fold_id | test_year | rank_ic_mean | rank_ic_hit_rate | gross_sharpe | net_3bps_sharpe | net_5bps_sharpe | gross_annual_return | strategy_days | n_factors_used_mean | prediction_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_2019 | 2019 | 0.0104 | 0.5328 | 0.3547 | -0.0075 | -0.2491 | 0.0296 | 49 | 92.0000 | 1.0000 |
| fold_2020 | 2020 | 0.0667 | 0.6132 | 1.9835 | 1.6229 | 1.3826 | 0.1664 | 49 | 92.0000 | 1.0000 |
| fold_2021 | 2021 | 0.0143 | 0.5226 | -0.6717 | -0.9595 | -1.1514 | -0.0706 | 49 | 92.0000 | 1.0000 |
| fold_2022 | 2022 | 0.0309 | 0.5661 | 0.9703 | 0.6833 | 0.4920 | 0.1022 | 49 | 92.0000 | 1.0000 |
| fold_2023 | 2023 | 0.0055 | 0.5124 | 0.2894 | -0.1481 | -0.4397 | 0.0200 | 49 | 92.0000 | 1.0000 |
| fold_2024 | 2024 | -0.0311 | 0.4449 | -0.9269 | -1.3954 | -1.7077 | -0.0598 | 48 | 92.0000 | 1.0000 |

## 覆盖率与 n_factors_used 摘要

| 指标 | 结果 |
| --- | --- |
| 策略调仓日数 | 293 |
| 预测交易日数 | 1450 |
| 策略覆盖率 | 0.2021 |
| 预测行覆盖率 | 1.0000 |
| n_factors_used 均值 | 92.0000 |
| n_factors_used P25 | 92.0000 |
| n_factors_used P50 | 92.0000 |
| n_factors_used P75 | 92.0000 |

## 使用边界

1. 点时：因子在 T 日收盘后形成，信号仅可用于 T+1 开盘交易；切分、方向选择与评价均未使用测试段。
2. 本模型是全体等权多因子地板基准，**不是**单因子排行榜，也**不是** Ridge / OLS 加权结果。
3. 弱因子与强因子同权；相关因子会重复计入同一信息——这是本版主动接受的代价。
4. 成本仅为单边 0/3/5/10 基点敏感性，不得解读为已核验的真实手续费或冲击成本。
5. 2025 未参与主评价；若另做补充样本外，须单独配置与目录，并标注为补充证据、非封存测试。
6. 未核验合约乘数、涨跌停、成交容量与真实交易成本，结果仅属研究回测。
7. 「全体」指 handoff 注册表全部字段（当前 92），与 Ridge 输入宇宙一致；弱信号与 quality_flag 同权是本地板基准主动接受的代价。
