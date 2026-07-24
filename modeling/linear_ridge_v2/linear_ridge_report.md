# Ridge线性回归模型测试报告（筛弱/去重 v2）

> 本报告：先验 predictor（N=36）+ 折内训练窗剔弱/去相关后，对入选集做 L2 Ridge；2025不参与模型选择或评价。

## Handoff 数据接口

| 项目                | 值                                                                  |
|:--------------------|:--------------------------------------------------------------------|
| handoff目录         | D:\软件2\AI-15-demo-main\--main2\--main - 副本 (3)\training_handoff |
| 特征文件            | ml_features_development.csv.gz                                      |
| 注册表              | ml_feature_registry.csv                                             |
| 标签文件            | labels_development.csv.gz                                           |
| 折定义              | fold_definitions.csv                                                |
| 注册特征数(handoff) | 92                                                                  |
| 本模型先验宇宙      | 36                                                                  |
| 特征行数            | 70822                                                               |
| 特征日期            | 2015-01-05 — 2024-12-31                                             |
| 品种数              | 30                                                                  |

## 模型与标签选择

- 模型：`linear_ridge_v2` / Ridge + `selection_mode=train_ic_corr`；先验 `['predictor']`，折均入选 12.7。
- 原始收益标签：`future_return_5d`（T 收盘形成信号后，同一真实合约从 T+1 开盘持有到 T+h+1 开盘）。
- 训练标签：同日横截面中心化收益排名 `target_cs_rank_5d`，范围[-1,1]。
- 正则参数：每个扩展训练窗口内部用最后一年验证，按日均 RankIC 选择（仅在入选集上）。
- 外层验证：读取 `fold_definitions.csv` 的扩展窗口（2015 起点不变，2019–2024 逐年验证）；训练净化规则为 `label_end_date_5d < valid_start`。

## 策略定义

- 每个调仓信号日按预测值排序，最高20%做多、最低20%做空。
- 多头和空头各占0.5风险资金，组合净敞口0、总敞口1。
- 持有 5 个交易日，每 5 个交易日调仓。
- 主成本口径单边 5 bp；敏感性报告单边 0/3/5/10 bp。
- 真实交易所手续费仅作辅助诊断，不替代统一成本口径。

## 输入字段组成（先验宇宙）

| usage_type   | category              |   字段数 |
|:-------------|:----------------------|---------:|
| predictor    | basis_spot            |        4 |
| predictor    | factor_interaction    |        2 |
| predictor    | industry_chain        |        1 |
| predictor    | information_news      |        7 |
| predictor    | inventory_fundamental |        5 |
| predictor    | term_structure        |        6 |
| predictor    | trading_liquidity     |        2 |
| predictor    | trend_momentum        |        6 |
| predictor    | value_reversion       |        3 |

## 折内训练窗筛选摘要

- `selection_mode = train_ic_corr`：仅用该折训练窗；**未**使用样本外 IC/Sharpe 挑因子。
- 剔弱阈值 `min_train_rank_ic = 0.01`。
- 去重阈值 `max_pairwise_corr = 0.7`。
- 筛选后在入选集上再选 `alpha` 并拟合 Ridge。

各折入选数量与名单：

| fold_id   |   test_year |   n_selected | selected_features                                                                                                                                                                                                                                                                                                                                                                                                  |
|:----------|------------:|-------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| fold_2019 |        2019 |           15 | basis_fresh__csz, carry_annualized__csz, information_divergence__csz, inventory_price_divergence__csz, inventory_scarcity__csz, inventory_surprise__csz, momentum_carry_interaction__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, vol_managed_momentum__csz, volume_price_divergence_20__csz, warehouse_surprise__csz, warehouse_tightness__csz |
| fold_2020 |        2020 |           12 | basis_fresh__csz, basis_shock_20__csz, information_divergence__csz, inventory_scarcity__csz, inventory_surprise__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, vol_managed_momentum__csz, volume_price_divergence_20__csz, warehouse_tightness__csz                                                                                              |
| fold_2021 |        2021 |           13 | basis_fresh__csz, information_divergence__csz, inventory_price_divergence__csz, inventory_scarcity__csz, inventory_surprise__csz, nonlinear_tightness__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, vol_managed_momentum__csz, volume_price_divergence_20__csz, warehouse_surprise__csz                                                         |
| fold_2022 |        2022 |           12 | basis_fresh__csz, information_divergence__csz, inventory_price_divergence__csz, inventory_scarcity__csz, inventory_surprise__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, vol_managed_momentum__csz, volume_price_divergence_20__csz, warehouse_tightness__csz                                                                                  |
| fold_2023 |        2023 |           13 | basis_fresh__csz, fundamental_underreaction__csz, information_divergence__csz, inventory_acceleration__csz, inventory_change_5__csz, inventory_price_divergence__csz, short_reversal_5__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, tightness_composite__csz, value_252__csz, vol_managed_momentum__csz                                                        |
| fold_2024 |        2024 |           11 | basis_fresh__csz, information_divergence__csz, inventory_acceleration__csz, inventory_change_5__csz, inventory_price_divergence__csz, nonlinear_tightness__csz, term_structure_acceleration__csz, term_structure_momentum__csz, term_structure_slope__csz, value_252__csz, warehouse_surprise__csz                                                                                                                 |

筛选原因计数（跨折汇总）：

| selection_reason           |   记录数 |
|:---------------------------|---------:|
| dropped_weak_train_rank_ic |      115 |
| selected                   |       76 |
| dropped_high_corr          |       25 |

## 总体样本外预测结果

| 指标           | 结果            |
|:---------------|:----------------|
| 模型版本       | linear_ridge_v2 |
| 筛选模式       | train_ic_corr   |
| 样本外年份     | 2019—2024       |
| 先验宇宙大小   | 36              |
| 折均入选数     | 12.6667         |
| 样本外记录     | 43234           |
| 交易日         | 1450            |
| 调仓日         | 293             |
| 平均Pearson IC | 0.0194          |
| 平均RankIC     | 0.0217          |
| RankIC胜率     | 0.5572          |
| 目标R²         | -0.0007         |
| 目标MAE        | 0.5168          |

## 与 v1（全体 Ridge）对照

同协议、同 handoff；v2 仅改变先验宇宙与折内训练窗筛选，再对入选集做 Ridge。

| 版本               | 宇宙/入选         |   RankIC |   毛Sharpe |   净5bp Sharpe |
|:-------------------|:------------------|---------:|-----------:|---------------:|
| v1 全体92 Ridge    | 92                |   0.0247 |     0.3973 |        -0.2194 |
| v2 筛弱/去重 Ridge | 36先验 / 折均12.7 |   0.0217 |     0.35   |        -0.2609 |

## 策略总体结果

| 成本     |   annual_return |   annual_volatility |   sharpe |   max_drawdown |   total_return |   hit_rate |
|:---------|----------------:|--------------------:|---------:|---------------:|---------------:|-----------:|
| 毛收益   |          0.0289 |              0.0825 |   0.35   |        -0.1733 |         0.1596 |     0.5529 |
| 单边0bp  |          0.0289 |              0.0825 |   0.35   |        -0.1733 |         0.1596 |     0.5529 |
| 单边3bp  |         -0.0014 |              0.0825 |  -0.0166 |        -0.2432 |        -0.0273 |     0.5017 |
| 单边5bp  |         -0.0215 |              0.0825 |  -0.2609 |        -0.2925 |        -0.1349 |     0.4846 |
| 单边10bp |         -0.0719 |              0.0825 |  -0.8719 |        -0.4288 |        -0.3548 |     0.4539 |

## 分年度扩展窗口结果

| fold_id   |   test_year |   selected_alpha | selection_mode   |   prior_universe_size |   universe_size | train_start   | train_end   |   train_rows |   test_rows |   test_days |   rebalance_days |   pearson_ic_mean |   rank_ic_mean |   rank_icir |   rank_ic_hit_rate |   target_r2 |   target_mae |   gross_annual_return |   gross_annual_volatility |   gross_sharpe |   gross_max_drawdown |   gross_total_return |   gross_hit_rate |   net_0bps_annual_return |   net_0bps_annual_volatility |   net_0bps_sharpe |   net_0bps_max_drawdown |   net_0bps_total_return |   net_0bps_hit_rate |   net_3bps_annual_return |   net_3bps_annual_volatility |   net_3bps_sharpe |   net_3bps_max_drawdown |   net_3bps_total_return |   net_3bps_hit_rate |   net_5bps_annual_return |   net_5bps_annual_volatility |   net_5bps_sharpe |   net_5bps_max_drawdown |   net_5bps_total_return |   net_5bps_hit_rate |   net_10bps_annual_return |   net_10bps_annual_volatility |   net_10bps_sharpe |   net_10bps_max_drawdown |   net_10bps_total_return |   net_10bps_hit_rate |
|:----------|------------:|-----------------:|:-----------------|----------------------:|----------------:|:--------------|:------------|-------------:|------------:|------------:|-----------------:|------------------:|---------------:|------------:|-------------------:|------------:|-------------:|----------------------:|--------------------------:|---------------:|---------------------:|---------------------:|-----------------:|-------------------------:|-----------------------------:|------------------:|------------------------:|------------------------:|--------------------:|-------------------------:|-----------------------------:|------------------:|------------------------:|------------------------:|--------------------:|-------------------------:|-----------------------------:|------------------:|------------------------:|------------------------:|--------------------:|--------------------------:|------------------------------:|-------------------:|-------------------------:|-------------------------:|---------------------:|
| fold_2019 |        2019 |               10 | train_ic_corr    |                    36 |              15 | 2015-01-05    | 2018-12-20  |        26220 |        7092 |         244 |               49 |            0.003  |         0.0011 |      0.0044 |             0.541  |     -0.0041 |       0.5179 |                0.0746 |                    0.0706 |         1.0563 |              -0.0625 |               0.0726 |           0.6531 |                   0.0746 |                       0.0706 |            1.0563 |                 -0.0625 |                  0.0726 |              0.6531 |                   0.0443 |                       0.0706 |            0.628  |                 -0.067  |                  0.0415 |              0.5918 |                   0.0242 |                       0.0706 |            0.3424 |                 -0.0722 |                  0.0213 |              0.5714 |                   -0.0262 |                        0.0706 |            -0.3716 |                  -0.086  |                  -0.0275 |               0.551  |
| fold_2020 |        2020 |              100 | train_ic_corr    |                    36 |              12 | 2015-01-05    | 2019-12-23  |        33306 |        7268 |         243 |               49 |            0.0649 |         0.0954 |      0.4636 |             0.7078 |      0.0053 |       0.5152 |                0.1346 |                    0.0895 |         1.5037 |              -0.0398 |               0.1352 |           0.6122 |                   0.1346 |                       0.0895 |            1.5037 |                 -0.0398 |                  0.1352 |              0.6122 |                   0.1043 |                       0.0895 |            1.1658 |                 -0.045  |                  0.1024 |              0.6122 |                   0.0842 |                       0.0895 |            0.9405 |                 -0.0484 |                  0.0811 |              0.6122 |                    0.0338 |                        0.0895 |             0.3772 |                  -0.057  |                   0.0294 |               0.5918 |
| fold_2021 |        2021 |              100 | train_ic_corr    |                    36 |              13 | 2015-01-05    | 2020-12-23  |        40574 |        7287 |         243 |               49 |            0.0155 |         0.0171 |      0.0765 |             0.535  |     -0.0002 |       0.5164 |               -0.018  |                    0.09   |        -0.2002 |              -0.0771 |              -0.0212 |           0.551  |                  -0.018  |                       0.09   |           -0.2002 |                 -0.0771 |                 -0.0212 |              0.551  |                  -0.0483 |                       0.09   |           -0.5362 |                 -0.0855 |                 -0.0496 |              0.4898 |                  -0.0684 |                       0.09   |           -0.7602 |                 -0.0938 |                 -0.068  |              0.449  |                   -0.1188 |                        0.09   |            -1.3202 |                  -0.1277 |                  -0.1127 |               0.4082 |
| fold_2022 |        2022 |              100 | train_ic_corr    |                    36 |              12 | 2015-01-05    | 2021-12-23  |        47861 |        7252 |         242 |               49 |            0.0122 |         0.0178 |      0.0779 |             0.5289 |     -0.0007 |       0.5166 |                0.0261 |                    0.0815 |         0.3204 |              -0.0447 |               0.0225 |           0.5306 |                   0.0261 |                       0.0815 |            0.3204 |                 -0.0447 |                  0.0225 |              0.5306 |                  -0.0041 |                       0.0815 |           -0.0504 |                 -0.051  |                 -0.0071 |              0.4898 |                  -0.0243 |                       0.0815 |           -0.2977 |                 -0.0552 |                 -0.0264 |              0.449  |                   -0.0747 |                        0.0815 |            -0.9157 |                  -0.0814 |                  -0.073  |               0.449  |
| fold_2023 |        2023 |               10 | train_ic_corr    |                    36 |              13 | 2015-01-05    | 2022-12-22  |        55113 |        7259 |         242 |               49 |            0.0163 |        -0.0031 |     -0.0131 |             0.5165 |     -0.004  |       0.5176 |               -0.0387 |                    0.0895 |        -0.4327 |              -0.1501 |              -0.0406 |           0.4898 |                  -0.0387 |                       0.0895 |           -0.4327 |                 -0.1501 |                 -0.0406 |              0.4898 |                  -0.069  |                       0.0895 |           -0.7706 |                 -0.1587 |                 -0.0685 |              0.449  |                  -0.0891 |                       0.0895 |           -0.9959 |                 -0.1644 |                 -0.0866 |              0.449  |                   -0.1395 |                        0.0895 |            -1.5591 |                  -0.1787 |                  -0.1304 |               0.3673 |
| fold_2024 |        2024 |              100 | train_ic_corr    |                    36 |              11 | 2015-01-05    | 2023-12-21  |        62372 |        7076 |         236 |               48 |            0.0039 |         0.0016 |      0.0069 |             0.5127 |     -0.0009 |       0.5172 |               -0.006  |                    0.073  |        -0.0819 |              -0.0824 |              -0.0081 |           0.4792 |                  -0.006  |                       0.073  |           -0.0819 |                 -0.0824 |                 -0.0081 |              0.4792 |                  -0.0362 |                       0.073  |           -0.4961 |                 -0.101  |                 -0.0363 |              0.375  |                  -0.0564 |                       0.073  |           -0.7722 |                 -0.1132 |                 -0.0546 |              0.375  |                   -0.1068 |                        0.073  |            -1.4625 |                  -0.1429 |                  -0.099  |               0.3542 |

## 训练窗口内部正则参数选择

| fold_id   |   outer_test_year |   n_selected_features |   inner_validation_year |   alpha |   mean_rank_ic |   rows |
|:----------|------------------:|----------------------:|------------------------:|--------:|---------------:|-------:|
| fold_2019 |              2019 |                    15 |                    2018 |   10    |         0.0652 |   6690 |
| fold_2019 |              2019 |                    15 |                    2018 |    1    |         0.0651 |   6690 |
| fold_2019 |              2019 |                    15 |                    2018 |    0.1  |         0.065  |   6690 |
| fold_2019 |              2019 |                    15 |                    2018 |    0.01 |         0.0649 |   6690 |
| fold_2019 |              2019 |                    15 |                    2018 |  100    |         0.0642 |   6690 |
| fold_2020 |              2020 |                    12 |                    2019 |  100    |         0.0111 |   6912 |
| fold_2020 |              2020 |                    12 |                    2019 |   10    |         0.0079 |   6912 |
| fold_2020 |              2020 |                    12 |                    2019 |    1    |         0.0078 |   6912 |
| fold_2020 |              2020 |                    12 |                    2019 |    0.01 |         0.0077 |   6912 |
| fold_2020 |              2020 |                    12 |                    2019 |    0.1  |         0.0077 |   6912 |
| fold_2021 |              2021 |                    13 |                    2020 |  100    |         0.0937 |   7088 |
| fold_2021 |              2021 |                    13 |                    2020 |    1    |         0.0935 |   7088 |
| fold_2021 |              2021 |                    13 |                    2020 |    0.01 |         0.0933 |   7088 |
| fold_2021 |              2021 |                    13 |                    2020 |    0.1  |         0.0933 |   7088 |
| fold_2021 |              2021 |                    13 |                    2020 |   10    |         0.0931 |   7088 |
| fold_2022 |              2022 |                    12 |                    2021 |  100    |         0.0155 |   7107 |
| fold_2022 |              2022 |                    12 |                    2021 |   10    |         0.0152 |   7107 |
| fold_2022 |              2022 |                    12 |                    2021 |    1    |         0.0151 |   7107 |
| fold_2022 |              2022 |                    12 |                    2021 |    0.01 |         0.0151 |   7107 |
| fold_2022 |              2022 |                    12 |                    2021 |    0.1  |         0.0151 |   7107 |
| fold_2023 |              2023 |                    13 |                    2022 |   10    |         0.0322 |   7072 |
| fold_2023 |              2023 |                    13 |                    2022 |    1    |         0.032  |   7072 |
| fold_2023 |              2023 |                    13 |                    2022 |  100    |         0.0318 |   7072 |
| fold_2023 |              2023 |                    13 |                    2022 |    0.1  |         0.0318 |   7072 |
| fold_2023 |              2023 |                    13 |                    2022 |    0.01 |         0.0318 |   7072 |
| fold_2024 |              2024 |                    11 |                    2023 |  100    |         0.0179 |   7079 |
| fold_2024 |              2024 |                    11 |                    2023 |   10    |         0.0157 |   7079 |
| fold_2024 |              2024 |                    11 |                    2023 |    1    |         0.0151 |   7079 |
| fold_2024 |              2024 |                    11 |                    2023 |    0.1  |         0.0151 |   7079 |
| fold_2024 |              2024 |                    11 |                    2023 |    0.01 |         0.0151 |   7079 |

## 标准化系数绝对值前20名

系数反映条件关联而非因果；相关特征可能相互分摊权重。

| feature_name                     |   coefficient_mean | coefficient_std   |   mean_abs_coefficient |   positive_share |   folds |   sign_consistency | source_factor               | usage_type   | category              |   correlation_cluster | cluster_representative           |
|:---------------------------------|-------------------:|:------------------|-----------------------:|-----------------:|--------:|-------------------:|:----------------------------|:-------------|:----------------------|----------------------:|:---------------------------------|
| information_divergence__csz      |             0.0159 | 0.0052            |                 0.0159 |              1   |       6 |                1   | information_divergence      | predictor    | information_news      |                    47 | information_divergence__csz      |
| tightness_composite__csz         |            -0.0141 | —                 |                 0.0141 |              0   |       1 |                1   | tightness_composite         | predictor    | inventory_fundamental |                    28 | tightness_composite__csz         |
| vol_managed_momentum__csz        |             0.0135 | 0.0026            |                 0.0135 |              1   |       5 |                1   | vol_managed_momentum        | predictor    | trend_momentum        |                    51 | vol_managed_momentum__csz        |
| value_252__csz                   |             0.0116 | 0.0043            |                 0.0116 |              1   |       6 |                1   | value_252                   | predictor    | value_reversion       |                    17 | value_252__csz                   |
| volume_price_divergence_20__csz  |             0.0116 | 0.0042            |                 0.0116 |              1   |       4 |                1   | volume_price_divergence_20  | predictor    | trading_liquidity     |                    62 | volume_price_divergence_20__csz  |
| nonlinear_tightness__csz         |            -0.0097 | 0.0045            |                 0.0097 |              0   |       2 |                1   | nonlinear_tightness         | predictor    | factor_interaction    |                    28 | tightness_composite__csz         |
| carry_annualized__csz            |            -0.0088 | —                 |                 0.0088 |              0   |       1 |                1   | carry_annualized            | predictor    | term_structure        |                     7 | carry_annualized__csz            |
| term_structure_acceleration__csz |             0.0071 | 0.0016            |                 0.0071 |              1   |       6 |                1   | term_structure_acceleration | predictor    | term_structure        |                    57 | term_structure_acceleration__csz |
| inventory_scarcity__csz          |             0.0071 | 0.0017            |                 0.0071 |              1   |       4 |                1   | inventory_scarcity          | predictor    | inventory_fundamental |                    20 | inventory_scarcity__csz          |
| warehouse_surprise__csz          |             0.007  | 0.0033            |                 0.007  |              1   |       3 |                1   | warehouse_surprise          | predictor    | information_news      |                    48 | warehouse_surprise__csz          |
| term_structure_slope__csz        |             0.0069 | 0.0044            |                 0.0069 |              1   |       6 |                1   | term_structure_slope        | predictor    | term_structure        |                    54 | term_structure_slope__csz        |
| momentum_carry_interaction__csz  |            -0.0063 | —                 |                 0.0063 |              0   |       1 |                1   | momentum_carry_interaction  | predictor    | factor_interaction    |                    63 | momentum_carry_interaction__csz  |
| basis_fresh__csz                 |             0.0054 | 0.0026            |                 0.0054 |              1   |       6 |                1   | basis_fresh                 | predictor    | basis_spot            |                    11 | basis_fresh__csz                 |
| basis_shock_20__csz              |             0.005  | —                 |                 0.005  |              1   |       1 |                1   | basis_shock_20              | predictor    | basis_spot            |                    13 | basis_shock_20__csz              |
| warehouse_tightness__csz         |             0.0047 | 0.0006            |                 0.0047 |              1   |       3 |                1   | warehouse_tightness         | predictor    | inventory_fundamental |                    26 | warehouse_tightness__csz         |
| term_structure_momentum__csz     |             0.0042 | 0.0023            |                 0.0042 |              1   |       6 |                1   | term_structure_momentum     | predictor    | term_structure        |                    55 | term_structure_momentum__csz     |
| fundamental_underreaction__csz   |             0.0036 | —                 |                 0.0036 |              1   |       1 |                1   | fundamental_underreaction   | predictor    | information_news      |                    46 | fundamental_underreaction__csz   |
| inventory_surprise__csz          |             0.0026 | 0.0013            |                 0.0026 |              1   |       4 |                1   | inventory_surprise          | predictor    | information_news      |                    24 | inventory_surprise__csz          |
| short_reversal_5__csz            |             0.0024 | —                 |                 0.0024 |              1   |       1 |                1   | short_reversal_5            | predictor    | value_reversion       |                    19 | short_reversal_5__csz            |
| inventory_price_divergence__csz  |             0.0011 | 0.0020            |                 0.0021 |              0.8 |       5 |                0.8 | inventory_price_divergence  | predictor    | information_news      |                    22 | inventory_change_5__csz          |

## 使用边界

1. 本测试验证的是协议主策略：T收盘信号 + T+1开盘成交 + 5日持有/调仓。
2. 若成本后收益为负，不能用毛收益结果声称策略可实盘。
3. 2025不在本 handoff 包内，不得作为开发阶段独立封存期窥视对象。
4. 没有核验合约乘数、真实手续费、冲击、涨跌停和成交容量，结果只属于研究回测。
5. 筛选与方向定号仅使用该折训练窗，未使用测试段。
6. 入选名单允许折间变化；阈值写死，未做网格搜索；v1 仍为 92 全体输入地板基准。
