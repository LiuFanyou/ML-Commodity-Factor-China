# Ridge线性回归模型测试报告

> 本报告由 handoff 矩阵接口的 92 个注册字段自动生成；2025不参与模型选择或评价。

## Handoff 数据接口

| 项目        | 值                                                                  |
|:------------|:--------------------------------------------------------------------|
| handoff目录 | D:\软件2\AI-15-demo-main\--main2\--main - 副本 (3)\training_handoff |
| 特征文件    | ml_features_development.csv.gz                                      |
| 注册表      | ml_feature_registry.csv                                             |
| 标签文件    | labels_development.csv.gz                                           |
| 折定义      | fold_definitions.csv                                                |
| 特征数      | 92                                                                  |
| 特征行数    | 70822                                                               |
| 特征日期    | 2015-01-05 — 2024-12-31                                             |
| 品种数      | 30                                                                  |

## 模型与标签选择

- 模型：带 L2 正则的 Ridge 线性回归，全部 92 个注册字段统一输入。
- 原始收益标签：`future_return_5d`（T 收盘形成信号后，同一真实合约从 T+1 开盘持有到 T+h+1 开盘）。
- 训练标签：同日横截面中心化收益排名 `target_cs_rank_5d`，范围[-1,1]。
- 正则参数：每个扩展训练窗口内部用最后一年验证，按日均 RankIC 选择。
- 外层验证：读取 `fold_definitions.csv` 的扩展窗口（2015 起点不变，2019–2024 逐年验证）；训练净化规则为 `label_end_date_5d < valid_start`。

## 策略定义

- 每个调仓信号日按预测值排序，最高20%做多、最低20%做空。
- 多头和空头各占0.5风险资金，组合净敞口0、总敞口1。
- 持有 5 个交易日，每 5 个交易日调仓。
- 主成本口径单边 5 bp；敏感性报告单边 0/3/5/10 bp。
- 真实交易所手续费仅作辅助诊断，不替代统一成本口径。

## 输入字段组成

| usage_type   | category              |   字段数 |
|:-------------|:----------------------|---------:|
| conditional  | meta_factor           |        1 |
| conditional  | trading_liquidity     |        4 |
| predictor    | basis_spot            |        4 |
| predictor    | factor_interaction    |        2 |
| predictor    | industry_chain        |        1 |
| predictor    | information_news      |        7 |
| predictor    | inventory_fundamental |        5 |
| predictor    | term_structure        |        6 |
| predictor    | trading_liquidity     |        2 |
| predictor    | trend_momentum        |        6 |
| predictor    | value_reversion       |        3 |
| quality_flag | data_quality          |       45 |
| risk         | trading_liquidity     |        1 |
| risk         | volatility_risk       |        5 |

## 总体样本外预测结果

| 指标           | 结果      |
|:---------------|:----------|
| 样本外年份     | 2019—2024 |
| 样本外记录     | 43234     |
| 交易日         | 1450      |
| 调仓日         | 293       |
| 平均Pearson IC | 0.0239    |
| 平均RankIC     | 0.0247    |
| RankIC胜率     | 0.5448    |
| 目标R²         | -0.0064   |
| 目标MAE        | 0.5172    |

## 策略总体结果

| 成本     |   annual_return |   annual_volatility |   sharpe |   max_drawdown |   total_return |   hit_rate |
|:---------|----------------:|--------------------:|---------:|---------------:|---------------:|-----------:|
| 毛收益   |          0.0325 |              0.0817 |   0.3973 |        -0.1373 |         0.1846 |     0.5461 |
| 单边0bp  |          0.0325 |              0.0817 |   0.3973 |        -0.1373 |         0.1846 |     0.5461 |
| 单边3bp  |          0.0022 |              0.0817 |   0.0273 |        -0.214  |        -0.0063 |     0.5222 |
| 单边5bp  |         -0.0179 |              0.0817 |  -0.2194 |        -0.2614 |        -0.1162 |     0.4915 |
| 单边10bp |         -0.0683 |              0.0817 |  -0.8362 |        -0.3893 |        -0.3409 |     0.4642 |

## 分年度扩展窗口结果

| fold_id   |   test_year |   selected_alpha | train_start   | train_end   |   train_rows |   test_rows |   test_days |   rebalance_days |   pearson_ic_mean |   rank_ic_mean |   rank_icir |   rank_ic_hit_rate |   target_r2 |   target_mae |   gross_annual_return |   gross_annual_volatility |   gross_sharpe |   gross_max_drawdown |   gross_total_return |   gross_hit_rate |   net_0bps_annual_return |   net_0bps_annual_volatility |   net_0bps_sharpe |   net_0bps_max_drawdown |   net_0bps_total_return |   net_0bps_hit_rate |   net_3bps_annual_return |   net_3bps_annual_volatility |   net_3bps_sharpe |   net_3bps_max_drawdown |   net_3bps_total_return |   net_3bps_hit_rate |   net_5bps_annual_return |   net_5bps_annual_volatility |   net_5bps_sharpe |   net_5bps_max_drawdown |   net_5bps_total_return |   net_5bps_hit_rate |   net_10bps_annual_return |   net_10bps_annual_volatility |   net_10bps_sharpe |   net_10bps_max_drawdown |   net_10bps_total_return |   net_10bps_hit_rate |
|:----------|------------:|-----------------:|:--------------|:------------|-------------:|------------:|------------:|-----------------:|------------------:|---------------:|------------:|-------------------:|------------:|-------------:|----------------------:|--------------------------:|---------------:|---------------------:|---------------------:|-----------------:|-------------------------:|-----------------------------:|------------------:|------------------------:|------------------------:|--------------------:|-------------------------:|-----------------------------:|------------------:|------------------------:|------------------------:|--------------------:|-------------------------:|-----------------------------:|------------------:|------------------------:|------------------------:|--------------------:|--------------------------:|------------------------------:|-------------------:|-------------------------:|-------------------------:|---------------------:|
| fold_2019 |        2019 |            100   | 2015-01-05    | 2018-12-20  |        26220 |        7092 |         244 |               49 |            0.0667 |         0.0553 |      0.2222 |             0.5943 |     -0.0001 |       0.5157 |                0.1163 |                    0.0728 |         1.5962 |              -0.0519 |               0.1167 |           0.6327 |                   0.1163 |                       0.0728 |            1.5962 |                 -0.0519 |                  0.1167 |              0.6327 |                   0.086  |                       0.0728 |            1.1811 |                 -0.061  |                  0.0844 |              0.6327 |                   0.0659 |                       0.0728 |            0.9043 |                 -0.067  |                  0.0634 |              0.6122 |                    0.0155 |                        0.0728 |             0.2123 |                  -0.0848 |                   0.0126 |               0.551  |
| fold_2020 |        2020 |            100   | 2015-01-05    | 2019-12-23  |        33306 |        7268 |         243 |               49 |           -0.0048 |         0.0203 |      0.0987 |             0.5021 |     -0.0116 |       0.5179 |               -0.0216 |                    0.0871 |        -0.2476 |              -0.0679 |              -0.0243 |           0.4898 |                  -0.0216 |                       0.0871 |           -0.2476 |                 -0.0679 |                 -0.0243 |              0.4898 |                  -0.0518 |                       0.0871 |           -0.5947 |                 -0.0828 |                 -0.0526 |              0.4694 |                  -0.072  |                       0.0871 |           -0.8261 |                 -0.0952 |                 -0.071  |              0.3878 |                   -0.1224 |                        0.0871 |            -1.4045 |                  -0.1256 |                  -0.1155 |               0.3878 |
| fold_2021 |        2021 |            100   | 2015-01-05    | 2020-12-23  |        40574 |        7287 |         243 |               49 |           -0.0009 |         0.0207 |      0.0883 |             0.5267 |     -0.0073 |       0.517  |               -0.0482 |                    0.0815 |        -0.5917 |              -0.0944 |              -0.0489 |           0.5102 |                  -0.0482 |                       0.0815 |           -0.5917 |                 -0.0944 |                 -0.0489 |              0.5102 |                  -0.0785 |                       0.0815 |           -0.9627 |                 -0.1124 |                 -0.0765 |              0.4694 |                  -0.0986 |                       0.0815 |           -1.21   |                 -0.1244 |                 -0.0944 |              0.449  |                   -0.149  |                        0.0815 |            -1.8282 |                  -0.1539 |                  -0.1378 |               0.4286 |
| fold_2022 |        2022 |            100   | 2015-01-05    | 2021-12-23  |        47861 |        7252 |         242 |               49 |            0.0233 |         0.0238 |      0.0949 |             0.562  |     -0.0043 |       0.517  |                0.061  |                    0.0862 |         0.7073 |              -0.0646 |               0.0573 |           0.551  |                   0.061  |                       0.0862 |            0.7073 |                 -0.0646 |                  0.0573 |              0.551  |                   0.0308 |                       0.0862 |            0.3566 |                 -0.0692 |                  0.0267 |              0.551  |                   0.0106 |                       0.0862 |            0.1228 |                 -0.0767 |                  0.0068 |              0.5306 |                   -0.0398 |                        0.0862 |            -0.4617 |                  -0.0979 |                  -0.0414 |               0.5306 |
| fold_2023 |        2023 |              1   | 2015-01-05    | 2022-12-22  |        55113 |        7259 |         242 |               49 |            0.0588 |         0.0319 |      0.1299 |             0.5744 |     -0.0034 |       0.5169 |                0.0733 |                    0.0745 |         0.9841 |              -0.1038 |               0.071  |           0.6327 |                   0.0733 |                       0.0745 |            0.9841 |                 -0.1038 |                  0.071  |              0.6327 |                   0.0431 |                       0.0745 |            0.5784 |                 -0.1109 |                  0.04   |              0.5918 |                   0.0229 |                       0.0745 |            0.3079 |                 -0.1158 |                  0.0198 |              0.5714 |                   -0.0275 |                        0.0745 |            -0.3684 |                  -0.1282 |                  -0.0289 |               0.5102 |
| fold_2024 |        2024 |              0.1 | 2015-01-05    | 2023-12-21  |        62372 |        7076 |         236 |               48 |           -0.0003 |        -0.0048 |     -0.0204 |             0.5085 |     -0.0115 |       0.5189 |                0.0136 |                    0.0887 |         0.1538 |              -0.0877 |               0.0095 |           0.4583 |                   0.0136 |                       0.0887 |            0.1538 |                 -0.0877 |                  0.0095 |              0.4583 |                  -0.0166 |                       0.0887 |           -0.1872 |                 -0.1014 |                 -0.0192 |              0.4167 |                  -0.0368 |                       0.0887 |           -0.4145 |                 -0.1104 |                 -0.0379 |              0.3958 |                   -0.0872 |                        0.0887 |            -0.9827 |                  -0.1324 |                  -0.083  |               0.375  |

## 训练窗口内部正则参数选择

| fold_id   |   outer_test_year |   inner_validation_year |   alpha |   mean_rank_ic |   rows |
|:----------|------------------:|------------------------:|--------:|---------------:|-------:|
| fold_2019 |              2019 |                    2018 |  100    |         0.0209 |   6690 |
| fold_2019 |              2019 |                    2018 |    1    |         0.0168 |   6690 |
| fold_2019 |              2019 |                    2018 |    0.1  |         0.016  |   6690 |
| fold_2019 |              2019 |                    2018 |   10    |         0.0155 |   6690 |
| fold_2019 |              2019 |                    2018 |    0.01 |         0.015  |   6690 |
| fold_2020 |              2020 |                    2019 |  100    |         0.0549 |   6912 |
| fold_2020 |              2020 |                    2019 |    0.01 |         0.0492 |   6912 |
| fold_2020 |              2020 |                    2019 |    0.1  |         0.0491 |   6912 |
| fold_2020 |              2020 |                    2019 |   10    |         0.0491 |   6912 |
| fold_2020 |              2020 |                    2019 |    1    |         0.0485 |   6912 |
| fold_2021 |              2021 |                    2020 |  100    |         0.0212 |   7088 |
| fold_2021 |              2021 |                    2020 |   10    |         0.0144 |   7088 |
| fold_2021 |              2021 |                    2020 |    1    |         0.013  |   7088 |
| fold_2021 |              2021 |                    2020 |    0.01 |         0.0123 |   7088 |
| fold_2021 |              2021 |                    2020 |    0.1  |         0.0123 |   7088 |
| fold_2022 |              2022 |                    2021 |  100    |         0.0156 |   7107 |
| fold_2022 |              2022 |                    2021 |   10    |         0.0153 |   7107 |
| fold_2022 |              2022 |                    2021 |    1    |         0.014  |   7107 |
| fold_2022 |              2022 |                    2021 |    0.1  |         0.0139 |   7107 |
| fold_2022 |              2022 |                    2021 |    0.01 |         0.0138 |   7107 |
| fold_2023 |              2023 |                    2022 |    1    |         0.0295 |   7072 |
| fold_2023 |              2023 |                    2022 |    0.01 |         0.0293 |   7072 |
| fold_2023 |              2023 |                    2022 |    0.1  |         0.0292 |   7072 |
| fold_2023 |              2023 |                    2022 |   10    |         0.0284 |   7072 |
| fold_2023 |              2023 |                    2022 |  100    |         0.0241 |   7072 |
| fold_2024 |              2024 |                    2023 |    0.1  |         0.0324 |   7079 |
| fold_2024 |              2024 |                    2023 |    0.01 |         0.0322 |   7079 |
| fold_2024 |              2024 |                    2023 |    1    |         0.032  |   7079 |
| fold_2024 |              2024 |                    2023 |   10    |         0.0319 |   7079 |
| fold_2024 |              2024 |                    2023 |  100    |         0.031  |   7079 |

## 标准化系数绝对值前20名

系数反映条件关联而非因果；相关特征可能相互分摊权重，高相关字段按协议保留。

| feature_name                        |   coefficient_mean |   coefficient_std |   mean_abs_coefficient |   positive_share |   folds |   sign_consistency | source_factor              | usage_type   | category        |   correlation_cluster | cluster_representative           |
|:------------------------------------|-------------------:|------------------:|-----------------------:|-----------------:|--------:|-------------------:|:---------------------------|:-------------|:----------------|----------------------:|:---------------------------------|
| inventory_price_divergence__missing |            -0.0494 |            0.0673 |                 0.0494 |              0   |       6 |                1   | inventory_price_divergence | quality_flag | data_quality    |                    23 | inventory_change_5__missing      |
| tsmom_60__csz                       |            -0.0356 |            0.0097 |                 0.0356 |              0   |       6 |                1   | tsmom_60                   | predictor    | trend_momentum  |                     1 | tsmom_60__csz                    |
| inventory_change_5__missing         |             0.0327 |            0.0432 |                 0.0327 |              1   |       6 |                1   | inventory_change_5         | quality_flag | data_quality    |                    23 | inventory_change_5__missing      |
| carry_change_20__missing            |            -0.0309 |            0.0021 |                 0.0309 |              0   |       6 |                1   | carry_change_20            | quality_flag | data_quality    |                    10 | carry_change_20__missing         |
| basis_momentum_20__csz              |            -0.0269 |            0.0042 |                 0.0269 |              0   |       6 |                1   | basis_momentum_20          | predictor    | basis_spot      |                    14 | basis_momentum_20__csz           |
| vol_managed_momentum__csz           |             0.0235 |            0.0021 |                 0.0235 |              1   |       6 |                1   | vol_managed_momentum       | predictor    | trend_momentum  |                    51 | vol_managed_momentum__csz        |
| volume_price_divergence_20__missing |             0.0224 |            0.021  |                 0.0224 |              1   |       6 |                1   | volume_price_divergence_20 | quality_flag | data_quality    |                    33 | price_oi_interaction_20__missing |
| low_volatility_20__csz              |            -0.0209 |            0.0137 |                 0.0209 |              0   |       6 |                1   | low_volatility_20          | risk         | volatility_risk |                    38 | realized_volatility_20__csz      |
| inventory_acceleration__missing     |             0.019  |            0.0224 |                 0.019  |              1   |       6 |                1   | inventory_acceleration     | quality_flag | data_quality    |                    23 | inventory_change_5__missing      |
| momentum_quality_60__csz            |             0.0173 |            0.0058 |                 0.0173 |              1   |       6 |                1   | momentum_quality_60        | predictor    | trend_momentum  |                    50 | momentum_quality_60__csz         |
| basis_fresh__csz                    |             0.0137 |            0.0052 |                 0.0137 |              1   |       6 |                1   | basis_fresh                | predictor    | basis_spot      |                    11 | basis_fresh__csz                 |
| fundamental_composite_news__missing |            -0.0132 |            0.0073 |                 0.0132 |              0   |       6 |                1   | fundamental_composite_news | quality_flag | data_quality    |                    12 | basis_fresh__missing             |
| carry_annualized__missing           |             0.0131 |            0.001  |                 0.0131 |              1   |       6 |                1   | carry_annualized           | quality_flag | data_quality    |                     8 | carry_annualized__missing        |
| realized_volatility_20__csz         |            -0.0049 |            0.0148 |                 0.0128 |              0.5 |       6 |                0.5 | realized_volatility_20     | risk         | volatility_risk |                    38 | realized_volatility_20__csz      |
| seasonality_3y__csz                 |            -0.0126 |            0.0051 |                 0.0126 |              0   |       6 |                1   | seasonality_3y             | predictor    | value_reversion |                    40 | seasonality_3y__csz              |
| realized_skewness_60__csz           |            -0.0124 |            0.0022 |                 0.0124 |              0   |       6 |                1   | realized_skewness_60       | risk         | volatility_risk |                    39 | realized_skewness_60__csz        |
| price_oi_interaction_20__missing    |            -0.0121 |            0.0184 |                 0.0121 |              0   |       6 |                1   | price_oi_interaction_20    | quality_flag | data_quality    |                    33 | price_oi_interaction_20__missing |
| volume_shock_20__missing            |            -0.0118 |            0.0016 |                 0.0118 |              0   |       6 |                1   | volume_shock_20            | quality_flag | data_quality    |                    35 | volume_shock_20__missing         |
| information_divergence__missing     |             0.0117 |            0.013  |                 0.0117 |              1   |       6 |                1   | information_divergence     | quality_flag | data_quality    |                    29 | tightness_composite__missing     |
| value_252__missing                  |            -0.0114 |            0.0037 |                 0.0114 |              0   |       6 |                1   | value_252                  | quality_flag | data_quality    |                    18 | value_252__missing               |

## 使用边界

1. 本测试验证的是协议主策略：T收盘信号 + T+1开盘成交 + 5日持有/调仓。
2. 若成本后收益为负，不能用毛收益结果声称策略可实盘。
3. 2025不在本 handoff 包内，不得作为开发阶段独立封存期窥视对象。
4. 没有核验合约乘数、真实手续费、冲击、涨跌停和成交容量，结果只属于研究回测。
5. 本模型不负责因子加工；更换因子时只需替换 handoff 矩阵文件。
