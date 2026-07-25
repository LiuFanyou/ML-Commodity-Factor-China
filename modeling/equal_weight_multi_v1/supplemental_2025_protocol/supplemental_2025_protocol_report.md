# 2025年等权多因子模型补充回测报告（协议口径）

> 结论：**2025未提供正向预测证据**。方向仅用开发期训练窗冻结；测试只用 `ml_2025_backtest_pack`。

## 测试边界

- 训练：`training_handoff` 2015—2024；不在 2025 上重新选方向/筛因子。
- 测试：`ml_2025_backtest_pack`。
- 合成：direction-align → `cs_rank_centered` → 等权平均（`mean_available`）。
- 执行：`future_return_5d`，持有/调仓 5/5 日，多空各 20%。
- 成本：0/3/5/10 bp；主口径 5 bp。

## 核心发现

- 2025平均RankIC为-0.0629。
- 毛收益年化为-5.47%，Sharpe为-0.984。
- 单边3bp净年化为-8.49%；单边5bp净年化为-10.51%。

## 预测与回测指标

| 指标         | 结果                          |
|:-------------|:------------------------------|
| 训练数据     | training_handoff（2015—2024） |
| 测试数据包   | ml_2025_backtest_pack         |
| 训练区间     | 2015-01-05 至 2024-12-23      |
| 2025测试记录 | 7100                          |
| 调仓日       | 48                            |
| 入选因子数   | 92                            |
| 平均RankIC   | -0.0629                       |
| RankIC胜率   | 0.3924                        |
| 毛Sharpe     | -0.9844                       |
| 净5bp Sharpe | -1.8915                       |

## 成本敏感性

| 成本情景   |   annual_return |   annual_volatility |   sharpe |   max_drawdown |   total_return |   hit_rate |
|:-----------|----------------:|--------------------:|---------:|---------------:|---------------:|-----------:|
| 毛收益     |         -0.0547 |              0.0556 |  -0.9844 |        -0.0594 |        -0.0522 |     0.4167 |
| 单边0bp    |         -0.0547 |              0.0556 |  -0.9844 |        -0.0594 |        -0.0522 |     0.4167 |
| 单边3bp    |         -0.0849 |              0.0556 |  -1.5286 |        -0.0787 |        -0.0791 |     0.4167 |
| 单边5bp    |         -0.1051 |              0.0556 |  -1.8915 |        -0.0944 |        -0.0966 |     0.375  |
| 单边10bp   |         -0.1555 |              0.0556 |  -2.7985 |        -0.1331 |        -0.1391 |     0.3125 |

## 2025分月表现

| month   |   rebalance_days |   gross_return_total |   gross_return_hit_rate |   net_0bps_return_total |   net_0bps_return_hit_rate |   net_3bps_return_total |   net_3bps_return_hit_rate |   net_5bps_return_total |   net_5bps_return_hit_rate |   net_10bps_return_total |   net_10bps_return_hit_rate |
|:--------|-----------------:|---------------------:|------------------------:|------------------------:|---------------------------:|------------------------:|---------------------------:|------------------------:|---------------------------:|-------------------------:|----------------------------:|
| 2025-01 |                4 |              -0.0202 |                  0.25   |                 -0.0202 |                     0.25   |                 -0.0226 |                     0.25   |                 -0.0242 |                     0.25   |                  -0.0281 |                      0      |
| 2025-02 |                4 |              -0.0187 |                  0.25   |                 -0.0187 |                     0.25   |                 -0.0211 |                     0.25   |                 -0.0227 |                     0.25   |                  -0.0266 |                      0.25   |
| 2025-03 |                4 |              -0.0046 |                  0.25   |                 -0.0046 |                     0.25   |                 -0.007  |                     0.25   |                 -0.0086 |                     0.25   |                  -0.0126 |                      0.25   |
| 2025-04 |                4 |              -0.013  |                  0.25   |                 -0.013  |                     0.25   |                 -0.0154 |                     0.25   |                 -0.017  |                     0.25   |                  -0.0209 |                      0      |
| 2025-05 |                4 |              -0.0056 |                  0.25   |                 -0.0056 |                     0.25   |                 -0.008  |                     0.25   |                 -0.0096 |                     0.25   |                  -0.0135 |                      0.25   |
| 2025-06 |                4 |               0.0073 |                  0.5    |                  0.0073 |                     0.5    |                  0.0049 |                     0.5    |                  0.0033 |                     0.5    |                  -0.0008 |                      0.5    |
| 2025-07 |                4 |               0.0079 |                  0.75   |                  0.0079 |                     0.75   |                  0.0055 |                     0.75   |                  0.0039 |                     0.75   |                  -0.0001 |                      0.5    |
| 2025-08 |                5 |              -0.024  |                  0.2    |                 -0.024  |                     0.2    |                 -0.027  |                     0.2    |                 -0.0289 |                     0.2    |                  -0.0338 |                      0.2    |
| 2025-09 |                4 |               0.0021 |                  0.25   |                  0.0021 |                     0.25   |                 -0.0003 |                     0.25   |                 -0.0019 |                     0.25   |                  -0.0059 |                      0.25   |
| 2025-10 |                3 |               0.0051 |                  0.6667 |                  0.0051 |                     0.6667 |                  0.0033 |                     0.6667 |                  0.0021 |                     0.3333 |                  -0.0009 |                      0.3333 |
| 2025-11 |                4 |               0.0032 |                  0.75   |                  0.0032 |                     0.75   |                  0.0008 |                     0.75   |                 -0.0008 |                     0.5    |                  -0.0048 |                      0.5    |
| 2025-12 |                4 |               0.008  |                  0.75   |                  0.008  |                     0.75   |                  0.0056 |                     0.75   |                  0.004  |                     0.75   |                  -0      |                      0.75   |

## 训练窗冻结方向（摘要）

| feature_name                 |   catalog_direction |   resolved_direction | direction_source   |   train_rank_ic |
|:-----------------------------|--------------------:|---------------------:|:-------------------|----------------:|
| tsmom_60__csz                |                   1 |                    1 | catalog            |         -0.0174 |
| tsmom_60__missing            |                   1 |                    1 | catalog            |         -0.0059 |
| csmom_60__csz                |                   1 |                    1 | catalog            |         -0.0196 |
| csmom_60__missing            |                   1 |                    1 | catalog            |         -0.0059 |
| trend_efficiency_60__csz     |                   0 |                    1 | train_rank_ic      |          0.0009 |
| trend_efficiency_60__missing |                   0 |                   -1 | train_rank_ic      |          0.0059 |
| multi_horizon_trend__csz     |                   1 |                    1 | catalog            |         -0.0081 |
| multi_horizon_trend__missing |                   1 |                    1 | catalog            |          0.0035 |
| carry_annualized__csz        |                   1 |                    1 | catalog            |          0.0239 |
| carry_annualized__missing    |                   1 |                    1 | catalog            |         -0.007  |
| carry_change_20__csz         |                   1 |                    1 | catalog            |         -0.0064 |
| carry_change_20__missing     |                   1 |                    1 | catalog            |         -0.0338 |
| basis_fresh__csz             |                   1 |                    1 | catalog            |          0.0277 |
| basis_fresh__missing         |                   1 |                    1 | catalog            |          0.0254 |
| basis_shock_20__csz          |                   1 |                    1 | catalog            |          0.0047 |
| basis_shock_20__missing      |                   1 |                    1 | catalog            |          0.0252 |
| basis_momentum_20__csz       |                   1 |                    1 | catalog            |         -0.0167 |
| basis_momentum_20__missing   |                   1 |                    1 | catalog            |          0.0253 |
| curve_curvature__csz         |                   0 |                   -1 | train_rank_ic      |          0.0063 |
| curve_curvature__missing     |                   0 |                   -1 | train_rank_ic      |          0.0103 |
| value_252__csz               |                   1 |                    1 | catalog            |          0.0176 |
| value_252__missing           |                   1 |                    1 | catalog            |         -0.0039 |
| short_reversal_5__csz        |                   1 |                    1 | catalog            |          0.012  |
| inventory_scarcity__csz      |                   1 |                    1 | catalog            |          0.0011 |
| inventory_scarcity__missing  |                   1 |                    1 | catalog            |         -0.0065 |
| inventory_change_5__csz      |                   1 |                    1 | catalog            |          0.0122 |
| inventory_change_5__missing  |                   1 |                    1 | catalog            |         -0.0002 |
| inventory_surprise__csz      |                   1 |                    1 | catalog            |          0.0079 |
| inventory_surprise__missing  |                   1 |                    1 | catalog            |          0.003  |
| warehouse_tightness__csz     |                   1 |                    1 | catalog            |          0.0302 |

## 使用限制

1. 不得用本次 2025 结果回头修改方向规则、特征清单或持仓规则。
2. 若修改模型，应登记新版本并用更新年份验证。

