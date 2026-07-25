# 2025年单因子排序模型补充回测报告（协议口径）

> 每个因子的方向仅用开发期训练窗冻结；2025 只做一次评价，不参与选因子。

## 测试边界

- 训练：`training_handoff` 2015—2024。
- 测试：`ml_2025_backtest_pack`。
- 字段数：92；评价因子数：92。
- 执行：`future_return_5d`，持有/调仓 5/5 日。
- 成本：0/3/5/10 bp；主口径 5 bp。

## 核心发现

- predictor（或全体）平均 RankIC 为 -0.0098，RankIC>0 占比 0.3611。
- predictor（或全体）平均净 5bp Sharpe 为 -1.7259。
- 按 RankIC 排名第一：`amihud_20__csz`（RankIC=0.1114）。

## Top 排行榜（按 RankIC）

|   rank | feature_name                     | usage_type   |   rank_ic_mean |   rank_ic_hit_rate |   gross_sharpe |   net_5bps_sharpe |   gross_annual_return |   net_5bps_annual_return |   applied_direction | direction_source   |
|-------:|:---------------------------------|:-------------|---------------:|-------------------:|---------------:|------------------:|----------------------:|-------------------------:|--------------------:|:-------------------|
|      1 | amihud_20__csz                   | risk         |         0.1114 |             0.7131 |         2.9267 |            2.2403 |                0.2149 |                   0.1645 |                  -1 | catalog            |
|      2 | warehouse_surprise__missing      | quality_flag |         0.0863 |             0.6875 |        -2.596  |           -3.8735 |               -0.1024 |                  -0.1528 |                   1 | catalog            |
|      3 | warehouse_tightness__csz         | predictor    |         0.0644 |             0.6538 |         1.2064 |            0.1544 |                0.0578 |                   0.0074 |                   1 | catalog            |
|      4 | idiosyncratic_volatility_60__csz | risk         |         0.0505 |             0.5485 |         1.1409 |            0.5407 |                0.0958 |                   0.0454 |                   1 | train_rank_ic      |
|      5 | multi_horizon_trend__csz         | predictor    |         0.0497 |             0.5907 |         1.2283 |            0.6389 |                0.105  |                   0.0546 |                   1 | catalog            |
|      6 | volume_price_divergence_20__csz  | predictor    |         0.035  |             0.5654 |         0.8954 |            0.4071 |                0.0924 |                   0.042  |                   1 | train_rank_ic      |
|      7 | carry_annualized__csz            | predictor    |         0.0322 |             0.5823 |         0.4379 |           -0.0423 |                0.046  |                  -0.0044 |                   1 | catalog            |
|      8 | warehouse_tightness__missing     | quality_flag |         0.0311 |             0.5676 |        -2.5885 |           -3.7071 |               -0.1166 |                  -0.167  |                   1 | catalog            |
|      9 | volume_shock_20__csz             | conditional  |         0.0288 |             0.5232 |         1.4658 |            0.7549 |                0.1039 |                   0.0535 |                  -1 | train_rank_ic      |
|     10 | turnover_oi__csz                 | conditional  |         0.0265 |             0.5823 |         1.6721 |            0.9835 |                0.1224 |                   0.072  |                   1 | train_rank_ic      |
|     11 | tsmom_60__csz                    | predictor    |         0.0252 |             0.5401 |         0.4565 |           -0.1048 |                0.041  |                  -0.0094 |                   1 | catalog            |
|     12 | momentum_quality_60__csz         | predictor    |         0.0251 |             0.5485 |         0.4272 |           -0.1342 |                0.0384 |                  -0.012  |                   1 | catalog            |
|     13 | low_volatility_20__csz           | risk         |         0.0196 |             0.5401 |         0.0688 |           -0.4448 |                0.0068 |                  -0.0436 |                   1 | catalog            |
|     14 | inventory_acceleration__csz      | predictor    |         0.0192 |             0.5865 |         0.357  |           -0.7313 |                0.0165 |                  -0.0339 |                   1 | catalog            |
|     15 | term_structure_slope__csz        | predictor    |         0.0172 |             0.5696 |        -0.0665 |           -0.5043 |               -0.0077 |                  -0.0581 |                   1 | catalog            |
|     16 | trend_efficiency_60__csz         | predictor    |         0.014  |             0.4979 |        -0.7272 |           -1.3754 |               -0.0566 |                  -0.107  |                   1 | train_rank_ic      |
|     17 | vol_managed_momentum__csz        | predictor    |         0.0113 |             0.557  |         0.0966 |           -0.8565 |                0.0051 |                  -0.0453 |                   1 | catalog            |
|     18 | csmom_60__csz                    | predictor    |         0.0105 |             0.5148 |        -0.0984 |           -0.5464 |               -0.0111 |                  -0.0615 |                   1 | catalog            |
|     19 | fundamental_underreaction__csz   | predictor    |         0.0057 |             0.5417 |        -2.0003 |           -3.0919 |               -0.0924 |                  -0.1428 |                   1 | catalog            |
|     20 | short_reversal_5__csz            | predictor    |         0.0045 |             0.5148 |         1.0074 |            0.476  |                0.0956 |                   0.0452 |                   1 | catalog            |

## 使用限制

1. 不得用 2025 结果回头增删因子或改方向规则后再把同一年当最终成绩。
2. quality_flag / `__missing` 字段可出现在全量表中，解读时需与 predictor 区分。

