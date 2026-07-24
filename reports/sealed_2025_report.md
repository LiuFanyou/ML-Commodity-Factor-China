# 2025封存测试报告

本结果在代码、配置、开发期分类及特征注册表冻结后生成，且系统禁止覆盖。

| feature_name                          | classification   |   coverage |     ic_mean |   rank_ic_mean |   group_spread | sealed_direction_consistent   |
|:--------------------------------------|:-----------------|-----------:|------------:|---------------:|---------------:|:------------------------------|
| carry                                 | A                |          1 |  0.0481869  |     0.0425003  |    0.00154995  | True                          |
| term_structure_slope                  | A                |          1 |  0.0508331  |     0.0385882  |    0.00136111  | True                          |
| basis_momentum_20                     | B                |          1 | -0.031653   |    -0.0265384  |   -0.0021164   | False                         |
| basis_momentum_5                      | B                |          1 |  0.0333582  |     0.034737   |    0.00203543  | True                          |
| basis_momentum_60                     | B                |          1 | -0.00304795 |    -0.0358269  |   -0.00164224  | False                         |
| carry_change_20                       | B                |          1 | -0.0373159  |    -0.026531   |   -0.00202054  | False                         |
| carry_change_60                       | B                |          1 | -0.00492484 |    -0.0375282  |   -0.0019054   | False                         |
| curve_curvature                       | B                |          1 | -0.0128465  |    -0.0123854  |    1.37925e-05 | True                          |
| interaction__tsmom_20__oi_surprise_20 | B                |          1 | -0.0221608  |    -0.0116357  |   -0.00135852  | True                          |
| oi_surprise_60                        | B                |          1 |  0.00736006 |     0.0133893  |   -0.000631351 | True                          |
| reversal_20                           | B                |          1 | -0.0637893  |    -0.0221834  |   -0.00383749  | False                         |
| reversal_5                            | B                |          1 | -0.0442539  |    -0.00939419 |   -0.00153223  | False                         |
| seasonality_3y_month                  | B                |          1 |  0.0693938  |     0.0518242  |    0.00334738  | True                          |
| term_structure_momentum_60            | B                |          1 | -0.0816964  |    -0.0507286  |   -0.00549553  | False                         |
| tsmom_120                             | B                |          1 |  0.0936604  |     0.0623438  |    0.00415074  | True                          |
| trend_efficiency_120                  | C                |          1 |  0.0704223  |     0.0381993  |    0.00453191  | True                          |
| trend_efficiency_20                   | C                |          1 |  0.0454784  |     0.055264   |    0.00276008  | True                          |
| trend_efficiency_60                   | C                |          1 |  0.0146002  |     0.00667969 |   -0.000190021 | True                          |
| volume_oi_turnover                    | C                |          1 |  0.0403998  |     0.00987839 |    0.00307375  | True                          |
| volume_price_divergence_20            | C                |          1 |  0.00518159 |    -0.0107028  |    0.00100084  | True                          |
| volume_shock_20                       | C                |          1 |  0.0269234  |     0.0282517  |    0.0017303   | True                          |
| volume_shock_60                       | C                |          1 |  0.00230248 |     0.00648152 |    0.00131474  | True                          |
| amihud_20                             | D                |          1 | -0.0299485  |    -0.10764    |   -0.00648748  | True                          |
| correlation_risk_20_120               | D                |          1 |  0.00107514 |    -0.0206018  |    0.00167341  | True                          |
| idiosyncratic_vol_60                  | D                |          1 | -0.011283   |    -0.0157879  |    0.000545408 | True                          |
| realized_skew_120                     | D                |          1 | -0.0347178  |    -0.0372608  |    1.60207e-05 | True                          |
| realized_skew_60                      | D                |          1 |  0.0190644  |     0.0107253  |    0.00164076  | False                         |
| realized_vol_120                      | D                |          1 | -0.0805463  |    -0.0658229  |   -0.00452157  | True                          |
| vol_managed_momentum_120              | D                |          1 |  0.0462311  |     0.0448696  |    0.00186008  | True                          |
| vol_managed_momentum_20               | D                |          1 |  0.0510937  |     0.0626672  |    0.00408629  | True                          |
| vol_managed_momentum_60               | D                |          1 | -0.0180379  |    -0.020871   |    0.000614986 | False                         |
