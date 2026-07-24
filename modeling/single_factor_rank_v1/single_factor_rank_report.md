# 单因子排序模型测试报告

> 本报告由 handoff 矩阵接口自动生成；模型为**单因子横截面排序**，不含多因子加权或回归系数。
> **主结果是每一个因子各自的指标**；不再输出 TopK 短名单。
> 美化图表与有序表见 `outputs/figures/`、`outputs/tables/`。
> 2025 不在本 handoff 包内，不参与因子选择、方向选择或主评价。

## 运行设置

| 项目 | 值 |
| --- | --- |
| 样本外年份 | 2019—2024 |
| handoff目录 | D:\软件2\AI-15-demo-main\--main2\--main - 副本 (3)\training_handoff |
| 评测字段数 | 92 |
| 短名单TopK | 关闭 |
| score_transform | cs_rank_centered |
| 主标签 | future_return_5d |
| 图表目录 | modeling/single_factor_rank_v1/outputs/visuals |
| 有序表目录 | modeling/single_factor_rank_v1/outputs/tables |

## 字段组成

| usage_type | 字段数 |
| --- | --- |
| conditional | 5 |
| predictor | 36 |
| quality_flag | 45 |
| risk | 6 |

## 全部字段：按 RankIC 排序

| rank | feature_name | source_factor | usage_type | category | rank_ic_mean | rank_ic_hit_rate | rank_icir | gross_annual_return | gross_sharpe | net_0bps_sharpe | net_3bps_sharpe | net_5bps_sharpe | net_10bps_sharpe | strategy_days | years_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | volume_shock_20__missing | volume_shock_20 | quality_flag | data_quality | 0.1416 | 0.8000 | 0.6877 | -0.0035 | -0.0630 | -0.0630 | -0.6078 | -0.9710 | -1.8790 | 293 | 6 |
| 2 | correlation_risk_20_120__missing | correlation_risk_20_120 | quality_flag | data_quality | 0.0595 | 0.6333 | 0.3765 | 0.0003 | 0.0059 | 0.0059 | -0.5409 | -0.9055 | -1.8169 | 293 | 6 |
| 3 | amihud_20__missing | amihud_20 | quality_flag | data_quality | 0.0514 | 0.5897 | 0.2379 | -0.0023 | -0.0422 | -0.0422 | -0.5865 | -0.9493 | -1.8565 | 293 | 6 |
| 4 | basis_surprise__missing | basis_surprise | quality_flag | data_quality | 0.0321 | 0.5220 | 0.1212 | 0.0332 | 0.4119 | 0.4119 | 0.0367 | -0.2134 | -0.8388 | 293 | 6 |
| 5 | basis_fresh__missing | basis_fresh | quality_flag | data_quality | 0.0320 | 0.5215 | 0.1205 | 0.0343 | 0.4216 | 0.4216 | 0.0503 | -0.1972 | -0.8160 | 293 | 6 |
| 6 | basis_momentum_20__missing | basis_momentum_20 | quality_flag | data_quality | 0.0318 | 0.5200 | 0.1201 | 0.0345 | 0.4227 | 0.4227 | 0.0517 | -0.1956 | -0.8139 | 293 | 6 |
| 7 | basis_shock_20__missing | basis_shock_20 | quality_flag | data_quality | 0.0311 | 0.5192 | 0.1172 | 0.0356 | 0.4384 | 0.4384 | 0.0665 | -0.1815 | -0.8014 | 293 | 6 |
| 8 | fundamental_composite_news__missing | fundamental_composite_news | quality_flag | data_quality | 0.0298 | 0.5216 | 0.1164 | 0.0297 | 0.3770 | 0.3770 | -0.0073 | -0.2635 | -0.9039 | 293 | 6 |
| 9 | fundamental_underreaction__missing | fundamental_underreaction | quality_flag | data_quality | 0.0298 | 0.5216 | 0.1164 | 0.0297 | 0.3770 | 0.3770 | -0.0073 | -0.2635 | -0.9039 | 293 | 6 |
| 10 | carry_annualized__csz | carry_annualized | predictor | term_structure | 0.0291 | 0.5469 | 0.1111 | 0.0255 | 0.2788 | 0.2788 | -0.0515 | -0.2717 | -0.8222 | 293 | 6 |
| 11 | basis_fresh__csz | basis_fresh | predictor | basis_spot | 0.0285 | 0.5571 | 0.1293 | 0.0370 | 0.4635 | 0.4635 | 0.0845 | -0.1682 | -0.7999 | 293 | 6 |
| 12 | warehouse_tightness__csz | warehouse_tightness | predictor | inventory_fundamental | 0.0230 | 0.5402 | 0.1127 | 0.0538 | 0.8314 | 0.8314 | 0.3645 | 0.0532 | -0.7249 | 293 | 6 |
| 13 | information_divergence__csz | information_divergence | predictor | information_news | 0.0227 | 0.5387 | 0.1060 | 0.0213 | 0.2843 | 0.2843 | -0.1187 | -0.3874 | -1.0591 | 293 | 6 |
| 14 | inventory_price_divergence__csz | inventory_price_divergence | predictor | information_news | 0.0214 | 0.5457 | 0.1156 | 0.0183 | 0.3133 | 0.3133 | -0.2055 | -0.5513 | -1.4160 | 293 | 6 |
| 15 | term_structure_slope__csz | term_structure_slope | predictor | term_structure | 0.0204 | 0.5462 | 0.0770 | 0.0109 | 0.1072 | 0.1072 | -0.1898 | -0.3877 | -0.8826 | 293 | 6 |
| 16 | nonlinear_tightness__csz | nonlinear_tightness | predictor | factor_interaction | 0.0197 | 0.5241 | 0.0922 | 0.0394 | 0.5389 | 0.5389 | 0.1253 | -0.1504 | -0.8397 | 293 | 6 |
| 17 | tightness_composite__csz | tightness_composite | predictor | inventory_fundamental | 0.0194 | 0.5248 | 0.0906 | 0.0383 | 0.5235 | 0.5235 | 0.1100 | -0.1656 | -0.8547 | 293 | 6 |
| 18 | information_consistency__csz | information_consistency | predictor | information_news | 0.0190 | 0.5366 | 0.0884 | 0.0235 | 0.3216 | 0.3216 | -0.0916 | -0.3671 | -1.0557 | 293 | 6 |
| 19 | short_reversal_5__csz | short_reversal_5 | predictor | value_reversion | 0.0182 | 0.5262 | 0.0662 | 0.0469 | 0.4378 | 0.4378 | 0.1554 | -0.0328 | -0.5034 | 293 | 6 |
| 20 | inventory_acceleration__csz | inventory_acceleration | predictor | inventory_fundamental | 0.0178 | 0.5441 | 0.0985 | 0.0129 | 0.1953 | 0.1953 | -0.2626 | -0.5679 | -1.3311 | 293 | 6 |
| 21 | term_structure_acceleration__csz | term_structure_acceleration | predictor | term_structure | 0.0159 | 0.5345 | 0.0698 | 0.0107 | 0.1276 | 0.1276 | -0.2344 | -0.4757 | -1.0790 | 293 | 6 |
| 22 | idiosyncratic_volatility_60__csz | idiosyncratic_volatility_60 | risk | volatility_risk | 0.0151 | 0.5338 | 0.0572 | 0.0192 | 0.1926 | 0.1926 | -0.1102 | -0.3121 | -0.8168 | 293 | 6 |
| 23 | curve_curvature__missing | curve_curvature | quality_flag | data_quality | 0.0145 | 0.5207 | 0.0623 | -0.0007 | -0.0119 | -0.0119 | -0.5616 | -0.9280 | -1.8442 | 293 | 6 |
| 24 | realized_skewness_60__csz | realized_skewness_60 | risk | volatility_risk | 0.0144 | 0.5414 | 0.0639 | 0.0119 | 0.1495 | 0.1495 | -0.2302 | -0.4834 | -1.1163 | 293 | 6 |
| 25 | information_consistency__missing | information_consistency | quality_flag | data_quality | 0.0136 | 0.5206 | 0.0604 | 0.0106 | 0.1416 | 0.1416 | -0.2608 | -0.5291 | -1.1997 | 293 | 6 |
| 26 | information_divergence__missing | information_divergence | quality_flag | data_quality | 0.0136 | 0.5206 | 0.0604 | 0.0106 | 0.1416 | 0.1416 | -0.2608 | -0.5291 | -1.1997 | 293 | 6 |
| 27 | nonlinear_tightness__missing | nonlinear_tightness | quality_flag | data_quality | 0.0136 | 0.5206 | 0.0604 | 0.0106 | 0.1416 | 0.1416 | -0.2608 | -0.5291 | -1.1997 | 293 | 6 |
| 28 | tightness_composite__missing | tightness_composite | quality_flag | data_quality | 0.0136 | 0.5206 | 0.0604 | 0.0106 | 0.1416 | 0.1416 | -0.2608 | -0.5291 | -1.1997 | 293 | 6 |
| 29 | inventory_change_5__csz | inventory_change_5 | predictor | inventory_fundamental | 0.0130 | 0.5277 | 0.0708 | 0.0220 | 0.3767 | 0.3767 | -0.1421 | -0.4880 | -1.3527 | 293 | 6 |
| 30 | term_structure_momentum__csz | term_structure_momentum | predictor | term_structure | 0.0128 | 0.5400 | 0.0538 | 0.0011 | 0.0119 | 0.0119 | -0.3298 | -0.5575 | -1.1269 | 293 | 6 |
| 31 | fundamental_underreaction__csz | fundamental_underreaction | predictor | information_news | 0.0127 | 0.5176 | 0.0555 | -0.0081 | -0.1105 | -0.1105 | -0.5253 | -0.8018 | -1.4931 | 293 | 6 |
| 32 | realized_volatility_20__csz | realized_volatility_20 | risk | volatility_risk | 0.0127 | 0.5124 | 0.0402 | 0.0278 | 0.2149 | 0.2149 | -0.0190 | -0.1749 | -0.5646 | 293 | 6 |
| 33 | value_252__csz | value_252 | predictor | value_reversion | 0.0126 | 0.5076 | 0.0455 | -0.0022 | -0.0218 | -0.0218 | -0.3203 | -0.5193 | -1.0168 | 293 | 6 |
| 34 | amihud_20__csz | amihud_20 | risk | trading_liquidity | 0.0114 | 0.5276 | 0.0532 | 0.0216 | 0.2464 | 0.2464 | -0.0992 | -0.3297 | -0.9058 | 293 | 6 |
| 35 | momentum_carry_interaction__missing | momentum_carry_interaction | quality_flag | data_quality | 0.0113 | 0.5000 | 0.0741 | -0.0063 | -0.1129 | -0.1129 | -0.6564 | -1.0186 | -1.9243 | 293 | 6 |
| 36 | sector_relative_value_60__csz | sector_relative_value_60 | predictor | industry_chain | 0.0109 | 0.5241 | 0.0437 | 0.0298 | 0.3172 | 0.3172 | -0.0049 | -0.2196 | -0.7563 | 293 | 6 |
| 37 | warehouse_surprise__missing | warehouse_surprise | quality_flag | data_quality | 0.0104 | 0.5146 | 0.0517 | -0.0056 | -0.1000 | -0.1000 | -0.6364 | -0.9940 | -1.8879 | 293 | 6 |
| 38 | basis_surprise__csz | basis_surprise | predictor | basis_spot | 0.0074 | 0.5095 | 0.0339 | -0.0039 | -0.0556 | -0.0556 | -0.4885 | -0.7772 | -1.4988 | 293 | 6 |
| 39 | realized_skewness_60__missing | realized_skewness_60 | quality_flag | data_quality | 0.0064 | 0.5167 | 0.0398 | -0.0039 | -0.0707 | -0.0707 | -0.6176 | -0.9821 | -1.8935 | 293 | 6 |
| 40 | trend_efficiency_60__missing | trend_efficiency_60 | quality_flag | data_quality | 0.0064 | 0.5167 | 0.0398 | -0.0039 | -0.0707 | -0.0707 | -0.6176 | -0.9821 | -1.8935 | 293 | 6 |
| 41 | price_oi_interaction_20__csz | price_oi_interaction_20 | predictor | trading_liquidity | 0.0057 | 0.5110 | 0.0252 | -0.0049 | -0.0531 | -0.0531 | -0.3825 | -0.6020 | -1.1510 | 293 | 6 |
| 42 | fundamental_composite_news__csz | fundamental_composite_news | predictor | information_news | 0.0043 | 0.5096 | 0.0217 | -0.0081 | -0.1389 | -0.1389 | -0.6569 | -1.0022 | -1.8654 | 293 | 6 |
| 43 | warehouse_tightness__missing | warehouse_tightness | quality_flag | data_quality | 0.0023 | 0.4928 | 0.0109 | -0.0186 | -0.2895 | -0.2895 | -0.7608 | -1.0750 | -1.8606 | 293 | 6 |
| 44 | warehouse_surprise__csz | warehouse_surprise | predictor | information_news | 0.0021 | 0.4895 | 0.0106 | 0.0051 | 0.0770 | 0.0770 | -0.3784 | -0.6819 | -1.4409 | 293 | 6 |
| 45 | basis_shock_20__csz | basis_shock_20 | predictor | basis_spot | 0.0015 | 0.5058 | 0.0068 | -0.0465 | -0.6223 | -0.6223 | -1.0271 | -1.2970 | -1.9717 | 293 | 6 |
| 46 | turnover_oi__csz | turnover_oi | conditional | trading_liquidity | 0.0005 | 0.5069 | 0.0019 | -0.0354 | -0.4573 | -0.4573 | -0.8482 | -1.1089 | -1.7604 | 293 | 6 |
| 47 | vol_managed_momentum__csz | vol_managed_momentum | predictor | trend_momentum | -0.0006 | 0.5131 | -0.0025 | 0.0020 | 0.0312 | 0.0312 | -0.4303 | -0.7380 | -1.5072 | 293 | 6 |
| 48 | oi_surprise_60__csz | oi_surprise_60 | conditional | trading_liquidity | -0.0006 | 0.5090 | -0.0028 | 0.0270 | 0.3699 | 0.3699 | -0.0449 | -0.3214 | -1.0128 | 293 | 6 |
| 49 | inventory_surprise__csz | inventory_surprise | predictor | information_news | -0.0007 | 0.4869 | -0.0040 | -0.0229 | -0.4380 | -0.4380 | -1.0175 | -1.4039 | -2.3698 | 293 | 6 |
| 50 | curve_curvature__csz | curve_curvature | predictor | term_structure | -0.0027 | 0.4972 | -0.0131 | 0.0123 | 0.1668 | 0.1668 | -0.2418 | -0.5142 | -1.1952 | 293 | 6 |
| 51 | trend_efficiency_60__csz | trend_efficiency_60 | predictor | trend_momentum | -0.0050 | 0.4903 | -0.0205 | -0.0315 | -0.3564 | -0.3564 | -0.6988 | -0.9271 | -1.4978 | 293 | 6 |
| 52 | idiosyncratic_volatility_60__missing | idiosyncratic_volatility_60 | quality_flag | data_quality | -0.0051 | 0.4915 | -0.0314 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 53 | basis_momentum_20__csz | basis_momentum_20 | predictor | basis_spot | -0.0057 | 0.4841 | -0.0239 | -0.0243 | -0.3318 | -0.3318 | -0.7450 | -1.0204 | -1.7090 | 293 | 6 |
| 54 | inventory_scarcity__csz | inventory_scarcity | predictor | inventory_fundamental | -0.0058 | 0.4868 | -0.0328 | -0.0214 | -0.3839 | -0.3839 | -0.9275 | -1.2900 | -2.1961 | 293 | 6 |
| 55 | volume_shock_20__csz | volume_shock_20 | conditional | trading_liquidity | -0.0064 | 0.4876 | -0.0285 | 0.0144 | 0.1908 | 0.1908 | -0.2087 | -0.4750 | -1.1408 | 293 | 6 |
| 56 | csmom_60__missing | csmom_60 | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 57 | momentum_quality_60__missing | momentum_quality_60 | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 58 | sector_relative_value_60__missing | sector_relative_value_60 | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 59 | tsmom_60__missing | tsmom_60 | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 60 | vol_managed_momentum__missing | vol_managed_momentum | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 61 | momentum_carry_interaction__csz | momentum_carry_interaction | predictor | factor_interaction | -0.0065 | 0.4786 | -0.0298 | -0.0190 | -0.2480 | -0.2480 | -0.6427 | -0.9058 | -1.5637 | 293 | 6 |
| 62 | tsmom_60__csz | tsmom_60 | predictor | trend_momentum | -0.0068 | 0.4917 | -0.0254 | -0.0192 | -0.1889 | -0.1889 | -0.4857 | -0.6836 | -1.1782 | 293 | 6 |
| 63 | momentum_quality_60__csz | momentum_quality_60 | predictor | trend_momentum | -0.0070 | 0.4945 | -0.0262 | -0.0127 | -0.1236 | -0.1236 | -0.4176 | -0.6136 | -1.1035 | 293 | 6 |
| 64 | seasonality_3y__missing | seasonality_3y | quality_flag | data_quality | -0.0080 | 0.4823 | -0.0380 | -0.0166 | -0.2703 | -0.2703 | -0.7643 | -1.0936 | -1.9168 | 293 | 6 |
| 65 | volume_price_divergence_20__csz | volume_price_divergence_20 | predictor | trading_liquidity | -0.0086 | 0.4800 | -0.0336 | -0.0271 | -0.2770 | -0.2770 | -0.5862 | -0.7924 | -1.3079 | 293 | 6 |
| 66 | term_structure_momentum__missing | term_structure_momentum | quality_flag | data_quality | -0.0086 | 0.4809 | -0.0362 | 0.0034 | 0.0476 | 0.0476 | -0.3753 | -0.6572 | -1.3621 | 293 | 6 |
| 67 | csmom_60__csz | csmom_60 | predictor | trend_momentum | -0.0087 | 0.4972 | -0.0300 | -0.0289 | -0.2481 | -0.2481 | -0.5074 | -0.6804 | -1.1126 | 293 | 6 |
| 68 | carry_change_20__csz | carry_change_20 | predictor | term_structure | -0.0090 | 0.4766 | -0.0419 | -0.0433 | -0.5067 | -0.5067 | -0.8606 | -1.0965 | -1.6863 | 293 | 6 |
| 69 | multi_horizon_trend__csz | multi_horizon_trend | predictor | trend_momentum | -0.0095 | 0.4952 | -0.0375 | 0.0002 | 0.0029 | 0.0029 | -0.3502 | -0.5856 | -1.1741 | 293 | 6 |
| 70 | oi_growth_20__csz | oi_growth_20 | conditional | trading_liquidity | -0.0105 | 0.4634 | -0.0488 | -0.0412 | -0.4994 | -0.4994 | -0.8663 | -1.1109 | -1.7224 | 293 | 6 |
| 71 | term_structure_acceleration__missing | term_structure_acceleration | quality_flag | data_quality | -0.0124 | 0.4775 | -0.0516 | 0.0054 | 0.0746 | 0.0746 | -0.3435 | -0.6222 | -1.3190 | 293 | 6 |
| 72 | low_volatility_20__csz | low_volatility_20 | risk | volatility_risk | -0.0127 | 0.4876 | -0.0402 | -0.0278 | -0.2149 | -0.2149 | -0.4487 | -0.6046 | -0.9943 | 293 | 6 |
| 73 | inventory_surprise__missing | inventory_surprise | quality_flag | data_quality | -0.0133 | 0.4542 | -0.0559 | -0.0164 | -0.3006 | -0.3006 | -0.8544 | -1.2236 | -2.1466 | 293 | 6 |
| 74 | seasonality_3y__csz | seasonality_3y | predictor | value_reversion | -0.0138 | 0.4862 | -0.0563 | 0.0063 | 0.0730 | 0.0730 | -0.2774 | -0.5109 | -1.0949 | 293 | 6 |
| 75 | term_structure_slope__missing | term_structure_slope | quality_flag | data_quality | -0.0145 | 0.4680 | -0.0623 | -0.0028 | -0.0409 | -0.0409 | -0.4772 | -0.7681 | -1.4954 | 293 | 6 |
| 76 | inventory_scarcity__missing | inventory_scarcity | quality_flag | data_quality | -0.0165 | 0.4626 | -0.0714 | -0.0277 | -0.4823 | -0.4823 | -1.0091 | -1.3603 | -2.2383 | 293 | 6 |
| 77 | inventory_change_5__missing | inventory_change_5 | quality_flag | data_quality | -0.0183 | 0.4571 | -0.0766 | -0.0273 | -0.4247 | -0.4247 | -0.8951 | -1.2088 | -1.9929 | 293 | 6 |
| 78 | inventory_price_divergence__missing | inventory_price_divergence | quality_flag | data_quality | -0.0183 | 0.4571 | -0.0766 | -0.0273 | -0.4247 | -0.4247 | -0.8951 | -1.2088 | -1.9929 | 293 | 6 |
| 79 | inventory_acceleration__missing | inventory_acceleration | quality_flag | data_quality | -0.0187 | 0.4566 | -0.0782 | -0.0277 | -0.4320 | -0.4320 | -0.9030 | -1.2170 | -2.0021 | 293 | 6 |
| 80 | carry_annualized__missing | carry_annualized | quality_flag | data_quality | -0.0205 | 0.4375 | -0.1491 | -0.0030 | -0.0533 | -0.0533 | -0.5979 | -0.9610 | -1.8688 | 293 | 6 |
| 81 | carry_change_20__missing | carry_change_20 | quality_flag | data_quality | -0.0246 | 0.4824 | -0.1274 | -0.0024 | -0.0436 | -0.0436 | -0.5928 | -0.9589 | -1.8742 | 293 | 6 |
| 82 | correlation_risk_20_120__csz | correlation_risk_20_120 | risk | volatility_risk | -0.0309 | 0.4407 | -0.1256 | -0.0414 | -0.4559 | -0.4559 | -0.7893 | -1.0115 | -1.5672 | 293 | 6 |
| 83 | volume_price_divergence_20__missing | volume_price_divergence_20 | quality_flag | data_quality | -0.0516 | 0.4762 | -0.2636 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 84 | oi_surprise_60__missing | oi_surprise_60 | quality_flag | data_quality | -0.0577 | 0.4762 | -0.3099 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 85 | value_252__missing | value_252 | quality_flag | data_quality | -0.0582 | 0.3651 | -0.3696 | -0.0082 | -0.1424 | -0.1424 | -0.6688 | -1.0197 | -1.8969 | 293 | 6 |
| 86 | multi_horizon_trend__missing | multi_horizon_trend | quality_flag | data_quality | -0.0595 | 0.3667 | -0.3765 | -0.0080 | -0.1394 | -0.1394 | -0.6657 | -1.0165 | -1.8937 | 293 | 6 |
| 87 | low_volatility_20__missing | low_volatility_20 | quality_flag | data_quality | -0.0687 | 0.4500 | -0.3730 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 88 | oi_growth_20__missing | oi_growth_20 | quality_flag | data_quality | -0.0687 | 0.4500 | -0.3730 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 89 | price_oi_interaction_20__missing | price_oi_interaction_20 | quality_flag | data_quality | -0.0687 | 0.4500 | -0.3730 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 90 | realized_volatility_20__missing | realized_volatility_20 | quality_flag | data_quality | -0.0815 | 0.3500 | -0.4569 | -0.0034 | -0.0604 | -0.0604 | -0.6055 | -0.9689 | -1.8773 | 293 | 6 |
| 91 | factor_momentum__missing | factor_momentum | quality_flag | data_quality | — | — | — | -0.0040 | -0.0715 | -0.0715 | -0.6162 | -0.9794 | -1.8872 | 293 | 6 |
| 92 | factor_momentum__tsz | factor_momentum | conditional | meta_factor | — | — | — | -0.0040 | -0.0715 | -0.0715 | -0.6162 | -0.9794 | -1.8872 | 293 | 6 |

## 预测型因子（predictor）：按 RankIC 排序

| rank | feature_name | source_factor | usage_type | category | rank_ic_mean | rank_ic_hit_rate | rank_icir | gross_annual_return | gross_sharpe | net_0bps_sharpe | net_3bps_sharpe | net_5bps_sharpe | net_10bps_sharpe | strategy_days | years_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | carry_annualized__csz | carry_annualized | predictor | term_structure | 0.0291 | 0.5469 | 0.1111 | 0.0255 | 0.2788 | 0.2788 | -0.0515 | -0.2717 | -0.8222 | 293 | 6 |
| 11 | basis_fresh__csz | basis_fresh | predictor | basis_spot | 0.0285 | 0.5571 | 0.1293 | 0.0370 | 0.4635 | 0.4635 | 0.0845 | -0.1682 | -0.7999 | 293 | 6 |
| 12 | warehouse_tightness__csz | warehouse_tightness | predictor | inventory_fundamental | 0.0230 | 0.5402 | 0.1127 | 0.0538 | 0.8314 | 0.8314 | 0.3645 | 0.0532 | -0.7249 | 293 | 6 |
| 13 | information_divergence__csz | information_divergence | predictor | information_news | 0.0227 | 0.5387 | 0.1060 | 0.0213 | 0.2843 | 0.2843 | -0.1187 | -0.3874 | -1.0591 | 293 | 6 |
| 14 | inventory_price_divergence__csz | inventory_price_divergence | predictor | information_news | 0.0214 | 0.5457 | 0.1156 | 0.0183 | 0.3133 | 0.3133 | -0.2055 | -0.5513 | -1.4160 | 293 | 6 |
| 15 | term_structure_slope__csz | term_structure_slope | predictor | term_structure | 0.0204 | 0.5462 | 0.0770 | 0.0109 | 0.1072 | 0.1072 | -0.1898 | -0.3877 | -0.8826 | 293 | 6 |
| 16 | nonlinear_tightness__csz | nonlinear_tightness | predictor | factor_interaction | 0.0197 | 0.5241 | 0.0922 | 0.0394 | 0.5389 | 0.5389 | 0.1253 | -0.1504 | -0.8397 | 293 | 6 |
| 17 | tightness_composite__csz | tightness_composite | predictor | inventory_fundamental | 0.0194 | 0.5248 | 0.0906 | 0.0383 | 0.5235 | 0.5235 | 0.1100 | -0.1656 | -0.8547 | 293 | 6 |
| 18 | information_consistency__csz | information_consistency | predictor | information_news | 0.0190 | 0.5366 | 0.0884 | 0.0235 | 0.3216 | 0.3216 | -0.0916 | -0.3671 | -1.0557 | 293 | 6 |
| 19 | short_reversal_5__csz | short_reversal_5 | predictor | value_reversion | 0.0182 | 0.5262 | 0.0662 | 0.0469 | 0.4378 | 0.4378 | 0.1554 | -0.0328 | -0.5034 | 293 | 6 |
| 20 | inventory_acceleration__csz | inventory_acceleration | predictor | inventory_fundamental | 0.0178 | 0.5441 | 0.0985 | 0.0129 | 0.1953 | 0.1953 | -0.2626 | -0.5679 | -1.3311 | 293 | 6 |
| 21 | term_structure_acceleration__csz | term_structure_acceleration | predictor | term_structure | 0.0159 | 0.5345 | 0.0698 | 0.0107 | 0.1276 | 0.1276 | -0.2344 | -0.4757 | -1.0790 | 293 | 6 |
| 29 | inventory_change_5__csz | inventory_change_5 | predictor | inventory_fundamental | 0.0130 | 0.5277 | 0.0708 | 0.0220 | 0.3767 | 0.3767 | -0.1421 | -0.4880 | -1.3527 | 293 | 6 |
| 30 | term_structure_momentum__csz | term_structure_momentum | predictor | term_structure | 0.0128 | 0.5400 | 0.0538 | 0.0011 | 0.0119 | 0.0119 | -0.3298 | -0.5575 | -1.1269 | 293 | 6 |
| 31 | fundamental_underreaction__csz | fundamental_underreaction | predictor | information_news | 0.0127 | 0.5176 | 0.0555 | -0.0081 | -0.1105 | -0.1105 | -0.5253 | -0.8018 | -1.4931 | 293 | 6 |
| 33 | value_252__csz | value_252 | predictor | value_reversion | 0.0126 | 0.5076 | 0.0455 | -0.0022 | -0.0218 | -0.0218 | -0.3203 | -0.5193 | -1.0168 | 293 | 6 |
| 36 | sector_relative_value_60__csz | sector_relative_value_60 | predictor | industry_chain | 0.0109 | 0.5241 | 0.0437 | 0.0298 | 0.3172 | 0.3172 | -0.0049 | -0.2196 | -0.7563 | 293 | 6 |
| 38 | basis_surprise__csz | basis_surprise | predictor | basis_spot | 0.0074 | 0.5095 | 0.0339 | -0.0039 | -0.0556 | -0.0556 | -0.4885 | -0.7772 | -1.4988 | 293 | 6 |
| 41 | price_oi_interaction_20__csz | price_oi_interaction_20 | predictor | trading_liquidity | 0.0057 | 0.5110 | 0.0252 | -0.0049 | -0.0531 | -0.0531 | -0.3825 | -0.6020 | -1.1510 | 293 | 6 |
| 42 | fundamental_composite_news__csz | fundamental_composite_news | predictor | information_news | 0.0043 | 0.5096 | 0.0217 | -0.0081 | -0.1389 | -0.1389 | -0.6569 | -1.0022 | -1.8654 | 293 | 6 |
| 44 | warehouse_surprise__csz | warehouse_surprise | predictor | information_news | 0.0021 | 0.4895 | 0.0106 | 0.0051 | 0.0770 | 0.0770 | -0.3784 | -0.6819 | -1.4409 | 293 | 6 |
| 45 | basis_shock_20__csz | basis_shock_20 | predictor | basis_spot | 0.0015 | 0.5058 | 0.0068 | -0.0465 | -0.6223 | -0.6223 | -1.0271 | -1.2970 | -1.9717 | 293 | 6 |
| 47 | vol_managed_momentum__csz | vol_managed_momentum | predictor | trend_momentum | -0.0006 | 0.5131 | -0.0025 | 0.0020 | 0.0312 | 0.0312 | -0.4303 | -0.7380 | -1.5072 | 293 | 6 |
| 49 | inventory_surprise__csz | inventory_surprise | predictor | information_news | -0.0007 | 0.4869 | -0.0040 | -0.0229 | -0.4380 | -0.4380 | -1.0175 | -1.4039 | -2.3698 | 293 | 6 |
| 50 | curve_curvature__csz | curve_curvature | predictor | term_structure | -0.0027 | 0.4972 | -0.0131 | 0.0123 | 0.1668 | 0.1668 | -0.2418 | -0.5142 | -1.1952 | 293 | 6 |
| 51 | trend_efficiency_60__csz | trend_efficiency_60 | predictor | trend_momentum | -0.0050 | 0.4903 | -0.0205 | -0.0315 | -0.3564 | -0.3564 | -0.6988 | -0.9271 | -1.4978 | 293 | 6 |
| 53 | basis_momentum_20__csz | basis_momentum_20 | predictor | basis_spot | -0.0057 | 0.4841 | -0.0239 | -0.0243 | -0.3318 | -0.3318 | -0.7450 | -1.0204 | -1.7090 | 293 | 6 |
| 54 | inventory_scarcity__csz | inventory_scarcity | predictor | inventory_fundamental | -0.0058 | 0.4868 | -0.0328 | -0.0214 | -0.3839 | -0.3839 | -0.9275 | -1.2900 | -2.1961 | 293 | 6 |
| 61 | momentum_carry_interaction__csz | momentum_carry_interaction | predictor | factor_interaction | -0.0065 | 0.4786 | -0.0298 | -0.0190 | -0.2480 | -0.2480 | -0.6427 | -0.9058 | -1.5637 | 293 | 6 |
| 62 | tsmom_60__csz | tsmom_60 | predictor | trend_momentum | -0.0068 | 0.4917 | -0.0254 | -0.0192 | -0.1889 | -0.1889 | -0.4857 | -0.6836 | -1.1782 | 293 | 6 |
| 63 | momentum_quality_60__csz | momentum_quality_60 | predictor | trend_momentum | -0.0070 | 0.4945 | -0.0262 | -0.0127 | -0.1236 | -0.1236 | -0.4176 | -0.6136 | -1.1035 | 293 | 6 |
| 65 | volume_price_divergence_20__csz | volume_price_divergence_20 | predictor | trading_liquidity | -0.0086 | 0.4800 | -0.0336 | -0.0271 | -0.2770 | -0.2770 | -0.5862 | -0.7924 | -1.3079 | 293 | 6 |
| 67 | csmom_60__csz | csmom_60 | predictor | trend_momentum | -0.0087 | 0.4972 | -0.0300 | -0.0289 | -0.2481 | -0.2481 | -0.5074 | -0.6804 | -1.1126 | 293 | 6 |
| 68 | carry_change_20__csz | carry_change_20 | predictor | term_structure | -0.0090 | 0.4766 | -0.0419 | -0.0433 | -0.5067 | -0.5067 | -0.8606 | -1.0965 | -1.6863 | 293 | 6 |
| 69 | multi_horizon_trend__csz | multi_horizon_trend | predictor | trend_momentum | -0.0095 | 0.4952 | -0.0375 | 0.0002 | 0.0029 | 0.0029 | -0.3502 | -0.5856 | -1.1741 | 293 | 6 |
| 74 | seasonality_3y__csz | seasonality_3y | predictor | value_reversion | -0.0138 | 0.4862 | -0.0563 | 0.0063 | 0.0730 | 0.0730 | -0.2774 | -0.5109 | -1.0949 | 293 | 6 |

## 全部字段：按毛 Sharpe 排序

| rank | feature_name | source_factor | usage_type | category | rank_ic_mean | rank_ic_hit_rate | rank_icir | gross_annual_return | gross_sharpe | net_0bps_sharpe | net_3bps_sharpe | net_5bps_sharpe | net_10bps_sharpe | strategy_days | years_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | warehouse_tightness__csz | warehouse_tightness | predictor | inventory_fundamental | 0.0230 | 0.5402 | 0.1127 | 0.0538 | 0.8314 | 0.8314 | 0.3645 | 0.0532 | -0.7249 | 293 | 6 |
| 16 | nonlinear_tightness__csz | nonlinear_tightness | predictor | factor_interaction | 0.0197 | 0.5241 | 0.0922 | 0.0394 | 0.5389 | 0.5389 | 0.1253 | -0.1504 | -0.8397 | 293 | 6 |
| 17 | tightness_composite__csz | tightness_composite | predictor | inventory_fundamental | 0.0194 | 0.5248 | 0.0906 | 0.0383 | 0.5235 | 0.5235 | 0.1100 | -0.1656 | -0.8547 | 293 | 6 |
| 11 | basis_fresh__csz | basis_fresh | predictor | basis_spot | 0.0285 | 0.5571 | 0.1293 | 0.0370 | 0.4635 | 0.4635 | 0.0845 | -0.1682 | -0.7999 | 293 | 6 |
| 7 | basis_shock_20__missing | basis_shock_20 | quality_flag | data_quality | 0.0311 | 0.5192 | 0.1172 | 0.0356 | 0.4384 | 0.4384 | 0.0665 | -0.1815 | -0.8014 | 293 | 6 |
| 19 | short_reversal_5__csz | short_reversal_5 | predictor | value_reversion | 0.0182 | 0.5262 | 0.0662 | 0.0469 | 0.4378 | 0.4378 | 0.1554 | -0.0328 | -0.5034 | 293 | 6 |
| 6 | basis_momentum_20__missing | basis_momentum_20 | quality_flag | data_quality | 0.0318 | 0.5200 | 0.1201 | 0.0345 | 0.4227 | 0.4227 | 0.0517 | -0.1956 | -0.8139 | 293 | 6 |
| 5 | basis_fresh__missing | basis_fresh | quality_flag | data_quality | 0.0320 | 0.5215 | 0.1205 | 0.0343 | 0.4216 | 0.4216 | 0.0503 | -0.1972 | -0.8160 | 293 | 6 |
| 4 | basis_surprise__missing | basis_surprise | quality_flag | data_quality | 0.0321 | 0.5220 | 0.1212 | 0.0332 | 0.4119 | 0.4119 | 0.0367 | -0.2134 | -0.8388 | 293 | 6 |
| 8 | fundamental_composite_news__missing | fundamental_composite_news | quality_flag | data_quality | 0.0298 | 0.5216 | 0.1164 | 0.0297 | 0.3770 | 0.3770 | -0.0073 | -0.2635 | -0.9039 | 293 | 6 |
| 9 | fundamental_underreaction__missing | fundamental_underreaction | quality_flag | data_quality | 0.0298 | 0.5216 | 0.1164 | 0.0297 | 0.3770 | 0.3770 | -0.0073 | -0.2635 | -0.9039 | 293 | 6 |
| 29 | inventory_change_5__csz | inventory_change_5 | predictor | inventory_fundamental | 0.0130 | 0.5277 | 0.0708 | 0.0220 | 0.3767 | 0.3767 | -0.1421 | -0.4880 | -1.3527 | 293 | 6 |
| 48 | oi_surprise_60__csz | oi_surprise_60 | conditional | trading_liquidity | -0.0006 | 0.5090 | -0.0028 | 0.0270 | 0.3699 | 0.3699 | -0.0449 | -0.3214 | -1.0128 | 293 | 6 |
| 18 | information_consistency__csz | information_consistency | predictor | information_news | 0.0190 | 0.5366 | 0.0884 | 0.0235 | 0.3216 | 0.3216 | -0.0916 | -0.3671 | -1.0557 | 293 | 6 |
| 36 | sector_relative_value_60__csz | sector_relative_value_60 | predictor | industry_chain | 0.0109 | 0.5241 | 0.0437 | 0.0298 | 0.3172 | 0.3172 | -0.0049 | -0.2196 | -0.7563 | 293 | 6 |
| 14 | inventory_price_divergence__csz | inventory_price_divergence | predictor | information_news | 0.0214 | 0.5457 | 0.1156 | 0.0183 | 0.3133 | 0.3133 | -0.2055 | -0.5513 | -1.4160 | 293 | 6 |
| 13 | information_divergence__csz | information_divergence | predictor | information_news | 0.0227 | 0.5387 | 0.1060 | 0.0213 | 0.2843 | 0.2843 | -0.1187 | -0.3874 | -1.0591 | 293 | 6 |
| 10 | carry_annualized__csz | carry_annualized | predictor | term_structure | 0.0291 | 0.5469 | 0.1111 | 0.0255 | 0.2788 | 0.2788 | -0.0515 | -0.2717 | -0.8222 | 293 | 6 |
| 34 | amihud_20__csz | amihud_20 | risk | trading_liquidity | 0.0114 | 0.5276 | 0.0532 | 0.0216 | 0.2464 | 0.2464 | -0.0992 | -0.3297 | -0.9058 | 293 | 6 |
| 32 | realized_volatility_20__csz | realized_volatility_20 | risk | volatility_risk | 0.0127 | 0.5124 | 0.0402 | 0.0278 | 0.2149 | 0.2149 | -0.0190 | -0.1749 | -0.5646 | 293 | 6 |
| 20 | inventory_acceleration__csz | inventory_acceleration | predictor | inventory_fundamental | 0.0178 | 0.5441 | 0.0985 | 0.0129 | 0.1953 | 0.1953 | -0.2626 | -0.5679 | -1.3311 | 293 | 6 |
| 22 | idiosyncratic_volatility_60__csz | idiosyncratic_volatility_60 | risk | volatility_risk | 0.0151 | 0.5338 | 0.0572 | 0.0192 | 0.1926 | 0.1926 | -0.1102 | -0.3121 | -0.8168 | 293 | 6 |
| 55 | volume_shock_20__csz | volume_shock_20 | conditional | trading_liquidity | -0.0064 | 0.4876 | -0.0285 | 0.0144 | 0.1908 | 0.1908 | -0.2087 | -0.4750 | -1.1408 | 293 | 6 |
| 50 | curve_curvature__csz | curve_curvature | predictor | term_structure | -0.0027 | 0.4972 | -0.0131 | 0.0123 | 0.1668 | 0.1668 | -0.2418 | -0.5142 | -1.1952 | 293 | 6 |
| 24 | realized_skewness_60__csz | realized_skewness_60 | risk | volatility_risk | 0.0144 | 0.5414 | 0.0639 | 0.0119 | 0.1495 | 0.1495 | -0.2302 | -0.4834 | -1.1163 | 293 | 6 |
| 25 | information_consistency__missing | information_consistency | quality_flag | data_quality | 0.0136 | 0.5206 | 0.0604 | 0.0106 | 0.1416 | 0.1416 | -0.2608 | -0.5291 | -1.1997 | 293 | 6 |
| 26 | information_divergence__missing | information_divergence | quality_flag | data_quality | 0.0136 | 0.5206 | 0.0604 | 0.0106 | 0.1416 | 0.1416 | -0.2608 | -0.5291 | -1.1997 | 293 | 6 |
| 27 | nonlinear_tightness__missing | nonlinear_tightness | quality_flag | data_quality | 0.0136 | 0.5206 | 0.0604 | 0.0106 | 0.1416 | 0.1416 | -0.2608 | -0.5291 | -1.1997 | 293 | 6 |
| 28 | tightness_composite__missing | tightness_composite | quality_flag | data_quality | 0.0136 | 0.5206 | 0.0604 | 0.0106 | 0.1416 | 0.1416 | -0.2608 | -0.5291 | -1.1997 | 293 | 6 |
| 21 | term_structure_acceleration__csz | term_structure_acceleration | predictor | term_structure | 0.0159 | 0.5345 | 0.0698 | 0.0107 | 0.1276 | 0.1276 | -0.2344 | -0.4757 | -1.0790 | 293 | 6 |
| 15 | term_structure_slope__csz | term_structure_slope | predictor | term_structure | 0.0204 | 0.5462 | 0.0770 | 0.0109 | 0.1072 | 0.1072 | -0.1898 | -0.3877 | -0.8826 | 293 | 6 |
| 44 | warehouse_surprise__csz | warehouse_surprise | predictor | information_news | 0.0021 | 0.4895 | 0.0106 | 0.0051 | 0.0770 | 0.0770 | -0.3784 | -0.6819 | -1.4409 | 293 | 6 |
| 71 | term_structure_acceleration__missing | term_structure_acceleration | quality_flag | data_quality | -0.0124 | 0.4775 | -0.0516 | 0.0054 | 0.0746 | 0.0746 | -0.3435 | -0.6222 | -1.3190 | 293 | 6 |
| 74 | seasonality_3y__csz | seasonality_3y | predictor | value_reversion | -0.0138 | 0.4862 | -0.0563 | 0.0063 | 0.0730 | 0.0730 | -0.2774 | -0.5109 | -1.0949 | 293 | 6 |
| 66 | term_structure_momentum__missing | term_structure_momentum | quality_flag | data_quality | -0.0086 | 0.4809 | -0.0362 | 0.0034 | 0.0476 | 0.0476 | -0.3753 | -0.6572 | -1.3621 | 293 | 6 |
| 47 | vol_managed_momentum__csz | vol_managed_momentum | predictor | trend_momentum | -0.0006 | 0.5131 | -0.0025 | 0.0020 | 0.0312 | 0.0312 | -0.4303 | -0.7380 | -1.5072 | 293 | 6 |
| 30 | term_structure_momentum__csz | term_structure_momentum | predictor | term_structure | 0.0128 | 0.5400 | 0.0538 | 0.0011 | 0.0119 | 0.0119 | -0.3298 | -0.5575 | -1.1269 | 293 | 6 |
| 2 | correlation_risk_20_120__missing | correlation_risk_20_120 | quality_flag | data_quality | 0.0595 | 0.6333 | 0.3765 | 0.0003 | 0.0059 | 0.0059 | -0.5409 | -0.9055 | -1.8169 | 293 | 6 |
| 69 | multi_horizon_trend__csz | multi_horizon_trend | predictor | trend_momentum | -0.0095 | 0.4952 | -0.0375 | 0.0002 | 0.0029 | 0.0029 | -0.3502 | -0.5856 | -1.1741 | 293 | 6 |
| 23 | curve_curvature__missing | curve_curvature | quality_flag | data_quality | 0.0145 | 0.5207 | 0.0623 | -0.0007 | -0.0119 | -0.0119 | -0.5616 | -0.9280 | -1.8442 | 293 | 6 |
| 33 | value_252__csz | value_252 | predictor | value_reversion | 0.0126 | 0.5076 | 0.0455 | -0.0022 | -0.0218 | -0.0218 | -0.3203 | -0.5193 | -1.0168 | 293 | 6 |
| 75 | term_structure_slope__missing | term_structure_slope | quality_flag | data_quality | -0.0145 | 0.4680 | -0.0623 | -0.0028 | -0.0409 | -0.0409 | -0.4772 | -0.7681 | -1.4954 | 293 | 6 |
| 3 | amihud_20__missing | amihud_20 | quality_flag | data_quality | 0.0514 | 0.5897 | 0.2379 | -0.0023 | -0.0422 | -0.0422 | -0.5865 | -0.9493 | -1.8565 | 293 | 6 |
| 81 | carry_change_20__missing | carry_change_20 | quality_flag | data_quality | -0.0246 | 0.4824 | -0.1274 | -0.0024 | -0.0436 | -0.0436 | -0.5928 | -0.9589 | -1.8742 | 293 | 6 |
| 41 | price_oi_interaction_20__csz | price_oi_interaction_20 | predictor | trading_liquidity | 0.0057 | 0.5110 | 0.0252 | -0.0049 | -0.0531 | -0.0531 | -0.3825 | -0.6020 | -1.1510 | 293 | 6 |
| 80 | carry_annualized__missing | carry_annualized | quality_flag | data_quality | -0.0205 | 0.4375 | -0.1491 | -0.0030 | -0.0533 | -0.0533 | -0.5979 | -0.9610 | -1.8688 | 293 | 6 |
| 38 | basis_surprise__csz | basis_surprise | predictor | basis_spot | 0.0074 | 0.5095 | 0.0339 | -0.0039 | -0.0556 | -0.0556 | -0.4885 | -0.7772 | -1.4988 | 293 | 6 |
| 90 | realized_volatility_20__missing | realized_volatility_20 | quality_flag | data_quality | -0.0815 | 0.3500 | -0.4569 | -0.0034 | -0.0604 | -0.0604 | -0.6055 | -0.9689 | -1.8773 | 293 | 6 |
| 1 | volume_shock_20__missing | volume_shock_20 | quality_flag | data_quality | 0.1416 | 0.8000 | 0.6877 | -0.0035 | -0.0630 | -0.0630 | -0.6078 | -0.9710 | -1.8790 | 293 | 6 |
| 83 | volume_price_divergence_20__missing | volume_price_divergence_20 | quality_flag | data_quality | -0.0516 | 0.4762 | -0.2636 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 84 | oi_surprise_60__missing | oi_surprise_60 | quality_flag | data_quality | -0.0577 | 0.4762 | -0.3099 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 87 | low_volatility_20__missing | low_volatility_20 | quality_flag | data_quality | -0.0687 | 0.4500 | -0.3730 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 88 | oi_growth_20__missing | oi_growth_20 | quality_flag | data_quality | -0.0687 | 0.4500 | -0.3730 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 89 | price_oi_interaction_20__missing | price_oi_interaction_20 | quality_flag | data_quality | -0.0687 | 0.4500 | -0.3730 | -0.0037 | -0.0664 | -0.0664 | -0.6116 | -0.9751 | -1.8838 | 293 | 6 |
| 39 | realized_skewness_60__missing | realized_skewness_60 | quality_flag | data_quality | 0.0064 | 0.5167 | 0.0398 | -0.0039 | -0.0707 | -0.0707 | -0.6176 | -0.9821 | -1.8935 | 293 | 6 |
| 40 | trend_efficiency_60__missing | trend_efficiency_60 | quality_flag | data_quality | 0.0064 | 0.5167 | 0.0398 | -0.0039 | -0.0707 | -0.0707 | -0.6176 | -0.9821 | -1.8935 | 293 | 6 |
| 91 | factor_momentum__missing | factor_momentum | quality_flag | data_quality | — | — | — | -0.0040 | -0.0715 | -0.0715 | -0.6162 | -0.9794 | -1.8872 | 293 | 6 |
| 92 | factor_momentum__tsz | factor_momentum | conditional | meta_factor | — | — | — | -0.0040 | -0.0715 | -0.0715 | -0.6162 | -0.9794 | -1.8872 | 293 | 6 |
| 52 | idiosyncratic_volatility_60__missing | idiosyncratic_volatility_60 | quality_flag | data_quality | -0.0051 | 0.4915 | -0.0314 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 56 | csmom_60__missing | csmom_60 | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 57 | momentum_quality_60__missing | momentum_quality_60 | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 58 | sector_relative_value_60__missing | sector_relative_value_60 | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 59 | tsmom_60__missing | tsmom_60 | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 60 | vol_managed_momentum__missing | vol_managed_momentum | quality_flag | data_quality | -0.0064 | 0.4833 | -0.0398 | -0.0045 | -0.0791 | -0.0791 | -0.6161 | -0.9741 | -1.8691 | 293 | 6 |
| 37 | warehouse_surprise__missing | warehouse_surprise | quality_flag | data_quality | 0.0104 | 0.5146 | 0.0517 | -0.0056 | -0.1000 | -0.1000 | -0.6364 | -0.9940 | -1.8879 | 293 | 6 |
| 31 | fundamental_underreaction__csz | fundamental_underreaction | predictor | information_news | 0.0127 | 0.5176 | 0.0555 | -0.0081 | -0.1105 | -0.1105 | -0.5253 | -0.8018 | -1.4931 | 293 | 6 |
| 35 | momentum_carry_interaction__missing | momentum_carry_interaction | quality_flag | data_quality | 0.0113 | 0.5000 | 0.0741 | -0.0063 | -0.1129 | -0.1129 | -0.6564 | -1.0186 | -1.9243 | 293 | 6 |
| 63 | momentum_quality_60__csz | momentum_quality_60 | predictor | trend_momentum | -0.0070 | 0.4945 | -0.0262 | -0.0127 | -0.1236 | -0.1236 | -0.4176 | -0.6136 | -1.1035 | 293 | 6 |
| 42 | fundamental_composite_news__csz | fundamental_composite_news | predictor | information_news | 0.0043 | 0.5096 | 0.0217 | -0.0081 | -0.1389 | -0.1389 | -0.6569 | -1.0022 | -1.8654 | 293 | 6 |
| 86 | multi_horizon_trend__missing | multi_horizon_trend | quality_flag | data_quality | -0.0595 | 0.3667 | -0.3765 | -0.0080 | -0.1394 | -0.1394 | -0.6657 | -1.0165 | -1.8937 | 293 | 6 |
| 85 | value_252__missing | value_252 | quality_flag | data_quality | -0.0582 | 0.3651 | -0.3696 | -0.0082 | -0.1424 | -0.1424 | -0.6688 | -1.0197 | -1.8969 | 293 | 6 |
| 62 | tsmom_60__csz | tsmom_60 | predictor | trend_momentum | -0.0068 | 0.4917 | -0.0254 | -0.0192 | -0.1889 | -0.1889 | -0.4857 | -0.6836 | -1.1782 | 293 | 6 |
| 72 | low_volatility_20__csz | low_volatility_20 | risk | volatility_risk | -0.0127 | 0.4876 | -0.0402 | -0.0278 | -0.2149 | -0.2149 | -0.4487 | -0.6046 | -0.9943 | 293 | 6 |
| 61 | momentum_carry_interaction__csz | momentum_carry_interaction | predictor | factor_interaction | -0.0065 | 0.4786 | -0.0298 | -0.0190 | -0.2480 | -0.2480 | -0.6427 | -0.9058 | -1.5637 | 293 | 6 |
| 67 | csmom_60__csz | csmom_60 | predictor | trend_momentum | -0.0087 | 0.4972 | -0.0300 | -0.0289 | -0.2481 | -0.2481 | -0.5074 | -0.6804 | -1.1126 | 293 | 6 |
| 64 | seasonality_3y__missing | seasonality_3y | quality_flag | data_quality | -0.0080 | 0.4823 | -0.0380 | -0.0166 | -0.2703 | -0.2703 | -0.7643 | -1.0936 | -1.9168 | 293 | 6 |
| 65 | volume_price_divergence_20__csz | volume_price_divergence_20 | predictor | trading_liquidity | -0.0086 | 0.4800 | -0.0336 | -0.0271 | -0.2770 | -0.2770 | -0.5862 | -0.7924 | -1.3079 | 293 | 6 |
| 43 | warehouse_tightness__missing | warehouse_tightness | quality_flag | data_quality | 0.0023 | 0.4928 | 0.0109 | -0.0186 | -0.2895 | -0.2895 | -0.7608 | -1.0750 | -1.8606 | 293 | 6 |
| 73 | inventory_surprise__missing | inventory_surprise | quality_flag | data_quality | -0.0133 | 0.4542 | -0.0559 | -0.0164 | -0.3006 | -0.3006 | -0.8544 | -1.2236 | -2.1466 | 293 | 6 |
| 53 | basis_momentum_20__csz | basis_momentum_20 | predictor | basis_spot | -0.0057 | 0.4841 | -0.0239 | -0.0243 | -0.3318 | -0.3318 | -0.7450 | -1.0204 | -1.7090 | 293 | 6 |
| 51 | trend_efficiency_60__csz | trend_efficiency_60 | predictor | trend_momentum | -0.0050 | 0.4903 | -0.0205 | -0.0315 | -0.3564 | -0.3564 | -0.6988 | -0.9271 | -1.4978 | 293 | 6 |
| 54 | inventory_scarcity__csz | inventory_scarcity | predictor | inventory_fundamental | -0.0058 | 0.4868 | -0.0328 | -0.0214 | -0.3839 | -0.3839 | -0.9275 | -1.2900 | -2.1961 | 293 | 6 |
| 77 | inventory_change_5__missing | inventory_change_5 | quality_flag | data_quality | -0.0183 | 0.4571 | -0.0766 | -0.0273 | -0.4247 | -0.4247 | -0.8951 | -1.2088 | -1.9929 | 293 | 6 |
| 78 | inventory_price_divergence__missing | inventory_price_divergence | quality_flag | data_quality | -0.0183 | 0.4571 | -0.0766 | -0.0273 | -0.4247 | -0.4247 | -0.8951 | -1.2088 | -1.9929 | 293 | 6 |
| 79 | inventory_acceleration__missing | inventory_acceleration | quality_flag | data_quality | -0.0187 | 0.4566 | -0.0782 | -0.0277 | -0.4320 | -0.4320 | -0.9030 | -1.2170 | -2.0021 | 293 | 6 |
| 49 | inventory_surprise__csz | inventory_surprise | predictor | information_news | -0.0007 | 0.4869 | -0.0040 | -0.0229 | -0.4380 | -0.4380 | -1.0175 | -1.4039 | -2.3698 | 293 | 6 |
| 82 | correlation_risk_20_120__csz | correlation_risk_20_120 | risk | volatility_risk | -0.0309 | 0.4407 | -0.1256 | -0.0414 | -0.4559 | -0.4559 | -0.7893 | -1.0115 | -1.5672 | 293 | 6 |
| 46 | turnover_oi__csz | turnover_oi | conditional | trading_liquidity | 0.0005 | 0.5069 | 0.0019 | -0.0354 | -0.4573 | -0.4573 | -0.8482 | -1.1089 | -1.7604 | 293 | 6 |
| 76 | inventory_scarcity__missing | inventory_scarcity | quality_flag | data_quality | -0.0165 | 0.4626 | -0.0714 | -0.0277 | -0.4823 | -0.4823 | -1.0091 | -1.3603 | -2.2383 | 293 | 6 |
| 70 | oi_growth_20__csz | oi_growth_20 | conditional | trading_liquidity | -0.0105 | 0.4634 | -0.0488 | -0.0412 | -0.4994 | -0.4994 | -0.8663 | -1.1109 | -1.7224 | 293 | 6 |
| 68 | carry_change_20__csz | carry_change_20 | predictor | term_structure | -0.0090 | 0.4766 | -0.0419 | -0.0433 | -0.5067 | -0.5067 | -0.8606 | -1.0965 | -1.6863 | 293 | 6 |
| 45 | basis_shock_20__csz | basis_shock_20 | predictor | basis_spot | 0.0015 | 0.5058 | 0.0068 | -0.0465 | -0.6223 | -0.6223 | -1.0271 | -1.2970 | -1.9717 | 293 | 6 |

## 完整策略绩效宽表

下表对**每一个**候选因子单独给出样本外指标。CSV 见 `outputs/factor_strategy_metrics.csv` 与 `outputs/tables/`。

| rank | feature_name | source_factor | usage_type | rank_ic_mean | rank_ic_hit_rate | gross_annual_return | gross_annual_volatility | gross_sharpe | gross_max_drawdown | gross_total_return | gross_hit_rate | net_0bps_annual_return | net_0bps_sharpe | net_3bps_annual_return | net_3bps_sharpe | net_5bps_annual_return | net_5bps_sharpe | net_10bps_annual_return | net_10bps_sharpe | strategy_days | years_evaluated | years_gross_sharpe_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | volume_shock_20__missing | volume_shock_20 | quality_flag | 0.1416 | 0.8000 | -0.0035 | 0.0555 | -0.0630 | -0.1373 | -0.0288 | 0.4983 | -0.0035 | -0.0630 | -0.0337 | -0.6078 | -0.0539 | -0.9710 | -0.1043 | -1.8790 | 293 | 6 | 1 |
| 2 | correlation_risk_20_120__missing | correlation_risk_20_120 | quality_flag | 0.0595 | 0.6333 | 0.0003 | 0.0553 | 0.0059 | -0.1179 | -0.0069 | 0.4949 | 0.0003 | 0.0059 | -0.0299 | -0.5409 | -0.0501 | -0.9055 | -0.1005 | -1.8169 | 293 | 6 | 1 |
| 3 | amihud_20__missing | amihud_20 | quality_flag | 0.0514 | 0.5897 | -0.0023 | 0.0556 | -0.0422 | -0.1337 | -0.0223 | 0.5017 | -0.0023 | -0.0422 | -0.0326 | -0.5865 | -0.0527 | -0.9493 | -0.1031 | -1.8565 | 293 | 6 | 1 |
| 4 | basis_surprise__missing | basis_surprise | quality_flag | 0.0321 | 0.5220 | 0.0332 | 0.0806 | 0.4119 | -0.1006 | 0.1903 | 0.5154 | 0.0332 | 0.4119 | 0.0030 | 0.0367 | -0.0172 | -0.2134 | -0.0676 | -0.8388 | 293 | 6 | 4 |
| 5 | basis_fresh__missing | basis_fresh | quality_flag | 0.0320 | 0.5215 | 0.0343 | 0.0814 | 0.4216 | -0.0942 | 0.1977 | 0.5154 | 0.0343 | 0.4216 | 0.0041 | 0.0503 | -0.0161 | -0.1972 | -0.0665 | -0.8160 | 293 | 6 | 4 |
| 6 | basis_momentum_20__missing | basis_momentum_20 | quality_flag | 0.0318 | 0.5200 | 0.0345 | 0.0815 | 0.4227 | -0.0936 | 0.1985 | 0.5154 | 0.0345 | 0.4227 | 0.0042 | 0.0517 | -0.0159 | -0.1956 | -0.0663 | -0.8139 | 293 | 6 | 4 |
| 7 | basis_shock_20__missing | basis_shock_20 | quality_flag | 0.0311 | 0.5192 | 0.0356 | 0.0813 | 0.4384 | -0.0947 | 0.2069 | 0.5154 | 0.0356 | 0.4384 | 0.0054 | 0.0665 | -0.0148 | -0.1815 | -0.0652 | -0.8014 | 293 | 6 | 4 |
| 8 | fundamental_composite_news__missing | fundamental_composite_news | quality_flag | 0.0298 | 0.5216 | 0.0297 | 0.0787 | 0.3770 | -0.0956 | 0.1671 | 0.5154 | 0.0297 | 0.3770 | -0.0006 | -0.0073 | -0.0207 | -0.2635 | -0.0711 | -0.9039 | 293 | 6 | 4 |
| 9 | fundamental_underreaction__missing | fundamental_underreaction | quality_flag | 0.0298 | 0.5216 | 0.0297 | 0.0787 | 0.3770 | -0.0956 | 0.1671 | 0.5154 | 0.0297 | 0.3770 | -0.0006 | -0.0073 | -0.0207 | -0.2635 | -0.0711 | -0.9039 | 293 | 6 | 4 |
| 10 | carry_annualized__csz | carry_annualized | predictor | 0.0291 | 0.5469 | 0.0255 | 0.0916 | 0.2788 | -0.2093 | 0.1320 | 0.5119 | 0.0255 | 0.2788 | -0.0047 | -0.0515 | -0.0249 | -0.2717 | -0.0753 | -0.8222 | 293 | 6 | 4 |
| 11 | basis_fresh__csz | basis_fresh | predictor | 0.0285 | 0.5571 | 0.0370 | 0.0798 | 0.4635 | -0.1487 | 0.2170 | 0.5256 | 0.0370 | 0.4635 | 0.0067 | 0.0845 | -0.0134 | -0.1682 | -0.0638 | -0.7999 | 293 | 6 | 4 |
| 12 | warehouse_tightness__csz | warehouse_tightness | predictor | 0.0230 | 0.5402 | 0.0538 | 0.0648 | 0.8314 | -0.0566 | 0.3509 | 0.5495 | 0.0538 | 0.8314 | 0.0236 | 0.3645 | 0.0034 | 0.0532 | -0.0470 | -0.7249 | 293 | 6 | 5 |
| 13 | information_divergence__csz | information_divergence | predictor | 0.0227 | 0.5387 | 0.0213 | 0.0750 | 0.2843 | -0.1249 | 0.1137 | 0.5154 | 0.0213 | 0.2843 | -0.0089 | -0.1187 | -0.0291 | -0.3874 | -0.0795 | -1.0591 | 293 | 6 | 3 |
| 14 | inventory_price_divergence__csz | inventory_price_divergence | predictor | 0.0214 | 0.5457 | 0.0183 | 0.0583 | 0.3133 | -0.1048 | 0.1011 | 0.5427 | 0.0183 | 0.3133 | -0.0120 | -0.2055 | -0.0321 | -0.5513 | -0.0825 | -1.4160 | 293 | 6 | 3 |
| 15 | term_structure_slope__csz | term_structure_slope | predictor | 0.0204 | 0.5462 | 0.0109 | 0.1018 | 0.1072 | -0.2321 | 0.0337 | 0.5051 | 0.0109 | 0.1072 | -0.0193 | -0.1898 | -0.0395 | -0.3877 | -0.0899 | -0.8826 | 293 | 6 | 3 |
| 16 | nonlinear_tightness__csz | nonlinear_tightness | predictor | 0.0197 | 0.5241 | 0.0394 | 0.0731 | 0.5389 | -0.1108 | 0.2380 | 0.5290 | 0.0394 | 0.5389 | 0.0092 | 0.1253 | -0.0110 | -0.1504 | -0.0614 | -0.8397 | 293 | 6 | 4 |
| 17 | tightness_composite__csz | tightness_composite | predictor | 0.0194 | 0.5248 | 0.0383 | 0.0731 | 0.5235 | -0.1108 | 0.2300 | 0.5256 | 0.0383 | 0.5235 | 0.0080 | 0.1100 | -0.0121 | -0.1656 | -0.0625 | -0.8547 | 293 | 6 | 4 |
| 18 | information_consistency__csz | information_consistency | predictor | 0.0190 | 0.5366 | 0.0235 | 0.0732 | 0.3216 | -0.1029 | 0.1289 | 0.5495 | 0.0235 | 0.3216 | -0.0067 | -0.0916 | -0.0269 | -0.3671 | -0.0773 | -1.0557 | 293 | 6 | 5 |
| 19 | short_reversal_5__csz | short_reversal_5 | predictor | 0.0182 | 0.5262 | 0.0469 | 0.1071 | 0.4378 | -0.2026 | 0.2707 | 0.5427 | 0.0469 | 0.4378 | 0.0166 | 0.1554 | -0.0035 | -0.0328 | -0.0539 | -0.5034 | 293 | 6 | 4 |
| 20 | inventory_acceleration__csz | inventory_acceleration | predictor | 0.0178 | 0.5441 | 0.0129 | 0.0660 | 0.1953 | -0.1502 | 0.0643 | 0.5495 | 0.0129 | 0.1953 | -0.0173 | -0.2626 | -0.0375 | -0.5679 | -0.0879 | -1.3311 | 293 | 6 | 4 |
| 21 | term_structure_acceleration__csz | term_structure_acceleration | predictor | 0.0159 | 0.5345 | 0.0107 | 0.0835 | 0.1276 | -0.1881 | 0.0426 | 0.5051 | 0.0107 | 0.1276 | -0.0196 | -0.2344 | -0.0397 | -0.4757 | -0.0901 | -1.0790 | 293 | 6 | 4 |
| 22 | idiosyncratic_volatility_60__csz | idiosyncratic_volatility_60 | risk | 0.0151 | 0.5338 | 0.0192 | 0.0999 | 0.1926 | -0.1948 | 0.0864 | 0.5085 | 0.0192 | 0.1926 | -0.0110 | -0.1102 | -0.0312 | -0.3121 | -0.0816 | -0.8168 | 293 | 6 | 3 |
| 23 | curve_curvature__missing | curve_curvature | quality_flag | 0.0145 | 0.5207 | -0.0007 | 0.0550 | -0.0119 | -0.1486 | -0.0125 | 0.4778 | -0.0007 | -0.0119 | -0.0309 | -0.5616 | -0.0511 | -0.9280 | -0.1015 | -1.8442 | 293 | 6 | 2 |
| 24 | realized_skewness_60__csz | realized_skewness_60 | risk | 0.0144 | 0.5414 | 0.0119 | 0.0796 | 0.1495 | -0.1485 | 0.0521 | 0.5051 | 0.0119 | 0.1495 | -0.0183 | -0.2302 | -0.0385 | -0.4834 | -0.0889 | -1.1163 | 293 | 6 | 2 |
| 25 | information_consistency__missing | information_consistency | quality_flag | 0.0136 | 0.5206 | 0.0106 | 0.0752 | 0.1416 | -0.1142 | 0.0466 | 0.5188 | 0.0106 | 0.1416 | -0.0196 | -0.2608 | -0.0398 | -0.5291 | -0.0902 | -1.1997 | 293 | 6 | 4 |
| 26 | information_divergence__missing | information_divergence | quality_flag | 0.0136 | 0.5206 | 0.0106 | 0.0752 | 0.1416 | -0.1142 | 0.0466 | 0.5188 | 0.0106 | 0.1416 | -0.0196 | -0.2608 | -0.0398 | -0.5291 | -0.0902 | -1.1997 | 293 | 6 | 4 |
| 27 | nonlinear_tightness__missing | nonlinear_tightness | quality_flag | 0.0136 | 0.5206 | 0.0106 | 0.0752 | 0.1416 | -0.1142 | 0.0466 | 0.5188 | 0.0106 | 0.1416 | -0.0196 | -0.2608 | -0.0398 | -0.5291 | -0.0902 | -1.1997 | 293 | 6 | 4 |
| 28 | tightness_composite__missing | tightness_composite | quality_flag | 0.0136 | 0.5206 | 0.0106 | 0.0752 | 0.1416 | -0.1142 | 0.0466 | 0.5188 | 0.0106 | 0.1416 | -0.0196 | -0.2608 | -0.0398 | -0.5291 | -0.0902 | -1.1997 | 293 | 6 | 4 |
| 29 | inventory_change_5__csz | inventory_change_5 | predictor | 0.0130 | 0.5277 | 0.0220 | 0.0583 | 0.3767 | -0.1010 | 0.1250 | 0.5461 | 0.0220 | 0.3767 | -0.0083 | -0.1421 | -0.0284 | -0.4880 | -0.0788 | -1.3527 | 293 | 6 | 4 |
| 30 | term_structure_momentum__csz | term_structure_momentum | predictor | 0.0128 | 0.5400 | 0.0011 | 0.0885 | 0.0119 | -0.2165 | -0.0165 | 0.5085 | 0.0011 | 0.0119 | -0.0292 | -0.3298 | -0.0493 | -0.5575 | -0.0997 | -1.1269 | 293 | 6 | 3 |
| 31 | fundamental_underreaction__csz | fundamental_underreaction | predictor | 0.0127 | 0.5176 | -0.0081 | 0.0729 | -0.1105 | -0.1560 | -0.0604 | 0.4915 | -0.0081 | -0.1105 | -0.0383 | -0.5253 | -0.0585 | -0.8018 | -0.1089 | -1.4931 | 293 | 6 | 2 |
| 32 | realized_volatility_20__csz | realized_volatility_20 | risk | 0.0127 | 0.5124 | 0.0278 | 0.1293 | 0.2149 | -0.1936 | 0.1197 | 0.4915 | 0.0278 | 0.2149 | -0.0025 | -0.0190 | -0.0226 | -0.1749 | -0.0730 | -0.5646 | 293 | 6 | 5 |
| 33 | value_252__csz | value_252 | predictor | 0.0126 | 0.5076 | -0.0022 | 0.1013 | -0.0218 | -0.1471 | -0.0415 | 0.4642 | -0.0022 | -0.0218 | -0.0324 | -0.3203 | -0.0526 | -0.5193 | -0.1030 | -1.0168 | 293 | 6 | 2 |
| 34 | amihud_20__csz | amihud_20 | risk | 0.0114 | 0.5276 | 0.0216 | 0.0875 | 0.2464 | -0.2440 | 0.1086 | 0.5461 | 0.0216 | 0.2464 | -0.0087 | -0.0992 | -0.0288 | -0.3297 | -0.0792 | -0.9058 | 293 | 6 | 3 |
| 35 | momentum_carry_interaction__missing | momentum_carry_interaction | quality_flag | 0.0113 | 0.5000 | -0.0063 | 0.0556 | -0.1129 | -0.1513 | -0.0445 | 0.4949 | -0.0063 | -0.1129 | -0.0365 | -0.6564 | -0.0567 | -1.0186 | -0.1071 | -1.9243 | 293 | 6 | 1 |
| 36 | sector_relative_value_60__csz | sector_relative_value_60 | predictor | 0.0109 | 0.5241 | 0.0298 | 0.0939 | 0.3172 | -0.1203 | 0.1588 | 0.5427 | 0.0298 | 0.3172 | -0.0005 | -0.0049 | -0.0206 | -0.2196 | -0.0710 | -0.7563 | 293 | 6 | 4 |
| 37 | warehouse_surprise__missing | warehouse_surprise | quality_flag | 0.0104 | 0.5146 | -0.0056 | 0.0564 | -0.1000 | -0.1292 | -0.0411 | 0.4881 | -0.0056 | -0.1000 | -0.0359 | -0.6364 | -0.0560 | -0.9940 | -0.1064 | -1.8879 | 293 | 6 | 2 |
| 38 | basis_surprise__csz | basis_surprise | predictor | 0.0074 | 0.5095 | -0.0039 | 0.0698 | -0.0556 | -0.1883 | -0.0360 | 0.4676 | -0.0039 | -0.0556 | -0.0341 | -0.4885 | -0.0543 | -0.7772 | -0.1047 | -1.4988 | 293 | 6 | 2 |
| 39 | realized_skewness_60__missing | realized_skewness_60 | quality_flag | 0.0064 | 0.5167 | -0.0039 | 0.0553 | -0.0707 | -0.1394 | -0.0311 | 0.4949 | -0.0039 | -0.0707 | -0.0342 | -0.6176 | -0.0543 | -0.9821 | -0.1047 | -1.8935 | 293 | 6 | 1 |
| 40 | trend_efficiency_60__missing | trend_efficiency_60 | quality_flag | 0.0064 | 0.5167 | -0.0039 | 0.0553 | -0.0707 | -0.1394 | -0.0311 | 0.4949 | -0.0039 | -0.0707 | -0.0342 | -0.6176 | -0.0543 | -0.9821 | -0.1047 | -1.8935 | 293 | 6 | 1 |
| 41 | price_oi_interaction_20__csz | price_oi_interaction_20 | predictor | 0.0057 | 0.5110 | -0.0049 | 0.0918 | -0.0531 | -0.1768 | -0.0516 | 0.5188 | -0.0049 | -0.0531 | -0.0351 | -0.3825 | -0.0553 | -0.6020 | -0.1057 | -1.1510 | 293 | 6 | 3 |
| 42 | fundamental_composite_news__csz | fundamental_composite_news | predictor | 0.0043 | 0.5096 | -0.0081 | 0.0584 | -0.1389 | -0.1123 | -0.0555 | 0.5051 | -0.0081 | -0.1389 | -0.0384 | -0.6569 | -0.0585 | -1.0022 | -0.1089 | -1.8654 | 293 | 6 | 2 |
| 43 | warehouse_tightness__missing | warehouse_tightness | quality_flag | 0.0023 | 0.4928 | -0.0186 | 0.0642 | -0.2895 | -0.1563 | -0.1130 | 0.4949 | -0.0186 | -0.2895 | -0.0488 | -0.7608 | -0.0690 | -1.0750 | -0.1194 | -1.8606 | 293 | 6 | 0 |
| 44 | warehouse_surprise__csz | warehouse_surprise | predictor | 0.0021 | 0.4895 | 0.0051 | 0.0664 | 0.0770 | -0.0967 | 0.0171 | 0.4949 | 0.0051 | 0.0770 | -0.0251 | -0.3784 | -0.0453 | -0.6819 | -0.0957 | -1.4409 | 293 | 6 | 3 |
| 45 | basis_shock_20__csz | basis_shock_20 | predictor | 0.0015 | 0.5058 | -0.0465 | 0.0747 | -0.6223 | -0.3481 | -0.2492 | 0.4812 | -0.0465 | -0.6223 | -0.0767 | -1.0271 | -0.0969 | -1.2970 | -0.1473 | -1.9717 | 293 | 6 | 1 |
| 46 | turnover_oi__csz | turnover_oi | conditional | 0.0005 | 0.5069 | -0.0354 | 0.0774 | -0.4573 | -0.2644 | -0.2000 | 0.4949 | -0.0354 | -0.4573 | -0.0656 | -0.8482 | -0.0858 | -1.1089 | -0.1362 | -1.7604 | 293 | 6 | 1 |
| 47 | vol_managed_momentum__csz | vol_managed_momentum | predictor | -0.0006 | 0.5131 | 0.0020 | 0.0655 | 0.0312 | -0.1161 | -0.0005 | 0.4983 | 0.0020 | 0.0312 | -0.0282 | -0.4303 | -0.0484 | -0.7380 | -0.0988 | -1.5072 | 293 | 6 | 3 |
| 48 | oi_surprise_60__csz | oi_surprise_60 | conditional | -0.0006 | 0.5090 | 0.0270 | 0.0729 | 0.3699 | -0.1048 | 0.1517 | 0.5666 | 0.0270 | 0.3699 | -0.0033 | -0.0449 | -0.0234 | -0.3214 | -0.0738 | -1.0128 | 293 | 6 | 4 |
| 49 | inventory_surprise__csz | inventory_surprise | predictor | -0.0007 | 0.4869 | -0.0229 | 0.0522 | -0.4380 | -0.1541 | -0.1313 | 0.4608 | -0.0229 | -0.4380 | -0.0531 | -1.0175 | -0.0733 | -1.4039 | -0.1237 | -2.3698 | 293 | 6 | 0 |
| 50 | curve_curvature__csz | curve_curvature | predictor | -0.0027 | 0.4972 | 0.0123 | 0.0740 | 0.1668 | -0.1928 | 0.0575 | 0.5358 | 0.0123 | 0.1668 | -0.0179 | -0.2418 | -0.0381 | -0.5142 | -0.0885 | -1.1952 | 293 | 6 | 3 |
| 51 | trend_efficiency_60__csz | trend_efficiency_60 | predictor | -0.0050 | 0.4903 | -0.0315 | 0.0883 | -0.3564 | -0.1796 | -0.1859 | 0.4778 | -0.0315 | -0.3564 | -0.0617 | -0.6988 | -0.0819 | -0.9271 | -0.1323 | -1.4978 | 293 | 6 | 3 |
| 52 | idiosyncratic_volatility_60__missing | idiosyncratic_volatility_60 | quality_flag | -0.0051 | 0.4915 | -0.0045 | 0.0563 | -0.0791 | -0.1424 | -0.0345 | 0.4949 | -0.0045 | -0.0791 | -0.0347 | -0.6161 | -0.0549 | -0.9741 | -0.1053 | -1.8691 | 293 | 6 | 1 |
| 53 | basis_momentum_20__csz | basis_momentum_20 | predictor | -0.0057 | 0.4841 | -0.0243 | 0.0732 | -0.3318 | -0.2210 | -0.1451 | 0.4949 | -0.0243 | -0.3318 | -0.0545 | -0.7450 | -0.0747 | -1.0204 | -0.1251 | -1.7090 | 293 | 6 | 1 |
| 54 | inventory_scarcity__csz | inventory_scarcity | predictor | -0.0058 | 0.4868 | -0.0214 | 0.0556 | -0.3839 | -0.2128 | -0.1247 | 0.4608 | -0.0214 | -0.3839 | -0.0516 | -0.9275 | -0.0718 | -1.2900 | -0.1222 | -2.1961 | 293 | 6 | 2 |
| 55 | volume_shock_20__csz | volume_shock_20 | conditional | -0.0064 | 0.4876 | 0.0144 | 0.0757 | 0.1908 | -0.1974 | 0.0697 | 0.4744 | 0.0144 | 0.1908 | -0.0158 | -0.2087 | -0.0360 | -0.4750 | -0.0864 | -1.1408 | 293 | 6 | 3 |
| 56 | csmom_60__missing | csmom_60 | quality_flag | -0.0064 | 0.4833 | -0.0045 | 0.0563 | -0.0791 | -0.1424 | -0.0345 | 0.4949 | -0.0045 | -0.0791 | -0.0347 | -0.6161 | -0.0549 | -0.9741 | -0.1053 | -1.8691 | 293 | 6 | 1 |
| 57 | momentum_quality_60__missing | momentum_quality_60 | quality_flag | -0.0064 | 0.4833 | -0.0045 | 0.0563 | -0.0791 | -0.1424 | -0.0345 | 0.4949 | -0.0045 | -0.0791 | -0.0347 | -0.6161 | -0.0549 | -0.9741 | -0.1053 | -1.8691 | 293 | 6 | 1 |
| 58 | sector_relative_value_60__missing | sector_relative_value_60 | quality_flag | -0.0064 | 0.4833 | -0.0045 | 0.0563 | -0.0791 | -0.1424 | -0.0345 | 0.4949 | -0.0045 | -0.0791 | -0.0347 | -0.6161 | -0.0549 | -0.9741 | -0.1053 | -1.8691 | 293 | 6 | 1 |
| 59 | tsmom_60__missing | tsmom_60 | quality_flag | -0.0064 | 0.4833 | -0.0045 | 0.0563 | -0.0791 | -0.1424 | -0.0345 | 0.4949 | -0.0045 | -0.0791 | -0.0347 | -0.6161 | -0.0549 | -0.9741 | -0.1053 | -1.8691 | 293 | 6 | 1 |
| 60 | vol_managed_momentum__missing | vol_managed_momentum | quality_flag | -0.0064 | 0.4833 | -0.0045 | 0.0563 | -0.0791 | -0.1424 | -0.0345 | 0.4949 | -0.0045 | -0.0791 | -0.0347 | -0.6161 | -0.0549 | -0.9741 | -0.1053 | -1.8691 | 293 | 6 | 1 |
| 61 | momentum_carry_interaction__csz | momentum_carry_interaction | predictor | -0.0065 | 0.4786 | -0.0190 | 0.0766 | -0.2480 | -0.1448 | -0.1197 | 0.4983 | -0.0190 | -0.2480 | -0.0492 | -0.6427 | -0.0694 | -0.9058 | -0.1198 | -1.5637 | 293 | 6 | 3 |
| 62 | tsmom_60__csz | tsmom_60 | predictor | -0.0068 | 0.4917 | -0.0192 | 0.1019 | -0.1889 | -0.2150 | -0.1326 | 0.5017 | -0.0192 | -0.1889 | -0.0495 | -0.4857 | -0.0696 | -0.6836 | -0.1200 | -1.1782 | 293 | 6 | 3 |
| 63 | momentum_quality_60__csz | momentum_quality_60 | predictor | -0.0070 | 0.4945 | -0.0127 | 0.1029 | -0.1236 | -0.2047 | -0.0995 | 0.4949 | -0.0127 | -0.1236 | -0.0430 | -0.4176 | -0.0631 | -0.6136 | -0.1135 | -1.1035 | 293 | 6 | 3 |
| 64 | seasonality_3y__missing | seasonality_3y | quality_flag | -0.0080 | 0.4823 | -0.0166 | 0.0612 | -0.2703 | -0.1821 | -0.1016 | 0.4812 | -0.0166 | -0.2703 | -0.0468 | -0.7643 | -0.0670 | -1.0936 | -0.1174 | -1.9168 | 293 | 6 | 2 |
| 65 | volume_price_divergence_20__csz | volume_price_divergence_20 | predictor | -0.0086 | 0.4800 | -0.0271 | 0.0978 | -0.2770 | -0.3003 | -0.1692 | 0.5051 | -0.0271 | -0.2770 | -0.0573 | -0.5862 | -0.0775 | -0.7924 | -0.1279 | -1.3079 | 293 | 6 | 3 |
| 66 | term_structure_momentum__missing | term_structure_momentum | quality_flag | -0.0086 | 0.4809 | 0.0034 | 0.0715 | 0.0476 | -0.1684 | 0.0051 | 0.5085 | 0.0034 | 0.0476 | -0.0268 | -0.3753 | -0.0470 | -0.6572 | -0.0974 | -1.3621 | 293 | 6 | 4 |
| 67 | csmom_60__csz | csmom_60 | predictor | -0.0087 | 0.4972 | -0.0289 | 0.1166 | -0.2481 | -0.2830 | -0.1880 | 0.4744 | -0.0289 | -0.2481 | -0.0592 | -0.5074 | -0.0793 | -0.6804 | -0.1297 | -1.1126 | 293 | 6 | 3 |
| 68 | carry_change_20__csz | carry_change_20 | predictor | -0.0090 | 0.4766 | -0.0433 | 0.0854 | -0.5067 | -0.3019 | -0.2389 | 0.4505 | -0.0433 | -0.5067 | -0.0735 | -0.8606 | -0.0937 | -1.0965 | -0.1441 | -1.6863 | 293 | 6 | 0 |
| 69 | multi_horizon_trend__csz | multi_horizon_trend | predictor | -0.0095 | 0.4952 | 0.0002 | 0.0856 | 0.0029 | -0.2237 | -0.0196 | 0.4881 | 0.0002 | 0.0029 | -0.0300 | -0.3502 | -0.0502 | -0.5856 | -0.1006 | -1.1741 | 293 | 6 | 4 |
| 70 | oi_growth_20__csz | oi_growth_20 | conditional | -0.0105 | 0.4634 | -0.0412 | 0.0824 | -0.4994 | -0.3216 | -0.2282 | 0.4642 | -0.0412 | -0.4994 | -0.0714 | -0.8663 | -0.0916 | -1.1109 | -0.1420 | -1.7224 | 293 | 6 | 1 |
| 71 | term_structure_acceleration__missing | term_structure_acceleration | quality_flag | -0.0124 | 0.4775 | 0.0054 | 0.0723 | 0.0746 | -0.1560 | 0.0164 | 0.5256 | 0.0054 | 0.0746 | -0.0248 | -0.3435 | -0.0450 | -0.6222 | -0.0954 | -1.3190 | 293 | 6 | 3 |
| 72 | low_volatility_20__csz | low_volatility_20 | risk | -0.0127 | 0.4876 | -0.0278 | 0.1293 | -0.2149 | -0.2856 | -0.1896 | 0.5085 | -0.0278 | -0.2149 | -0.0580 | -0.4487 | -0.0782 | -0.6046 | -0.1286 | -0.9943 | 293 | 6 | 1 |
| 73 | inventory_surprise__missing | inventory_surprise | quality_flag | -0.0133 | 0.4542 | -0.0164 | 0.0546 | -0.3006 | -0.1565 | -0.0988 | 0.4778 | -0.0164 | -0.3006 | -0.0467 | -0.8544 | -0.0668 | -1.2236 | -0.1172 | -2.1466 | 293 | 6 | 1 |
| 74 | seasonality_3y__csz | seasonality_3y | predictor | -0.0138 | 0.4862 | 0.0063 | 0.0863 | 0.0730 | -0.1734 | 0.0152 | 0.5392 | 0.0063 | 0.0730 | -0.0239 | -0.2774 | -0.0441 | -0.5109 | -0.0945 | -1.0949 | 293 | 6 | 2 |
| 75 | term_structure_slope__missing | term_structure_slope | quality_flag | -0.0145 | 0.4680 | -0.0028 | 0.0693 | -0.0409 | -0.1686 | -0.0299 | 0.4949 | -0.0028 | -0.0409 | -0.0331 | -0.4772 | -0.0532 | -0.7681 | -0.1036 | -1.4954 | 293 | 6 | 2 |
| 76 | inventory_scarcity__missing | inventory_scarcity | quality_flag | -0.0165 | 0.4626 | -0.0277 | 0.0574 | -0.4823 | -0.1777 | -0.1568 | 0.4881 | -0.0277 | -0.4823 | -0.0579 | -1.0091 | -0.0781 | -1.3603 | -0.1285 | -2.2383 | 293 | 6 | 2 |
| 77 | inventory_change_5__missing | inventory_change_5 | quality_flag | -0.0183 | 0.4571 | -0.0273 | 0.0643 | -0.4247 | -0.1849 | -0.1569 | 0.4676 | -0.0273 | -0.4247 | -0.0575 | -0.8951 | -0.0777 | -1.2088 | -0.1281 | -1.9929 | 293 | 6 | 2 |
| 78 | inventory_price_divergence__missing | inventory_price_divergence | quality_flag | -0.0183 | 0.4571 | -0.0273 | 0.0643 | -0.4247 | -0.1849 | -0.1569 | 0.4676 | -0.0273 | -0.4247 | -0.0575 | -0.8951 | -0.0777 | -1.2088 | -0.1281 | -1.9929 | 293 | 6 | 2 |
| 79 | inventory_acceleration__missing | inventory_acceleration | quality_flag | -0.0187 | 0.4566 | -0.0277 | 0.0642 | -0.4320 | -0.1849 | -0.1590 | 0.4676 | -0.0277 | -0.4320 | -0.0580 | -0.9030 | -0.0781 | -1.2170 | -0.1285 | -2.0021 | 293 | 6 | 2 |
| 80 | carry_annualized__missing | carry_annualized | quality_flag | -0.0205 | 0.4375 | -0.0030 | 0.0555 | -0.0533 | -0.1347 | -0.0258 | 0.5017 | -0.0030 | -0.0533 | -0.0332 | -0.5979 | -0.0534 | -0.9610 | -0.1038 | -1.8688 | 293 | 6 | 1 |
| 81 | carry_change_20__missing | carry_change_20 | quality_flag | -0.0246 | 0.4824 | -0.0024 | 0.0551 | -0.0436 | -0.1284 | -0.0225 | 0.4949 | -0.0024 | -0.0436 | -0.0326 | -0.5928 | -0.0528 | -0.9589 | -0.1032 | -1.8742 | 293 | 6 | 1 |
| 82 | correlation_risk_20_120__csz | correlation_risk_20_120 | risk | -0.0309 | 0.4407 | -0.0414 | 0.0907 | -0.4559 | -0.3118 | -0.2323 | 0.4812 | -0.0414 | -0.4559 | -0.0716 | -0.7893 | -0.0918 | -1.0115 | -0.1422 | -1.5672 | 293 | 6 | 1 |
| 83 | volume_price_divergence_20__missing | volume_price_divergence_20 | quality_flag | -0.0516 | 0.4762 | -0.0037 | 0.0555 | -0.0664 | -0.1383 | -0.0299 | 0.4949 | -0.0037 | -0.0664 | -0.0339 | -0.6116 | -0.0541 | -0.9751 | -0.1045 | -1.8838 | 293 | 6 | 1 |
| 84 | oi_surprise_60__missing | oi_surprise_60 | quality_flag | -0.0577 | 0.4762 | -0.0037 | 0.0555 | -0.0664 | -0.1383 | -0.0299 | 0.4949 | -0.0037 | -0.0664 | -0.0339 | -0.6116 | -0.0541 | -0.9751 | -0.1045 | -1.8838 | 293 | 6 | 1 |
| 85 | value_252__missing | value_252 | quality_flag | -0.0582 | 0.3651 | -0.0082 | 0.0575 | -0.1424 | -0.1611 | -0.0555 | 0.4881 | -0.0082 | -0.1424 | -0.0384 | -0.6688 | -0.0586 | -1.0197 | -0.1090 | -1.8969 | 293 | 6 | 1 |
| 86 | multi_horizon_trend__missing | multi_horizon_trend | quality_flag | -0.0595 | 0.3667 | -0.0080 | 0.0575 | -0.1394 | -0.1602 | -0.0546 | 0.4881 | -0.0080 | -0.1394 | -0.0382 | -0.6657 | -0.0584 | -1.0165 | -0.1088 | -1.8937 | 293 | 6 | 1 |
| 87 | low_volatility_20__missing | low_volatility_20 | quality_flag | -0.0687 | 0.4500 | -0.0037 | 0.0555 | -0.0664 | -0.1383 | -0.0299 | 0.4949 | -0.0037 | -0.0664 | -0.0339 | -0.6116 | -0.0541 | -0.9751 | -0.1045 | -1.8838 | 293 | 6 | 1 |
| 88 | oi_growth_20__missing | oi_growth_20 | quality_flag | -0.0687 | 0.4500 | -0.0037 | 0.0555 | -0.0664 | -0.1383 | -0.0299 | 0.4949 | -0.0037 | -0.0664 | -0.0339 | -0.6116 | -0.0541 | -0.9751 | -0.1045 | -1.8838 | 293 | 6 | 1 |
| 89 | price_oi_interaction_20__missing | price_oi_interaction_20 | quality_flag | -0.0687 | 0.4500 | -0.0037 | 0.0555 | -0.0664 | -0.1383 | -0.0299 | 0.4949 | -0.0037 | -0.0664 | -0.0339 | -0.6116 | -0.0541 | -0.9751 | -0.1045 | -1.8838 | 293 | 6 | 1 |
| 90 | realized_volatility_20__missing | realized_volatility_20 | quality_flag | -0.0815 | 0.3500 | -0.0034 | 0.0555 | -0.0604 | -0.1366 | -0.0280 | 0.4949 | -0.0034 | -0.0604 | -0.0336 | -0.6055 | -0.0538 | -0.9689 | -0.1042 | -1.8773 | 293 | 6 | 1 |
| 91 | factor_momentum__missing | factor_momentum | quality_flag | — | — | -0.0040 | 0.0555 | -0.0715 | -0.1397 | -0.0315 | 0.4983 | -0.0040 | -0.0715 | -0.0342 | -0.6162 | -0.0544 | -0.9794 | -0.1048 | -1.8872 | 293 | 6 | 1 |
| 92 | factor_momentum__tsz | factor_momentum | conditional | — | — | -0.0040 | 0.0555 | -0.0715 | -0.1397 | -0.0315 | 0.4983 | -0.0040 | -0.0715 | -0.0342 | -0.6162 | -0.0544 | -0.9794 | -0.1048 | -1.8872 | 293 | 6 | 1 |

## 模型与策略口径

- 模型：每个注册字段单独作为横截面排序信号；不做多因子回归。
- 标签：`future_return_5d`；对照目标 `target_cs_rank_5d` 不参与选因子。
- 切分：`fold_definitions.csv` 扩展窗口；净化 `label_end_date_5d < valid_start`。
- 持有/调仓 5 日；多空各 20%。
- 成本敏感性：单边 0/3/5/10 bp（主口径 5 bp）。

## 使用边界

1. 点时：因子在 T 日收盘后形成，信号仅可用于 T+1 开盘交易。
2. 本模型是单字段排序，**不是**多因子加权或回归拟合结果。
3. 成本仅为单边 0/3/5/10 基点敏感性，不得解读为已核验真实手续费。
4. 2025 未参与主评价。
5. quality_flag / `__missing` 字段可出现在全量表中，解读时需与 predictor 区分。
